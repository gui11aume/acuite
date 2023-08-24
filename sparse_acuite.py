import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn.functional as F

from misc_acuite import (
      ZeroInflatedNegativeBinomial,
      warmup_scheduler,
      sc_data,
      read_sparse_matrix,
)

global K # Number of modules / set by user.
global B # Number of batches / from data.
global C # Number of types / from data.
global R # Number of groups / from data.
global G # Number of genes / from data.

DEBUG = False
SUBSMPL = 1024 if DEBUG else 256
#NUM_SAMPLES = 200

# Use only for debugging.
pyro.enable_validation(DEBUG)



# Helper functions.
def subset(tensor, idx):
   if tensor is None: return None
   return tensor.index_select(0, idx.to(tensor.device))



def model(data, generate=False, train_globals=True):

   ctype, batch, group, label, X, mask = data

   # Format observed labels for Dirichlet. Create a
   # one-hot encoding with label smoothing.
   ohl = F.one_hot(label, num_classes=K).to(X.dtype)
   lab = ((.99-.01/(K-1)) * ohl + .01/(K-1)).view(-1,1,K)

   # Make sure mask has proper dimension.
   mask = mask.view(-1,1)

   device = generate.device if X is None else X.device
   ncells = X.shape[0] if X is not None else generate.shape[0]

   subsample_size = None if ncells < SUBSMPL else SUBSMPL


   # Variance-to-mean ratio. Variance is modelled as
   # 's * u', where 'u' is mean gene expression and
   # 's' is a positive number with 90% chance of being
   # in the interval 1 + (0.3, 200).
   s = 1. + pyro.sample(
         name = "s",
         # dim: (P) x 1 x 1 | .
         fn = dist.LogNormal(
            2. * torch.ones(1,1).to(device),
            2. * torch.ones(1,1).to(device)
         )
   )

   # Zero-inflation factor 'pi'. The median is set at 
   # 0.15, with 5% chance that 'pi' is less than 0.01
   # and 5% chance that 'pi' is more than 50%.
   pi = pyro.sample(
         # dim: (P) x 1 x 1 | .
         name = "pi",
         fn = dist.Beta(
            1. * torch.ones(1,1).to(device),
            4. * torch.ones(1,1).to(device)
         )
   )

   # When there is only one module, there is no need to
   # compute latent Dirichlet allocation (the model is
   # degenerate and has only one case).
   with pyro.plate("K", K, dim=-1):

      # Weight of the transcriptional modules, showing
      # the representation of each module in the
      # transcriptomes of the cells. This is the same
      # prior as in the standard latent Dirichlet
      # allocation.
      alpha = pyro.sample(
            name = "alpha",
            # dim(alpha): (P x 1) x K
            fn = dist.Gamma(
               torch.ones(1).to(device) / K,
               torch.ones(1).to(device)
            )
      )


   # Per-gene sampling.
   with pyro.plate("G", G, dim=-1):

      # The baseline varies over batches and cell types.
      # the correlation is stronger between batches than
      # between cell types. As above, the average
      # expression in log-space has a Gaussian
      # distribution, but it has mean 1 and standard
      # deviation 3, so there is a 90% chance that a gene
      # has between 0 and 400 reads in standard form (i.e,
      # without considering sequencing depth).
      # Between batches, there is a 90% chance that the
      # responses for the same gene are within 25% of each
      # other. Between cells, there is a 90% chance that
      # the responses are within a factor 2 of each other.

      # 'cor_CB':
      #  | 1.00  0.99 0.99 |  0.97  0.97 ...
      #  | 0.99  1.00 0.99 |  0.97  0.97 ...
      #  ----------------------------
      #  | 0.97  0.97 0.99 |  1.00  0.99 ...
      #  | 0.97  0.97 0.99 |  0.99  1.00 ...
      #   ------  B  -----
      #   -----------  C blocks  -----------

      # Intermediate to construct the covariance matrix.
      B_block = (.02 * torch.ones(B,B))
      B_block.fill_diagonal_(.03)

      cor_CB = .97 + torch.block_diag(*([B_block]*C)).to(device)
      mu_CB = torch.ones(1,C*B).to(device)

      base = pyro.sample(
            name = "base",
            # dim(base): (P x 1) x G | C*B
            fn = dist.MultivariateNormal(
               1 * mu_CB, # dim:   1 x C*B 
               4 * cor_CB # dim: C*B x C*B
            )
      )

      # dim(base): G x C x B
      base = base.view(base.shape[:-1] + (C,B))


      with pyro.plate("KRxG", K*R, dim=-2):

         mix = torch.distributions.Categorical(
               torch.tensor([.9, .1]).to(device)
         )
         components = torch.distributions.Normal(
               torch.tensor([[[.00, 0.]]]).to(device),
               torch.tensor([[[.03, 2.]]]).to(device)
         )
         mod = pyro.sample(
               name = "modules",
               # dim(modules): (P) x K*R x G
               fn = dist.MixtureSameFamily(mix, components)
         )

#         mod = pyro.sample(
#               name = "modules",
#               # dim(modules): (P) x K*R x G
#               fn = dist.Normal(
#                  .0 * torch.zeros(1,1,1).to(device),
#                  .7 * torch.ones(1,1,1).to(device)
#               )
#         )

      # dim(mod): (P) x K x R x G
      mod = mod.view(mod.shape[:-2] + (K,R,G))


   with pyro.plate("ncells", ncells, dim=-2,
           subsample_size=subsample_size, device=device) as indx_n:

      # Proportion of the modules in the transcriptomes.
      # This is the same hierarchic model as the standard
      # latent Dirichlet allocation.
      theta = pyro.sample(
            name = "theta",
            # dim(theta): (P) x ncells x 1 | K
            fn = dist.Dirichlet(
               alpha.unsqueeze(-2) # dim: (P x 1) x K
            ),
            obs = subset(lab, indx_n),
            obs_mask = subset(mask, indx_n)
      )

      # Correction for the total number of reads in the
      # transcriptome. The shift in log space corresponds
      # to a cell-specific scaling of all the genes in
      # the transcriptome. In linear space, the median
      # is 1 by design (average 0 in log space).
      shift = pyro.sample(
            name = "shift",
            # dim(shift): (P) x ncells x 1 | .
            fn = dist.Cauchy(
               0. * torch.zeros(1,1).to(device),
               1. * torch.ones(1,1).to(device)
            )
      )

      # dim(base_n): (P) x ncells x G
      c_indx = subset(ctype, indx_n)
      b_indx = subset(batch, indx_n)
      base_n = base[...,c_indx,b_indx].squeeze().transpose(-2,-1)

      # dim(ohg): ncells x R
      ohg = subset(F.one_hot(group).to(mod.dtype), indx_n)

      # dim(mod_n): (P) x ncells x G
      mod_n = torch.einsum("...noK,...KRG,nR->...nG", theta, mod, ohg)

      # dim(u): (P) x ncells x G
      u = torch.exp(base_n + mod_n + shift)


      # Parameter 'u' is the average number of reads in the cell
      # and the variance is 's x u'. Parametrize 'r' and 'p' (the
      # values required for the negative binomial) as a function
      # of 'u' and 's'.
      
      p_ = 1. - 1. / s
      r_ = u / (s - 1)

   # ---------------------------------------------------------------
   # NOTE: if the variance is assumed to vary with a power 'a', i.e.
   # the variance is 's x u^a', then the equations above become:
   # p_ = 1. - 1. / (s*u^(a-1))
   # r_ = u / (s*u^(a-1) - 1)
   # ---------------------------------------------------------------

      # Make sure that parameters of the ZINB are valid. 'p'
      # must be in (0,1) and 'r' must be positive. Choose a
      # small number 'eps' to keep the values away from the
      # boundaries.

      eps = 1e-6
      p = torch.clamp(p_, min=0.+eps, max=1.-eps)
      r = torch.clamp(r_, min=0.+eps)

      with pyro.plate("ncellsxG", G, dim=-1):

         # Observations are sampled from a ZINB distribution.
         Y = pyro.sample(
               name = "Y",
               # dim(Y): ncells x G
               fn = ZeroInflatedNegativeBinomial(
                  total_count = r,
                  probs = p,
                  gate = pi
               ),
               obs = subset(X, indx_n),
               # FIXME: allow gene mask.
               # obs_mask = subset(gene_mask, indx_n)
         )

   return Y


def guide(data=None, generate=False, train_globals=True):

   ctype, batch, group, label, X, mask = data
   mask = mask.view(-1,1)

   device = generate.device if X is None else X.device
   ncells = X.shape[0] if X is not None else generate.shape[0]

   subsample_size = None if ncells < SUBSMPL else SUBSMPL


   # Posterior distribution of 's'.
   post_s_loc = pyro.param(
         "post_s_loc", # dim: 1
         lambda: 2 * torch.ones(1).to(device)
   )
   post_s_scale = pyro.param(
         "post_s_scale", # dim: 1
         lambda: 2 * torch.ones(1).to(device),
         constraint = torch.distributions.constraints.positive
   )
   
   post_s = pyro.sample(
         name = "s",
         # dim: (P x 1) x 1 | .
         fn = dist.LogNormal(
            post_s_loc,  # dim: 1
            post_s_scale # dim: 1
         )
   )
   
   # Posterior distribution of 'pi'.
   post_pi_0 = pyro.param(
         "post_pi_0", # dim: 1
         lambda: 1. * torch.ones(1).to(device),
         constraint = torch.distributions.constraints.positive
   )
   post_pi_1 = pyro.param(
         "post_pi_1", # dim: 1
         lambda: 4. * torch.ones(1).to(device),
         constraint = torch.distributions.constraints.positive
   )
   
   post_pi = pyro.sample(
         name = "pi",
         # dim: (P x 1) x 1 | .
         fn = dist.Beta(
            post_pi_0, # dim: 1
            post_pi_1  # dim: 1
         )
   )
   
   
   with pyro.plate("K", K, dim=-1):
   
      post_alpha_param = pyro.param(
            "post_alpha_param", # dim: K
            lambda: torch.ones(K).to(device) / K,
            constraint = torch.distributions.constraints.positive
      )
   
      post_alpha = pyro.sample(
            name = "alpha",
            # dim(alpha): (P x 1) x K | .
            fn = dist.Gamma(
               post_alpha_param, # dim: K
               torch.ones(1).to(device)
            )
      )
   
   
   with pyro.plate("G", G, dim=-1):
   
      # Posterior distribution of 'base'.
      post_base_loc = pyro.param(
            "post_base_loc", # dim: G x C*B
            lambda: 1 * torch.ones(G,C*B).to(device)
      )
      post_base_scale = pyro.param(
            "post_base_scale", # dim: G x 1
            lambda: 3 * torch.ones(G,1).to(device),
            constraint = torch.distributions.constraints.positive
      )
   
      post_base = pyro.sample(
            name = "base",
            # dim: (P x 1) x G | C*B
            fn = dist.Normal(
               post_base_loc,  # dim: G x C*B
               post_base_scale # dim: G x   1
            ).to_event(1)
      )
   
   
      with pyro.plate("KRxG", K*R, dim=-2):
   
         # Posterior distribution of 'mod'.
         post_mod_loc = pyro.param(
               "post_mod_loc", # dim: KR x G
               lambda: 0 * torch.zeros(K*R,G).to(device)
         )
         post_mod_scale = pyro.param(
               "post_mod_scale", # dim: 1 x G
               lambda: .25 * torch.ones(1,G).to(device),
               constraint = torch.distributions.constraints.positive
         )
   
         post_mod = pyro.sample(
               name = "modules",
               # dim: (P) x KR x G
               fn = dist.Normal(
                  post_mod_loc,  # dim: KR x G
                  post_mod_scale # dim:  1 x G
               )
         )


   with pyro.plate("ncells", ncells, dim=-2,
           subsample_size=subsample_size, device=device) as indx_n:

      # Posterior distribution of 'theta'.
      post_theta_param = pyro.param(
            "post_theta_param",
            lambda: torch.ones(ncells,1,K).to(device),
            constraint = torch.distributions.constraints.greater_than(0.5)
      )
      with pyro.poutine.mask(mask=subset(~mask, indx_n)):
         post_theta = pyro.sample(
               name = "theta_unobserved",
               # dim(theta): (P) x ncells x 1 | K
               fn = dist.Dirichlet(
                  subset(post_theta_param, indx_n) # dim: ncells x 1 x K
               )
         )

      # Posterior distribution of 'shift'.
      post_shift_loc = pyro.param(
            "post_shift_loc",
            lambda: 0 * torch.zeros(ncells,1).to(device),
      )
      post_shift_scale = pyro.param(
            "post_shift_scale",
            lambda: 1 * torch.ones(ncells,1).to(device),
            constraint = torch.distributions.constraints.positive
      )
      post_shift = pyro.sample(
            name = "shift",
            # dim: (P) x ncells x 1
            fn = dist.Normal(
               subset(post_shift_loc, indx_n),  # dim: ncells x 1
               subset(post_shift_scale, indx_n) # dim: ncells x 1
            )
      )



if __name__ == "__main__":

   pyro.set_rng_seed(123)
   torch.manual_seed(123)

   K = int(sys.argv[1])
   in_path = sys.argv[2]
   out_path = sys.argv[3]

   # Optionally specify device through command line.
   device = "cuda" if len(sys.argv) == 4 else sys.argv[4]


   # Read in the data.
   X = read_sparse_matrix(in_path).to(device)

   ctype = torch.zeros(X.shape[0]).to(device)
   batch = torch.zeros(X.shape[0]).to(device)
   group = torch.zeros(X.shape[0]).to(device)
   label = torch.zeros(X.shape[0]).to(device)
   mask = torch.ones(X.shape[0]).to(device)

   import pdb; pdb.set_trace()

   # Set the dimensions.
   B = int(batch.max() + 1)
   C = int(ctype.max() + 1)
   R = int(group.max() + 1)
   G = int(X.shape[-1])

   data = (ctype, batch, group, label, X, mask)

   # Use a warmup/decay learning-rate scheduler.
   scheduler = pyro.optim.PyroLRScheduler(
         scheduler_constructor = warmup_scheduler,
         optim_args = {
            "optimizer": torch.optim.AdamW,
            "optim_args": {"lr": 0.01}, "warmup": 200, "decay": 4000,
         },
         clip_args = {"clip_norm": 5.}
   )

   pyro.clear_param_store()

   ELBO = pyro.infer.Trace_ELBO if DEBUG else pyro.infer.JitTrace_ELBO

   svi = pyro.infer.SVI(
      model = model,
      guide = guide,
      optim = scheduler,
      loss = ELBO(
         num_particles = 16,
         vectorize_particles = True,
         max_plate_nesting = 2,
         ignore_jit_warnings = True,
      ),
   )

   loss = 0.
   for step in range(4000):
      loss += svi.step(data, train_globals=True)
      scheduler.step()
      # Print progress on screen every 500 steps.
      if (step+1) % 500 == 0:
         sys.stderr.write(f"iter {step+1}: loss = {round(loss/1e9,3)}\n")
         loss = 0.

   # Save model parameters.
   pyro.get_param_store().save(out_path)

#   # Posterior predictive sampling.
#   predictive = pyro.infer.Predictive(
#         model = model,
#         guide = guide,
#         num_samples = NUM_SAMPLES,
#         return_sites = ("_RETURN",),
#   )
#   with torch.no_grad():
#      # Resample transcriptome (and "smoothed" estimates as well).
#      sim = predictive(
#            data = (batch, ctype, label, None, mask),
#            generate = X
#      )
#
#   smpl = {
#         "tx": sim["_RETURN"][:,0,:,:].cpu(),
#         "sm": sim["_RETURN"][:,1,:,:].cpu(),
#   }
#
#   # Save model and posterior predictive samples.
#   torch.save({"params":params, "smpl":smpl}, out_fname)

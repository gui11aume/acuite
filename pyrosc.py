import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn.functional as F

from misc_pyrosc import (
      ZeroInflatedNegativeBinomial,
      warmup_scheduler,
      sc_data,
)

global K # Number of modules / set by user.
global B # Number of batches / from data.
global C # Number of types / from data.
global L # Number of labels / from data.
global G # Number of genes / from data.

DEBUG = False
#NUM_SAMPLES = 200

# Use only for debugging.
pyro.enable_validation(DEBUG)


def subset(tensor, idx):
   if tensor is None: return None
   return tensor.index_select(0, idx.to(tensor.device))


def model(data, generate=False):

   batch, ctype, label, X, masks = data
   label_mask, gene_mask = masks

   # Make sure labels (and masks) have proper dimension.
   label = label.view(-1,1)
   label_mask = label_mask.view(-1,1)

   if gene_mask.all(): gene_mask = None
   if label_mask.all(): label_mask = None

   device = generate.device if X is None else X.device
   ncells = X.shape[0] if X is not None else generate.shape[0]


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
   if K > 1:
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
               3 * cor_CB # dim: C*B x C*B
            )
      )

      # dim(base): G x C x B
      base = base.view(base.shape[:-1] + (C,B))


      with pyro.plate("KLxG", K*L, dim=-2):

         mod = pyro.sample(
               name = "modules",
               # dim(modules): (P) x K*L x G
               fn = dist.Normal(
                  0.0 * torch.zeros(1,1,1).to(device),
                  0.7 * torch.ones(1,1,1).to(device)
               )
         )

      # dim(mod): (P) x K x L x G
      mod = mod.view(mod.shape[:-2] + (K,L,G))


   with pyro.plate("ncells", ncells, dim=-2, subsample_size=256) as indx_n:

      # See comment above about not running latent Dirichlet
      # allocation when there is only one module.
      if K > 1:
          # Proportion of the modules in the transcriptomes.
          # This is the same hierarchic model as the standard
          # latent Dirichlet allocation.
          theta = pyro.sample(
                name = "theta",
                # dim(theta): (P) x ncells x 1 | K
                fn = dist.Dirichlet(
                   alpha.unsqueeze(-2) # dim: (P x 1) x K
                )
          )

      # Correction for the total number of reads in the
      # transcriptome. The shift in log space corresponds
      # to a cell-specific scaling of all the genes in
      # the transcriptome. In linear space, the median
      # is 1 by design (average 0 in log space). In
      # linear space, the scaling factor has a 90% chance
      # of being in the window (0.2, 5).
      shift = pyro.sample(
            name = "shift",
            # dim(shift): (P) x ncells x 1 | .
            fn = dist.Normal(
               0. * torch.zeros(1,1).to(device),
               1. * torch.ones(1,1).to(device)
            )
      )

      # The dimensions of the 'lab' tensor are different
      # if all the labels are known (deterministic) vs some
      # of them need to be imputed (random). From here on
      # we need to distinguish the cases for computations.
      impute_labels = label_mask is not None

      lab = pyro.sample(
            name = "label",
            # dim: L x (P) x ncells x 1 | . if impute_labels is True
            # dim:     (P) x ncells x 1 | . if impute_labels is False
            fn = dist.Categorical(
                torch.ones(1,1,L).to(device),
            ),
            obs = subset(label, indx_n),
            obs_mask = subset(label_mask, indx_n)
      )

      # dim(base_n): (P) x ncells x G
      c_indx = subset(ctype, indx_n)
      b_indx = subset(batch, indx_n)
      base_n = base[...,c_indx,b_indx].squeeze().transpose(-2,-1)

      # dim(ohl):  L  x ncells x L if impute_labels is True
      # dim(ohl): (P) x ncells x L if impute_labels is False
      ohl = F.one_hot(lab.squeeze()).to(mod.dtype)

      if impute_labels:
         # dim(mod_n): L x (P) x ncells x G
         mod_n = torch.einsum("...olG,Lnl->L...nG", mod, ohl) if K == 1 else \
                 torch.einsum("...noK,...KlG,Lnl->L...nG", theta, mod, ohl)
         
         # dim(base_n): 1 x (P) x ncells x G
         base_n = base_n.unsqueeze(0)

         # dim(shift): 1 x (P) x ncells x 1
         shift = shift.unsqueeze(0)

         # dim(s): 1 x (P) x 1 x 1
         s = s.unsqueeze(0)
      else:
         # dim(mod_n): (P) x ncells x G
         mod_n = torch.einsum("...oLG,...nL->...nG", mod, ohl) if K == 1 else \
                 torch.einsum("...noK,...KLG,...nL->...nG", theta, mod, ohl)

      # dim(u): L x (P) x ncells x G if impute_labels is True
      # dim(u):     (P) x ncells x G if impute_labels is False
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
               obs_mask = subset(gene_mask, indx_n)
         )

   return Y


def guide(data=None, generate=False):

   batch, ctype, label, X, masks = data
   label_mask, gene_mask = masks

   label = label.view(-1,1)
   label_mask = label_mask.view(-1,1)

   if gene_mask.all(): gene_mask = None
   if label_mask.all(): label_mask = None

   device = generate.device if X is None else X.device
   ncells = X.shape[0] if X is not None else generate.shape[0]

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


   if K > 1:
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


      with pyro.plate("KLxG", K*L, dim=-2):

         # Posterior distribution of 'mod'.
         post_mod_loc = pyro.param(
               "post_mod_loc", # dim: KL x G
               lambda: 0 * torch.zeros(K*L,G).to(device)
         )
         post_mod_scale = pyro.param(
               "post_mod_scale", # dim: 1 x G
               lambda: .25 * torch.ones(1,G).to(device),
               constraint = torch.distributions.constraints.positive
         )

         post_mod = pyro.sample(
               name = "modules",
               # dim: (P) x KL x G
               fn = dist.Normal(
                  post_mod_loc,  # dim: KL x G
                  post_mod_scale # dim:  1 x G
               )
         )


   with pyro.plate("ncells", ncells, dim=-2, subsample_size=256) as indx_n:

      if K > 1:
         # Posterior distribution of 'theta'.
         post_theta_param = pyro.param(
               "post_theta_param",
               lambda: torch.ones(ncells,1,K).to(device),
               constraint = torch.distributions.constraints.greater_than(0.5)
         )
         post_theta = pyro.sample(
               name = "theta",
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

      # Posterior distribution of unobserved labels.
      post_label_param = pyro.param(
            "post_label_param", # dim: ncells x 1 x L
            lambda: 1. * torch.ones(ncells,1,L).to(device),
            constraint = torch.distributions.constraints.simplex
      )
      with pyro.poutine.mask(mask=~subset(label_mask, indx_n)):
         post_lab_unobserved = pyro.sample(
               name = "label_unobserved",
               # dim: L x (P) x ncells x 1 | .
               fn = dist.Categorical(
                   subset(post_label_param, indx_n),
               ),
               infer = { "enumerate": "parallel" }
         )


if __name__ == "__main__":

   pyro.set_rng_seed(123)
   torch.manual_seed(123)

   K = int(sys.argv[1])
   in_fname = sys.argv[2]
   out_fname = sys.argv[3]

   # Optionally specify device through command line.
   device = "cuda" if len(sys.argv) == 4 else sys.argv[4]


   # Read in the data.
   _, ctype, batch, label, X, label_mask = data = sc_data(in_fname)

   # Move data to device.
   ctype = ctype.to(device)
   batch = batch.to(device)
   label = label.to(device)
   X = X.to(device)

   gene_mask = torch.ones_like(X).to(dtype=torch.bool)
   gene_mask[X.isnan()] = False
   X[X.isnan()] = 0

   label_mask = label_mask.to(device)

   # Set the dimensions.
   B = int(batch.max() + 1)
   C = int(ctype.max() + 1)
   L = int(label.max() + 1)
   G = int(X.shape[-1])

   data = batch, ctype, label, X, (label_mask, gene_mask)

   # Use a warmup/decay learning-rate scheduler.
   scheduler = pyro.optim.PyroLRScheduler(
         scheduler_constructor = warmup_scheduler,
         optim_args = {
            "optimizer": torch.optim.AdamW,
            "optim_args": {"lr": 0.01}, "warmup": 300, "decay": 3000,
         },
         clip_args = {"clip_norm": 5.}
   )

   pyro.clear_param_store()

   ELBO = pyro.infer.TraceEnum_ELBO if DEBUG else pyro.infer.JitTraceEnum_ELBO

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
   for step in range(3000):
      loss += svi.step(data)
      scheduler.step()
      # Print progress on screen every 500 steps.
      if (step+1) % 500 == 0:
         sys.stderr.write(f"iter {step+1}: loss = {round(loss/1e9,3)}\n")
         loss = 0.

   # Model parameters.
   names = (
      "post_s_loc", "post_s_scale",
      "post_pi_0", "post_pi_1",
      "post_base_loc", "post_base_scale",
      "post_shift_loc", "post_shift_scale",
      "post_label_param",
   )
   if K > 1:
      names += (
         "post_alpha_param", "post_theta_param",
      )

   ready = lambda x: x.detach().cpu().squeeze()
   params = { name: ready(pyro.param(name)) for name in names }

   torch.save({"params":params}, out_fname)

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

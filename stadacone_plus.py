import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn.functional as F

from misc_stadacone import (
      xSVI,
      ZeroInflatedNegativeBinomial,
      warmup_and_linear,
      sc_data,
      read_info,
      read_sparse_matrix,
)

global K # Number of modules / set by user.
global B # Number of batches / from data.
global C # Number of types / from data.
global R # Number of groups / from data.
global G # Number of genes / from data.

DEBUG = False
SUBSMPL = 256
NUM_PARTICLES = 8
CELL_COVERAGE = 150

CORR_BETWEEN_CTYPES = 0.8

# Use only for debugging.
pyro.enable_validation(DEBUG)


class Stadacone(torch.nn.Module):

   def __init__(self, data, inference_mode = True):
      super().__init__()

      # Store parameters of the priors. Users will eventually
      # be able to modify them.
      self.corr_between_ctypes = CORR_BETWEEN_CTYPES

      # Unpack data.
      self.ctype, self.batch, self.group, self.label, self.X, masks = data
      self.cmask, self.lmask, self.gmask = masks
   
      self.ctype = self.ctype.view(-1,1)
      self.cmask = self.cmask.view(-1,1)
      self.lmask = self.lmask.view(-1,1)
   
      self.device = self.X.device
      self.ncells = self.X.shape[0]
   
      self.subsample_size = None if self.ncells < SUBSMPL else SUBSMPL

      # Format observed labels. Create one-hot encoding with label smoothing.
      oh = F.one_hot(self.label, num_classes=K).to(self.X.dtype)
      self.smooth_lab = ((.99-.01/(K-1)) * oh + .01/(K-1)).view(-1,1,K) if K > 1 else 0.

      self.trace_elbo = pyro.infer.TraceEnum_ELBO(
         num_particles = NUM_PARTICLES,
         vectorize_particles = True,
         max_plate_nesting = 2,
         ignore_jit_warnings = True,
      )

      # Model parts.
      self.output_base = self.sample_base
      self.output_wiggle = self.sample_wiggle
      self.output_batch_fx = self.sample_batch_fx
      self.output_mod = self.sample_mod
      self.output_c_indx = self.sample_c_indx
      self.output_theta = self.sample_theta
      self.output_shift_n = self.sample_shift_n
      self.output_base_n = self.compute_base_n_enum
      self.output_batch_fx_n = self.compute_batch_fx_n
      self.output_mod_n = self.compute_mod_n
      self.output_x_i = self.compute_ELBO_rate_n

      # Guide parts.
      self.output_post_base = self.sample_post_base
      self.output_post_wiggle = self.sample_post_wiggle
      self.output_post_batch_fx = self.sample_post_batch_fx
      self.output_post_mod = self.sample_post_mod
      self.output_post_c_indx = self.sample_post_c_indx
      self.output_post_log_theta = self.sample_post_log_theta
      self.output_post_shift_n = self.sample_post_shift_n
      self.output_post_latent_x_i = self.zero

      # No modules.
      if K < 2:
         self.output_mod = self.zero
         self.output_theta = self.zero
         self.output_mod_n = self.zero
         self.output_post_mod = self.zero
         self.output_post_log_theta = self.zero

      # All cell types known.
      if cmask.all():
         # No need for enumeration.
         self.trace_elbo = pyro.infer.Trace_ELBO(
            num_particles = NUM_PARTICLES,
            vectorize_particles = True,
            max_plate_nesting = 2,
            ignore_jit_warnings = True,
         )
         # In this case, `c_indx` is just `ctype`.
         self.output_c_indx = self.subset_c_indx
         self.output_post_c_indx = self.zero
         self.output_base_n = self.compute_base_n_no_enum
     
      # Generation.
      if inference_mode is False:
         self.output_x_i = self.sample_rate_n
         self.output_post_latent_x_i = self.sample_post_rate_n


   def capture_params(self):
      with pyro.poutine.trace(param_only=True) as param_capture:
         self.trace_elbo.differentiable_loss(self.model, self.guide)
      params = set(site["value"].unconstrained()
             for site in param_capture.trace.nodes.values())
      return params
   
   def configure_optimizer(self):
      # Estimate number of steps.
      self.nsteps = max(4000, CELL_COVERAGE * int(self.ncells / SUBSMPL))
      self.warmup_steps = int(.05 * self.nsteps)
      self.decay_steps = int(.95 * self.nsteps)
      # Configure optimizer.
      params = self.capture_params()
      optimizer = torch.optim.Adam(params, lr=0.01)
      # Configure scheduler.
      warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.01,
            end_factor = 1.,
            total_iters = self.warmup_steps)
      decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 1.,
            end_factor = 0.01,
            total_iters = self.decay_steps)
      self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer = optimizer,
            schedulers = [warmup, decay],
            milestones = [self.warmup_steps]
      )
   
   def learn_parameters(self):
      self.configure_optimizer()
      cumloss = 0.
      for step in range(self.nsteps):
          loss = self.trace_elbo.differentiable_loss(self.model, self.guide)
          cumloss += float(loss)
          if loss.isnan():
              import pdb; pdb.set_trace()
              continue
          loss.backward()
          self.scheduler.optimizer.step()
          self.scheduler.optimizer.zero_grad()
          self.scheduler.step()
          if (step+1) % 500 == 0:
             lr = self.scheduler.get_last_lr()
             sys.stderr.write(
                f"iter {step+1}/{self.nsteps}: loss = {cumloss/500}, lr = {lr}\n"
             )
             cumloss = 0.
      # Save model parameters.
      pyro.get_param_store().save(out_path)


   #  == Helper functions == #
   def zero(self, *args, **kwargs):
      return 0.

   def subset(self, tensor, idx):
      if tensor is None: return None
      return tensor.index_select(0, idx.to(tensor.device))

   #  ==  Model parts == #
   def sample_base(self): 
        
      # Construct the matrix `cor_C` below, where
      # c is `self.corr_between_ctypes`

      #  | 1.0  c   c  ...
      #  |  c  1.0  c  ...
      #  |  c   c   c  ...

      # Intermediate to construct the covariance matrix.
      C_block = (self.corr_between_ctypes * torch.ones(C,C))
      C_block.fill_diagonal_(1.0)
      # Define parameters of the multivariate t.
      degf_C = torch.ones(1).to(self.device)
      mean_C = torch.zeros(1,C).to(self.device)
      corr_C = C_block.to(self.device)
      base = pyro.sample(
            name = "base",
            # dim(base): (P x 1) x G | C
            fn = dist.MultivariateStudentT(
               1.5 * degf_C, # dim:     1
               0.0 * mean_C, # dim: G x C
               1.0 * corr_C  # dim: C x C
            )
      )
      return base

   def sample_post_base(self):
      post_base_loc = pyro.param(
            "post_base_loc", # dim: G x C
            lambda: 1 * torch.ones(G,C).to(self.device)
      )
      post_base_scale = pyro.param(
            "post_base_scale", # dim: G x 1
            lambda: .3 * torch.ones(G,1).to(self.device),
            constraint = torch.distributions.constraints.positive
      )
      post_base = pyro.sample(
            name = "base",
            # dim: (P x 1) x G | C
            fn = dist.Normal(
               post_base_loc,  # dim: G x C
               post_base_scale # dim: G x 1
            ).to_event(1),
      )
      return post_base

   def sample_wiggle(self):
      wiggle = pyro.sample(
            name = "wiggle",
            # dim(wiggle): (P x 1) x G
            fn = dist.LogNormal(
                -2.0 * torch.ones(1).to(self.device),
                 1.2 * torch.ones(1).to(self.device)
            ),
      )
      return wiggle

   def sample_post_wiggle(self):
      post_wiggle_loc = pyro.param(
            "post_wiggle_loc", # dim: G
            lambda: .0 * torch.ones(G).to(self.device)
      )
      post_wiggle_scale = pyro.param(
            "post_wiggle_scale", # dim: G
            lambda: .1 * torch.ones(G).to(self.device),
            constraint = torch.distributions.constraints.positive
      )
      post_wiggle = pyro.sample(
            name = "wiggle",
            # dim(base): (P x 1) x G
            fn = dist.LogNormal(
                post_wiggle_loc,
                post_wiggle_scale
            ),
      )
      return post_wiggle

   def sample_batch_fx(self):
      batch_fx = pyro.sample(
            name = "batch_fx",
            # dim(base): (P) x B x G
            fn = dist.Normal(
               .00 * torch.zeros(1,1).to(self.device), # dim: B x G
               .05 * torch.ones(1,1).to(self.device)   # dim: B x G
            )
      )
      return batch_fx

   def sample_post_batch_fx(self):
      post_batch_fx_loc = pyro.param(
            "post_batch_fx_loc", # dim: B x G
            lambda: .0 * torch.zeros(B,G).to(self.device)
      )
      post_batch_fx_scale = pyro.param(
            "post_batch_fx_scale", # dim: 1 x G
            lambda: .1 * torch.ones(1,G).to(self.device),
            constraint = torch.distributions.constraints.positive
      )
      post_batch_fx = pyro.sample(
            name = "batch_fx",
            # dim(base): (P) x B x G
            fn = dist.Normal(
               post_batch_fx_loc,
               post_batch_fx_scale
            ),
      )
      return post_batch_fx


   def sample_mod(self):
      mod = pyro.sample(
            name = "mod",
            # dim(mod): (P) x K*R x G
            fn = dist.Normal(
               .0 * torch.zeros(1,1,1).to(self.device),
               .7 * torch.ones(1,1,1).to(self.device)
            )
      )
      # dim(mod): (P) x K x R x G
      mod = mod.view(mod.shape[:-2] + (K,R,G))
      return mod

   def sample_post_mod(self):
      post_mod_loc = pyro.param(
            "post_mod_loc", # dim: KR x G
            lambda: 0 * torch.zeros(K*R,G).to(self.device)
      )
      post_mod_scale = pyro.param(
            "post_mod_scale", # dim: 1 x G
            lambda: .1 * torch.ones(1,G).to(self.device),
            constraint = torch.distributions.constraints.positive
      )
      post_mod = pyro.sample(
            name = "mod",
            # dim: (P) x KR x G
            fn = dist.Normal(
               post_mod_loc,  # dim: KR x G
               post_mod_scale # dim:  1 x G
            ),
      )
      return post_mod

   def sample_c_indx(self, ctype, cmask, indx_n):
      c_indx = pyro.sample(
            name = "cell_type",
            # dim(c_indx): C x 1 x ncells x 1 | .
            fn = dist.Categorical(
               torch.ones(1,1,C).to(self.device),
            ),
            infer = {"enumerate": "parallel"},
            obs = self.subset(ctype, indx_n),
            obs_mask = self.subset(cmask, indx_n)
      )
      return c_indx

   def subset_c_indx(self, ctype, cmask, indx_n):
      # dim(c_indx): ncells x 1
      c_indx = self.subset(ctype, indx_n)
      return c_indx

   def sample_post_c_indx(self, cmask, indx_n):
      post_c_indx_param = pyro.param(
            "post_c_indx_param",
            lambda: torch.ones(self.ncells,1,C).to(self.device),
            constraint = torch.distributions.constraints.simplex
      )
      with pyro.poutine.mask(mask=self.subset(~cmask, indx_n)):
         post_c_indx = pyro.sample(
               name = "cell_type_unobserved",
               # dim(c_indx): C x 1 x 1 x 1 | .
               fn = dist.Categorical(
                  self.subset(post_c_indx_param, indx_n) # dim: ncells x 1 x C
               ),
               infer={"enumerate": "parallel"},
         )
         return post_c_indx

   def sample_theta(self, lab, lmask, indx_n):
      log_theta = pyro.sample(
            name = "log_theta",
            # dim(log_theta): (P) x ncells x 1 | K
            fn = dist.Normal(
               torch.zeros(1,1,K).to(self.device),
               torch.ones(1,1,K).to(self.device)
            ).to_event(1),
            obs = self.subset(self.smooth_lab, indx_n),
            obs_mask = self.subset(lmask, indx_n)
      ) 
      # dim(theta): (P) x ncells x 1 x K
      theta = log_theta.softmax(dim=-1)
      return theta

   def sample_post_log_theta(self, lmask, indx_n):
      post_log_theta_loc = pyro.param(
            "post_log_theta_loc",
            lambda: torch.zeros(self.ncells,1,K).to(self.device),
      )
      post_log_theta_scale = pyro.param(
            "post_log_theta_scale",
            lambda: torch.ones(self.ncells,1,K).to(self.device),
            constraint = torch.distributions.constraints.positive
      )
      with pyro.poutine.mask(mask=self.subset(~lmask, indx_n)):
         post_log_theta = pyro.sample(
               name = "log_theta_unobserved",
               # dim(theta): (P) x ncells x 1 | K
               fn = dist.Normal(
                  self.subset(post_log_theta_loc, indx_n),   # dim: ncells x 1 x K
                  self.subset(post_log_theta_scale, indx_n), # dim: ncells x 1 x K
               ).to_event(1)
         )
      return post_log_theta

   def sample_shift_n(self):
      shift_n = pyro.sample(
            name = "shift_n",
            # dim(shift_n): (P) x ncells x 1 | .
            fn = dist.Cauchy(
               0. * torch.zeros(1,1).to(self.device),
               1. * torch.ones(1,1).to(self.device)
            )
      )
      return shift_n

   def sample_post_shift_n(self, indx_n):
      post_shift_n_loc = pyro.param(
            "post_shift_n_loc",
            lambda: 0 * torch.zeros(self.ncells,1).to(self.device),
      )
      post_shift_n_scale = pyro.param(
            "post_shift_n_scale",
            lambda: .1 * torch.ones(self.ncells,1).to(self.device),
            constraint = torch.distributions.constraints.positive
      )
      post_shift_n = pyro.sample(
            name = "shift_n",
            # dim: (P) x ncells x 1
            fn = dist.Normal(
               self.subset(post_shift_n_loc, indx_n),  # dim: ncells x 1
               self.subset(post_shift_n_scale, indx_n) # dim: ncells x 1
            ),
      )
      return post_shift_n

   def compute_base_n_enum(self, c_indx, base):
      # dim(ohc): C x ncells x C
      ohc = F.one_hot(c_indx.squeeze(), num_classes=C).float()
      # dim(base_n): C x (P) x ncells x G
      base_n = torch.einsum("Cnc,...oGc->C...nG", ohc, base)
      return base_n

   def compute_base_n_no_enum(self, c_indx, base):
      # dim(ohc): ncells x C
      ohc = F.one_hot(c_indx.squeeze(), num_classes=C).float()
      # dim(base_n): (P) x ncells x G
      base_n = torch.einsum("nC,...oGC->...nG", ohc, base)
      return base_n

   def compute_batch_fx_n(self, batch, batch_fx, indx_n, dtype):
      # dim(ohg): ncells x B
      ohb = self.subset(F.one_hot(batch).to(dtype), indx_n)
      # dim(batch_fx_n): (P) x ncells x G
      batch_fx_n = torch.einsum("...BG,nB->...nG", batch_fx, ohb)
      return batch_fx_n

   def compute_mod_n(self, group, theta, mod, indx_n):
      # dim(ohg): ncells x R
      ohg = self.subset(F.one_hot(group).to(mod.dtype), indx_n)
      # dim(theta): (P) x ncells x 1 x K
      mod_n = torch.einsum("...noK,...KRG,nR->...nG", theta, mod, ohg)
      # dim(theta): 1 x (P) x ncells x 1 x K
      mod_n = mod_n.unsqueeze(0)
      return mod_n

   def compute_ELBO_rate_n(self, x_i, mu, sg, *args, **kwargs):
      # Parameters `mu` and `sg` are the prior parameters of the Poisson
      # LogNormal distribution. The variational posterior parameters
      # given the observations `x_i` are `mu_i` and `w2_i`. In this case
      # we can compute the ELBO analytically and maximize it with respect
      # to `mu_i` and `w2_i` so as to pass the gradient to `mu` and `sg`.
      # This allows us to compute the ELBO efficiently without having
      # to store parameters and gradients for `mu_i` and `w2_i`.
   
      # Fix parameters by detaching gradient.
      m = mu.detach()
      w2 = torch.square(sg.detach())
      # Initialize `mu_i`.
      mu_i = (x_i * w2 - 1) * torch.ones_like(m)
      # Perform 5 Newton-Raphson iterations.
      for _ in range(5):
         f = m + mu_i + w2 * .5 / (w2 * x_i + 1 - mu_i) - torch.log(x_i - mu_i / w2)
         df = 1 + w2 * .5 / torch.square(w2 * x_i + 1 - mu_i) + 1. / (w2 * x_i - mu_i)
         mu_i = torch.clamp(mu_i - f / df, max = x_i * w2 - 1e-2)
      # Set the optimal `w2_i` from the optimal `mu_i`.
      w2_i = 1. / (x_i + (1 - mu_i) / w2)
   
      # Compute ELBO term as a function of `mu` and `sg`,
      # for which we kept the gradient.
      mini_ELBO = - torch.exp(mu + mu_i + 0.5 * w2_i)      \
                  + (x_i * mu) - torch.log(sg)             \
                  - 0.5 * (mu_i * mu_i + w2_i) / (sg * sg)
   
      pyro.factor("PLN_ELBO_term", mini_ELBO)
      return x_i

   def sample_rate_n(self, x_i, mu, w, gmask):
      rate_n = pyro.sample(
            name = "rate_n",
            fn = dist.LogNormal(
               mu, # dim: C x (P) x ncells x G  /// (P) x ncells x G
               w,  # dim:     (P) x      1 x G
            ),
      )
      x_i = pyro.sample(
            name = "x_i",
            # dim(x_i): ncells x G
            fn = dist.Poisson(
               rate_n # dim: C x (P) x ncells x G  /// (P) x ncells x G
            ),
            obs = x_i,
            obs_mask = gmask
      )
      return x_i

   def sample_post_rate_n(self, indx_n):
      post_rate_n_loc = pyro.param(
            "post_rate_n_loc",
            lambda: 0 * torch.zeros(self.ncells,G).to(device),
      )
      post_rate_n_scale = pyro.param(
            "post_rate_n_scale",
            lambda: .1 * torch.ones(self.ncells,G).to(device),
            constraint = torch.distributions.constraints.positive
      )
      post_rate_n = pyro.sample(
            name = "rate_n",
            fn = dist.LogNormal(
               self.subset(post_rate_n_loc, indx_n),
               self.subset(post_rate_n_scale, indx_n),
            ),
      )
      return post_rate_n


   #  ==  model description == #
   def model(self):


      # Per-gene sampling.
      with pyro.plate("G", G, dim=-1):
   
         # Baselines represent the average expression per gene.
         # The parameters have a multivariate t distribution.
         # The distribution is centered on 0, because only the
         # variations between genes are considered here. The
         # parameters are correlated between cell types with 
         # parameter `self.corr_between_ctypes`. The prior is
         # chosen so that the parameters have a 90% chance of
         # lying in the interval (-3.5, 3.5), i.e., there is a
         # factor 1000 between the bottom 5% and the top 5%.
         # The distribution has a heavy tail, the top 1% is
         # 60,000 times higher than the average.

         # dim(base): (P x 1) x G | C
         base = self.output_base()

         # TODO: describe prior.

         # dim(base): (P x 1) x G
         wiggle = self.output_wiggle()
   
         # Per-batch, per-gene sampling.
         with pyro.plate("BxG", B, dim=-2):
   
            # Batch effects have a Gaussian distribution
            # centered on 0. They are weaker than 8% for
            # 95% of the genes.

            # dim(base): (P) x B x G
            batch_fx = self.output_batch_fx()
   
         # Per-module, per-type, per-gene sampling.
         with pyro.plate("KRxG", K*R, dim=-2):

            # TODO: describe prior.

            # dim(mod): (P) x K x R x G
            mod = self.output_mod()
   
   
      # Per-cell sampling (on dimension -2).
      with pyro.plate("ncells", self.ncells, dim=-2,
         subsample_size=self.subsample_size, device=self.device) as indx_n:
   
         # TODO: describe prior.

         # dim(c_indx): C x 1 x ncells x 1 | .  /// ncells x 1
         c_indx = self.output_c_indx(self.ctype, self.cmask, indx_n)

         # Proportion of the modules in the transcriptomes.
         # TODO: describe prior.

         # dim(theta): (P) x ncells x 1 x K  ///  *
         theta = self.output_theta(self.smooth_lab, self.lmask, indx_n)
   
         # Correction for the total number of reads in the
         # transcriptome. The shift in log space corresponds
         # to a cell-specific scaling of all the genes in
         # the transcriptome. In linear space, the median
         # is 1 by design (average 0 in log space).

         # dim(shift_n): (P) x ncells x 1 | .
         shift_n = self.output_shift_n()
   

         # Deterministic functions to obtain per-cell means.

         # dim(base_n): C x (P) x ncells x G  ///  (P) x ncells x G
         base_n = self.output_base_n(c_indx, base)
   
         # dim(batch_fx_n): (P) x ncells x G
         batch_fx_n = self.output_batch_fx_n(batch, batch_fx, indx_n, base.dtype)
   
         # dim(mod_n): (P) x ncells x G
         mod_n = self.output_mod_n(group, theta, mod, indx_n)


         # Per-cell, per-gene sampling.
         with pyro.plate("ncellsxG", G, dim=-1):
   
            mu = base_n + batch_fx_n + mod_n + shift_n
            x_i = self.subset(self.X, indx_n).to_dense()
            x_i_mask = self.subset(self.gmask, indx_n)

            self.output_x_i(x_i, mu, wiggle, x_i_mask)


   #  ==  guide description == #
   def guide(self):
      
      with pyro.plate("G", G, dim=-1):
      
         # Posterior distribution of `base`.
         post_base = self.output_post_base()

         # Posterior distribution of `w`.
         post_wiggle = self.output_post_wiggle()

         with pyro.plate("BxG", B, dim=-2):
   
            # Posterior distribution of `batch_fx`.
            post_batch_fx = self.output_post_batch_fx()
   
         with pyro.plate("KRxG", K*R, dim=-2):

            # Posterior distribution of `mod`.
            post_mod = self.output_post_mod()
   
   
      with pyro.plate("ncells", self.ncells, dim=-2,
         subsample_size=self.subsample_size, device=self.device) as indx_n:

         # Posterior distribution of `c_indx`.
         post_c_indx = self.output_post_c_indx(self.cmask, indx_n)
   
         # Posterior distribution of `log_theta`.
         post_log_theta = self.output_post_log_theta(self.lmask, indx_n)

         # Posterior distribution of `shift_n`.
         post_shift = self.output_post_shift_n(indx_n)

         with pyro.plate("ncellsxG", G, dim=-1):

             post_rate_n = self.output_post_latent_x_i(indx_n)
   

if __name__ == "__main__":

   pyro.set_rng_seed(123)
   torch.manual_seed(123)

   device = "cuda"

   K = int(sys.argv[1])
   info_path = sys.argv[2]
   expr_path = sys.argv[3]
   out_path = sys.argv[4]

   info = read_info(info_path)

   ctype = info[0].to(device)
   batch = info[1].to(device)
   group = info[2].to(device)
   label = info[3].to(device)
   cmask = info[4].to(device)

   X = torch.load(expr_path)

   lmask = torch.zeros(X.shape[0], dtype=torch.bool).to(device)

   # Set the dimensions.
   B = int(batch.max() + 1)
   C = int(ctype.max() + 1)
   R = int(group.max() + 1)
   G = int(X.shape[-1])


   data = (ctype, batch, group, label, X, (cmask, lmask, None))

   pyro.clear_param_store()
   inference = Stadacone(data)
   inference.learn_parameters()

#   ELBO = pyro.infer.TraceEnum_ELBO if DEBUG else pyro.infer.JitTraceEnum_ELBO
#
#   svi = xSVI(
#      model = inference.model,
#      guide = inference.guide,
#      optim = scheduler,
#      loss = ELBO(
#         num_particles = 8,
#         vectorize_particles = True,
#         max_plate_nesting = 2,
#         ignore_jit_warnings = True,
#      ),
#   )
#
#   sys.stderr.write("starting...\n")
#
#   loss = 0.
#   for step in range(nsteps):
#      loss += svi.step(data)
#      scheduler.step()
#      # Print progress on screen every 500 steps.
#      if (step+1) % 500 == 0:
#         sys.stderr.write(f"iter {step+1}/{nsteps}: loss = {loss}\n")
#         loss = 0.

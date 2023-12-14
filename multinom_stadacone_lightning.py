import lightning.pytorch as pl
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn.functional as F

from pyro.distributions import constraints
from pyro.infer.autoguide import (
      AutoGuideList,
      AutoGuide,
      AutoNormal,
)

from contextlib import ExitStack

from misc_stadacone import (
      xSVI,
      ZeroInflatedNegativeBinomial,
      warmup_and_linear,
      sc_data,
      read_info,
      read_sparse_matrix,
)

global K # Number of units / set by user.
global B # Number of batches / from data.
global C # Number of types / from data.
global R # Number of groups / from data.
global G # Number of genes / from data.


DEBUG = False
SUBSMPL = 256
NUM_PARTICLES = 12
NUM_EPOCHS = 2000

global DEBUG_COUNTER
DEBUG_COUNTER = 0

# Use only for debugging.
pyro.enable_validation(DEBUG)


def subset(tensor, idx):
   if idx is None: return tensor
   if tensor is None: return None
   return tensor.index_select(0, idx.to(tensor.device))


class plTrainHarness(pl.LightningModule):
   def __init__(self, stadacone, lr=.002):
      super().__init__()
      self.stadacone = stadacone
      self.pyro_model = stadacone.model
      self.pyro_guide = stadacone.guide
      self.lr = lr

      if stadacone.need_to_infer_cell_type:
         self.elbo = pyro.infer.TraceEnum_ELBO(
            num_particles = NUM_PARTICLES,
            vectorize_particles = True,
            max_plate_nesting = 2,
            ignore_jit_warnings = True,
         )
      else:
         self.elbo = pyro.infer.Trace_ELBO(
            num_particles = NUM_PARTICLES,
            vectorize_particles = True,
            max_plate_nesting = 2,
            ignore_jit_warnings = True,
         )

      # Auto-instantiate parameters.
      self.capture_params()

   def capture_params(self):
      with pyro.poutine.trace(param_only=True) as param_capture:
         self.elbo.differentiable_loss(
                 model = self.pyro_model,
                 guide = self.pyro_guide,
                 # Params.
                 idx = torch.tensor([0])
         )

   def configure_optimizers(self):
      optimizer = torch.optim.Adam(
          self.trainer.model.parameters(), lr=0.01,
      )

      n_steps = self.trainer.estimated_stepping_batches
      n_warmup_steps = int(0.05 * n_steps)
      n_decay_steps = int(0.95 * n_steps)

      warmup = torch.optim.lr_scheduler.LinearLR(
         optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup_steps
      )
      decay = torch.optim.lr_scheduler.LinearLR(
         optimizer, start_factor=1.0, end_factor=0.01, total_iters=n_decay_steps
      )

      scheduler = torch.optim.lr_scheduler.SequentialLR(
         optimizer=optimizer,
         schedulers=[warmup, decay],
         milestones=[n_warmup_steps],
      )

      return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
   
   def training_step(self, batch, batch_idx):
      idx = batch if self.stadacone.bsz >= SUBSMPL else None
      loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, idx)
      (lr,) = self.lr_schedulers().get_last_lr()
      info = { "loss": loss, "lr": lr }
      self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
      return loss


class cell_autoguide(AutoGuide):
   def __init__(self, pyro_model, device, X, ncells, bsz, cmask):
      super().__init__(model=pyro_model)
      self.X = X
      self.device = device
      self.ncells = ncells
      self.cmask = cmask

   def _setup_prototype(self):
      model = pyro.poutine.block(pyro.infer.enum.config_enumerate(self.model), self._prototype_hide_fn)
      get_trace = pyro.poutine.block(pyro.poutine.trace(model).get_trace)
      self.prototype_trace = get_trace(idx = torch.tensor([0]))
      self._cond_indep_stacks = {}
      self._prototype_frames = {}
      for name, site in self.prototype_trace.iter_stochastic_nodes():
         self._cond_indep_stacks[name] = site["cond_indep_stack"]
         for frame in site["cond_indep_stack"]:
            self._prototype_frames[frame.name] = frame
      pyro.infer.autoguide.utils.deep_setattr(self,
         "post_c_indx_probs",
         pyro.nn.module.PyroParam(
            torch.ones(self.ncells,C).to(self.device),
            constraint = torch.distributions.constraints.simplex,
            event_dim = 1
         )
      )
      pyro.infer.autoguide.utils.deep_setattr(self,
         "post_logits_n_loc",
         pyro.nn.module.PyroParam(
            torch.ones(C,1,1,self.ncells,G).to(self.device),
            event_dim = 1
         )
      )
      pyro.infer.autoguide.utils.deep_setattr(self,
         "post_logits_n_scale",
         pyro.nn.module.PyroParam(
            torch.ones(C,1,1,self.ncells,G).to(self.device),
            constraint = torch.distributions.constraints.positive,
            event_dim = 1
         )
      )

   def forward(self, idx=None):

      if self.prototype_trace is None:
          self._setup_prototype()

      plates = self._create_plates()

      with ExitStack() as stack:
         for frame in self._cond_indep_stacks["cell_type_unobserved"]:
             stack.enter_context(plates[frame.name])

         with pyro.poutine.mask(mask=subset(~self.cmask, idx)):
   
            # Posterior distribution of `c_indx`.
            post_c_indx_probs = pyro.infer.autoguide.utils.deep_getattr(self, "post_c_indx_probs")
            post_c_indx = pyro.sample(
               name = "cell_type_unobserved",
               # dim(c_indx): 1 | C
               fn = dist.OneHotCategorical(
                  post_c_indx_probs # dim: ncells | C
               ),
               infer={ "enumerate": "parallel" },
            )

      with ExitStack() as stack:
         for frame in self._cond_indep_stacks["logits_n"]:
             stack.enter_context(plates[frame.name])

         # Posterior distribution of `logits_n`.
         post_logits_n_loc = pyro.infer.autoguide.utils.deep_getattr(self, "post_logits_n_loc")
         post_logits_n_scale = pyro.infer.autoguide.utils.deep_getattr(self, "post_logits_n_scale")

         post_logits_n = pyro.sample(
               name = "logits_n",
               fn = dist.Normal(
                  post_logits_n_loc,
                  post_logits_n_scale,
               ).to_event(1),
         )

      dictionary = {
          "cell_type_unobserved": post_c_indx,
          "logits_n": post_logits_n,
      }

      return dictionary


class Stadacone(torch.nn.Module):

   def __init__(self, data, amortize = True):
      super().__init__()

      # Unpack data.
      self.ctype, self.batch, self.group, self.label, self.X, masks = data
      self.cmask, self.lmask, self.gmask = masks
   
      self.ctype = F.one_hot(self.ctype, num_classes=C).float()
   
      self.device = self.X.device
      self.ncells = int(self.X.shape[0])
   
      self.bsz = self.ncells if self.ncells < SUBSMPL else SUBSMPL

      # Format observed labels. Create one-hot encoding with label smoothing.
      oh = F.one_hot(self.label, num_classes=K).to(self.X.dtype)
      self.smooth_lab = ((.99-.01/(K-1)) * oh + .01/(K-1)).view(-1,1,K) if K > 1 else 0.

      # 1a) Define core parts of the model.
      self.output_scale_tril_unit = self.sample_scale_tril_unit
      self.output_scale_factor = self.sample_scale_factor
      self.output_log_fuzz_loc = self.sample_log_fuzz_loc
      self.output_log_fuzz_scale = self.sample_log_fuzz_scale
      self.output_global_base = self.sample_global_base
      self.output_fuzz = self.sample_fuzz
      self.output_base = self.sample_base

      # 1b) Define optional parts of the model.
      if K > 1:
         self.need_to_infer_units = True
         self.output_units = self.sample_units
         self.output_theta_n = self.sample_theta_n
         self.output_units_n = self.compute_units_n
      else:
         self.need_to_infer_units = False
         self.output_units = self.zero
         self.output_theta_n = self.zero
         self.output_units_n = self.zero

      if B > 1:
         self.need_to_infer_batch_fx = True
         self.output_batch_fx_scale = self.sample_batch_fx_scale
         self.output_batch_fx = self.sample_batch_fx
         self.output_batch_fx_n = self.compute_batch_fx_n
      else:
         self.need_to_infer_batch_fx = False
         self.output_batch_fx_scale = self.zero
         self.output_batch_fx = self.zero
         self.output_batch_fx_n = self.zero

      if cmask.all():
         self.need_to_infer_cell_type = False
         self.output_c_indx = self.subset_c_indx
         self.output_base_n = self.compute_base_n_no_enum
      else:
         self.need_to_infer_cell_type = True
         self.output_c_indx = self.sample_c_indx
         self.output_base_n = self.compute_base_n_enum
     
      # 2) Define the guide.
      self.guide = AutoGuideList(self.model, create_plates=self.create_ncells_plate)
      cell_variables = ["cell_type_unobserved", "logits_n"]
      self.guide.append(AutoNormal(
          pyro.poutine.block(self.model, hide = cell_variables)
      ))
      if self.need_to_infer_cell_type:
         self.guide.append(cell_autoguide(
                pyro.poutine.block(self.model, expose = cell_variables),
                device = self.device,
                X = self.X,
                ncells = self.ncells,
                bsz = self.bsz,
                cmask = self.cmask,
            ),
         )


   #  == Helper functions == #
   def zero(self, *args, **kwargs):
      return 0.

   def create_ncells_plate(self, idx=None):
      return pyro.plate("ncells", self.ncells, dim=-1,
         subsample=idx, device=self.device)


   #  ==  Model parts == #
   def sample_scale_tril_unit(self):
      scale_tril_unit = pyro.sample(
            name = "scale_tril_unit",
            # dim(scale_tril_unit): (P x 1) x 1 | C x C
            fn = dist.LKJCholesky(
                dim = C,
                concentration = torch.ones(1).to(self.device)
            ),
      )
      return scale_tril_unit

   def sample_scale_factor(self):
      scale_factor = pyro.sample(
            name = "scale_factor",
            # dim(scale_factor): (P x 1) x C
            fn = dist.Exponential(
                rate = torch.ones(1).to(self.device),
            ),
      )
      # dim(scale_factor): (P x 1) x 1 x C
      scale_factor = scale_factor.unsqueeze(-2)
      return scale_factor

   def sample_batch_fx_scale(self):
      batch_fx_scale = pyro.sample(
            name = "batch_fx_scale",
            # dim(base): (P x 1) x B
            fn = dist.Exponential(
               5. * torch.ones(1).to(self.device),
            ),
      )
      return batch_fx_scale

   def sample_log_fuzz_loc(self):
      log_fuzz_loc = pyro.sample(
            name = "log_fuzz_loc",
            # dim(log_fuzz_loc): (P) x 1 x 1
            fn = dist.Normal(
                0.5 * torch.ones(1,1).to(self.device),
                0.7 * torch.ones(1,1).to(self.device)
            ),
      )
      return log_fuzz_loc

   def sample_log_fuzz_scale(self):
      log_fuzz_scale = pyro.sample(
            name = "log_fuzz_scale",
            # dim(log_fuzz_scale): (P) x 1 x 1
            fn = dist.Exponential(
                1. * torch.ones(1,1).to(self.device),
            ),
      )
      return log_fuzz_scale

   def sample_global_base(self): 
      global_base = pyro.sample(
            name = "global_base",
            # dim(global_base): (P) x 1 x G
            fn = dist.StudentT(
               1.5 * torch.ones(1).to(self.device),
               0.0 * torch.zeros(1).to(self.device),
               1.0 * torch.ones(1).to(self.device)
            ),
      )
      return global_base

   def sample_base(self, loc, scale_tril): 
      base_0 = pyro.sample(
            name = "base_0",
            # dim(base_0): (P) x 1 x G | C
            fn = dist.MultivariateNormal(
                torch.zeros(C).to(self.device),
                scale_tril = scale_tril
            ),
      )
      # dim(base): (P) x 1 x G x C
      base = loc.unsqueeze(-1) + base_0
      return base

   def sample_fuzz(self, loc, scale):
      fuzz = pyro.sample(
            name = "fuzz",
            # dim(fuzz): (P) x 1 x G
            fn = dist.LogNormal(loc, scale),
      )
      return fuzz

   def sample_batch_fx(self, scale):
      batch_fx = pyro.sample(
            name = "batch_fx",
            # dim(base): (P) x G x B
            fn = dist.Normal(
               torch.zeros(1,1).to(self.device),
               scale
            ),
      )
      return batch_fx

   def sample_units(self):
      units_KR = pyro.sample(
            name = "units_KR",
            # dim(units_KR): (P) x G x KR
            fn = dist.Normal(
               .0 * torch.zeros(1,1).to(self.device),
               .7 * torch.ones(1,1).to(self.device)
            ),
      )
      # dim(units): (P) x G x K x R
      units = units_KR.view(units_KR.shape[:-2] + (G,K,R))
      return units

   def sample_c_indx(self, ctype_i, ctype_i_mask):
      c_indx = pyro.sample(
            name = "cell_type",
            # dim(c_indx): C x ncells | C
            fn = dist.OneHotCategorical(
               torch.ones(1,C).to(device),
            ),
            obs = ctype_i,
            obs_mask = ctype_i_mask,
      )
      return c_indx

   def subset_c_indx(self, ctype_i, cmask_i_mask):
      return ctype_i

   def sample_theta_n(self, lab, lmask, indx_n):
      log_theta_n = pyro.sample(
            name = "log_theta_n",
            # dim(log_theta_n): (P) x ncells x 1 | K
            fn = dist.Normal(
               torch.zeros(1,1,K).to(self.device),
               torch.ones(1,1,K).to(self.device)
            ).to_event(1),
            obs = subset(self.smooth_lab, indx_n),
            obs_mask = subset(lmask, indx_n)
      ) 
      # dim(theta_n): (P) x ncells x 1 x K
      theta_n = log_theta_n.softmax(dim=-1)
      return theta_n

   def compute_base_n_enum(self, c_indx, base):
      # dim(c_indx): z x ncells x C (z = 1 or C)
      c_indx = c_indx.view((-1,) + c_indx.shape[-2:])
      # dim(base_n): z x (P) x 1 x G x ncells (z = 1 or C)
      base_n = torch.einsum("znC,...GC->z...Gn", c_indx, base)
      # dim(base_n): z x (P) x G x ncells (z = 1 or C)
      base_n = base_n.squeeze(-3)
      return base_n

   def compute_base_n_no_enum(self, c_indx, base):
      # dim(base_n): (P) x ncells x G
      base_n = torch.einsum("nC,...oGC->...Gn", c_indx, base)
      return base_n

   def compute_batch_fx_n(self, batch, batch_fx, indx_n, dtype):
      # dim(ohg): ncells x B
      ohb = subset(F.one_hot(batch).to(dtype), indx_n)
      # dim(batch_fx_n): (P) x ncells x G
      batch_fx_n = torch.einsum("...GB,nB->...Gn", batch_fx, ohb)
      return batch_fx_n

   def compute_units_n(self, group, theta_n, units, indx_n):
      # dim(ohg): ncells x R
      ohg = subset(F.one_hot(group).to(units.dtype), indx_n)
      # dim(units_n): (P) x ncells x G
      units_n = torch.einsum("...noK,...GKR,nR->...Gn", theta_n, units, ohg)
      return units_n



   #  ==  model description == #

   def model(self, idx=None):

      # The correlation between cell types is given by the LKJ
      # distribution with parameter eta = 1, which is a uniform
      # prior over C x C correlation matrices. The parameter
      # `scale_tril_unit` is not the correlation matrix but the
      # lower Cholesky factor of the correlation matrix. It can
      # be passed directly to `MultivariateNormal`.

      # dim(scale_tril_unit): (P x 1) x 1 | C x C
      scale_tril_unit = self.output_scale_tril_unit()

      # The parameter `log_fuzz_loc` is the location parameter
      # for the parameter `fuzz`. The prior is Gaussian, with
      # 90% weight in the interval (-0.7, 1.7) and since `fuzz`
      # is log-normal, its median has 90% chance of being in the
      # interval (0.5, 5.3).

      # dim(log_fuzz_loc): (P x 1) x 1
      log_fuzz_loc = self.output_log_fuzz_loc()

      # The parameter `log_fuzz_scale` is the scale parameter
      # for the parameter `fuzz`. The prior is exponential,
      # with 90% weight in the interval (.05, 3.00), which
      # indicates the typical dispersion of `fuzz` between
      # genes as a log-normal variable.

      # dim(log_fuzz_scale): (P x 1) x 1
      log_fuzz_scale = self.output_log_fuzz_scale()

      with pyro.plate("C", C, dim=-1):

         # The parameter `scale_factor` describes the standard
         # deviations for every cell type from the global
         # baseline. The prior is exponential, with 90% weight
         # in the interval (0.05, 3.00). The standard deviation
         # is applied to all the genes so it describes how far
         # the cell type is from the global baseline.

         # dim(scale_factor): (P x 1) x 1 x C
         scale_factor = self.output_scale_factor()


      with pyro.plate("B", B, dim=-1):

         # The parameter `batch_fx_scale` describes the standard
         # deviations for every batch from the transcriptome of
         # the cell type. The prior is exponential, with 90% weight
         # in the interval (0.01, 0.60). The standard deviation
         # is applied to all the genes so it describes how far
         # the batch is from the prototype transcriptome.

         # dim(batch_fx_scale): (P x 1) x B
         batch_fx_scale = self.output_batch_fx_scale()


      # Set up `scale_tril` from the correlation and the standard
      # deviation. This is the lower Cholesky factor of the co-
      # variance matrix (can be used directly in `Normal`).

      # dim()scale_tril: (P x 1) x 1 x C x C
      scale_tril = scale_factor.unsqueeze(-1) * scale_tril_unit

      # Per-gene sampling.
      with pyro.plate("G", G, dim=-1):
   
         # The global baseline represents the prior average
         # expression per gene. The parameters have a Student's t
         # distribution. The distribution is centered on 0,
         # because only the variations between genes are
         # considered here. The prior is chosen so that the
         # parameters have a 90% chance of lying in the interval
         # (-3.5, 3.5), i.e., there is a factor 1000 between the
         # bottom 5% and the top 5%. The distribution has a heavy
         # tail, the top 1% is 60,000 times higher than the
         # average.

         # dim(base): (P) x 1 x G
         global_base = self.output_global_base()

         # The baselines represent the average expression per
         # gene in each cell type. The distribution is centered
         # on 0, because we consider the deviations from the
         # global baseline. The prior is chosen so that the
         # parameters have a 90% chance of lying in the interval
         # (-3.5, 3.5), i.e., there is a factor 1000 between the
         # bottom 5% and the top 5%. The distribution has a heavy
         # tail, the top 1% is 60,000 times higher than the
         # average.

         # dim(base): (P) x 1 x G | C
         base = self.output_base(global_base, scale_tril)

         # The parameter `fuzz` describes the standard deviations
         # for genes in the transcriptome. The prior is log-normal
         # with location parameter `log_fuzz_loc` and scale
         # parameter `log_fuzz_scale`. For every gene, the
         # standard deviation is applied to all the cells, so it
         # describes how "fuzzy" a gene is, or on the contrary how
         # much it is determined by the cell type and its break down
         # in transcriptional units.
   
         # TODO: describe prior.

         # dim(fuzz): (P) x 1 x G
         fuzz = self.output_fuzz(log_fuzz_loc, log_fuzz_scale)
   
         # Per-batch, per-gene sampling.
         with pyro.plate("BxG", B, dim=-2):
   
            # Batch effects have a Gaussian distribution
            # centered on 0. They are weaker than 8% for
            # 95% of the genes.

            # dim(base): (P) x B x G
            batch_fx = self.output_batch_fx(batch_fx_scale)
   
         # Per-unit, per-type, per-gene sampling.
         with pyro.plate("KRxG", K*R, dim=-2):

            # TODO: describe prior.

            # dim(units): (P) x G x K x R
            units = self.output_units()

         with pyro.plate("8xG", 8, dim=-2):
            
            cov_factor = pyro.sample(
                    "cov_factor",
                    # dim(cov_factor): (P) x 8 x G
                    dist.Normal(
                        torch.zeros(1,1).to(self.device),
                        torch.ones(1,1).to(self.device),
                    )
            )


      # Per-cell sampling.
      with pyro.plate("ncells", self.ncells, dim=-1,
         subsample=idx, device=self.device) as indx_n:

         # Subset data and mask.
         ctype_i = subset(self.ctype, indx_n)
         ctype_i_mask = subset(self.cmask, indx_n)
         x_i = subset(self.X, indx_n).to_dense()
         x_i_mask = subset(self.gmask, indx_n)
   
         # TODO: describe prior.

         # dim(c_indx): C x (P) x 1 x ncells | C
         c_indx = self.output_c_indx(ctype_i, ctype_i_mask)

         # Proportion of the units in the transcriptomes.
         # TODO: describe prior.

         # dim(theta_n): (P) x ncells x 1 x K  ///  *
         theta_n = self.output_theta_n(self.smooth_lab, self.lmask, indx_n)
   

         # Deterministic functions to obtain per-cell means.

         # dim(base_n): C x (P) x G x ncells
         base_n = self.output_base_n(c_indx, base)
   
         # dim(batch_fx_n): (P) x G x ncells
         batch_fx_n = self.output_batch_fx_n(batch, batch_fx, indx_n, base.dtype)
   
         # dim(units_n): (P) x G x ncells
         units_n = self.output_units_n(group, theta_n, units, indx_n)

         # dim(mu_n): C x (P) x ncells x G
         mu_n = base_n + batch_fx_n + units_n

#         # Per-cell, per-gene sampling.
#         with pyro.plate("Gxncells", G, dim=-2):
#
#            fuzz = fuzz.transpose(-1,-2)
#            logits_n = pyro.sample(
#                  name = "logits_n",
#                  fn = dist.Normal(
#                     mu_n, # dim: C x (P) x G x ncells
#                     fuzz, # dim:     (P) x G x 1
#                  ),
#            )

         mu_n = mu_n.transpose(-1,-2).unsqueeze(-3)
         cov_factor = cov_factor.transpose(-1,-2).unsqueeze(-3).unsqueeze(-3)
         fuzz = fuzz.unsqueeze(-3)
         logits_n = pyro.sample(
               name = "logits_n",
               # dim(logits_n): C x (P) x 1 x ncells | G
               fn = dist.LowRankMultivariateNormal(
                  loc = mu_n,              # dim: C x (P) x 1 x ncells x G
                  cov_factor = cov_factor, # dim:     (P) x 1 x 1      x G x 2
                  cov_diag = fuzz,         # dim:     (P) x 1 x 1      x G
               ),
         )

         pyro.sample(
               name = "x_i",
               # dim(x_i): ncells x G
               fn = dist.Multinomial(
                  logits = logits_n,
                  validate_args = False,
               ),
               obs = x_i,
               obs_mask = x_i_mask
         )

      global DEBUG_COUNTER
      DEBUG_COUNTER += 1
      if DEBUG_COUNTER > 1950:
         import pdb; pdb.set_trace()
      return x_i


if __name__ == "__main__":

   pl.seed_everything(123)
   pyro.set_rng_seed(123)
   torch.manual_seed(123)

   torch.set_float32_matmul_precision('medium')

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
   # XXX #
   idx = torch.randperm(int(X.shape[0]))[:250].sort().values
   idx = torch.cat([torch.arange(10), idx[10:]])
   X = subset(X, idx).to(device)


   lmask = torch.zeros(X.shape[0], dtype=torch.bool).to(device)

   # Set the dimensions.
   B = int(batch.max() + 1)
   C = int(ctype.max() + 1)
   R = int(group.max() + 1)
   G = int(X.shape[-1])

   # XXX #
   ctype = ctype[idx]
   batch = batch[idx]
   group = group[idx]
   label = label[idx]
   cmask = cmask[idx]

   data = (ctype, batch, group, label, X, (cmask, lmask, None))
   data_idx = range(X.shape[0])

   data_loader = torch.utils.data.DataLoader(
         dataset = data_idx,
         shuffle = True,
         batch_size = SUBSMPL,
   )


   pyro.clear_param_store()
   stadacone = Stadacone(data, amortize=False)
   harnessed = plTrainHarness(stadacone)

   trainer = pl.Trainer(
      default_root_dir = ".",
      strategy = pl.strategies.DeepSpeedStrategy(stage=2),
      accelerator = "gpu",
      precision = "32",
      devices = 1 if DEBUG else -1,
      gradient_clip_val = 1.0,
      max_epochs = NUM_EPOCHS,
      enable_progress_bar = True,
      enable_model_summary = True,
      logger = pl.loggers.CSVLogger("."),
      enable_checkpointing = True,
   )

   trainer.fit(harnessed, data_loader)

   # Save output to file.
   param_store = pyro.get_param_store().get_state()
   for key, value in param_store["params"].items():
       param_store["params"][key] = value.clone().cpu()
   torch.save(param_store, out_path)

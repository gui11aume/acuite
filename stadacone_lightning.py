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
NUM_PARTICLES = 8
CELL_COVERAGE = 100


# Use only for debugging.
pyro.enable_validation(DEBUG)

class DummyModule(torch.nn.Module):
   def __init__(self, params):
      super().__init__()
      self.params = params

def subset(tensor, idx):
   if tensor is None: return None
   return tensor.index_select(0, idx.to(tensor.device))


class plTrainHarness(pl.LightningModule):
   def __init__(self, stadacone, lr=.01):
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

      stadacone_params = self.capture_params()
      self.model = DummyModule(stadacone_params)

   def capture_params(self):
      idx = torch.randperm(self.stadacone.ncells)[:self.stadacone.bsz]
      with pyro.poutine.trace(param_only=True) as param_capture:
         self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, idx)
      params = set(site["value"].unconstrained()
            for site in param_capture.trace.nodes.values())
      return params

   def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=0.01)

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
      loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, batch)
      (current_lr,) = self.lr_schedulers().get_last_lr()
      info = { "loss": loss, "lr": current_lr }
      self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
      return loss


# This part of the model should be created with `AutoDiscreteParallel`
# but there is a bug (https://github.com/pyro-ppl/pyro/issues/3286).
# This should be eventually replaced when the bug is fixed.

class c_indx_autoguide(AutoGuide):
   def __init__(self, pyro_model, device, ncells, cmask):
      super().__init__(model=pyro_model)
      self.device = device
      self.ncells = ncells
      self.cmask = cmask

   def _setup_prototype(self):
      model = pyro.poutine.block(pyro.infer.enum.config_enumerate(self.model), self._prototype_hide_fn)
      self.prototype_trace = pyro.poutine.block(pyro.poutine.trace(model).get_trace)()
      self._cond_indep_stacks = {}
      self._prototype_frames = {}
      for name, site in self.prototype_trace.iter_stochastic_nodes():
         self._cond_indep_stacks[name] = site["cond_indep_stack"]
         for frame in site["cond_indep_stack"]:
            self._prototype_frames[frame.name] = frame
      pyro.infer.autoguide.utils.deep_setattr(self,
         "AutoGuideList.1.post_c_indx_probs",
         pyro.nn.module.PyroParam(
            torch.ones(self.ncells,1,C).to(self.device),
            constraint = torch.distributions.constraints.simplex,
            event_dim = 1
         ),
      )

   def forward(self, idx=None):
      if self.prototype_trace is None:
          self._setup_prototype()

      plates = self._create_plates()
      with ExitStack() as stack:
         for frame in self._cond_indep_stacks["cell_type_unobserved"]:
             stack.enter_context(plates[frame.name])
         # Posterior distribution of `c_indx`.
         post_c_indx_probs = pyro.infer.autoguide.utils.deep_getattr(self,
            "AutoGuideList.1.post_c_indx_probs")
         with pyro.poutine.mask(mask=subset(~self.cmask, idx)):
            post_c_indx = pyro.sample(
               name = "cell_type_unobserved",
               # dim(c_indx): C x 1 x 1 x 1 | .
               fn = dist.Categorical(
                  post_c_indx_probs # dim: ncells x 1 | C
               ),
               infer={ "enumerate": "parallel" },
            )
      return { "cell_type_unobserved": post_c_indx }



class Stadacone(torch.nn.Module):

   def __init__(self, data, inference_mode = True):
      super().__init__()

      # Unpack data.
      self.ctype, self.batch, self.group, self.label, self.X, masks = data
      self.cmask, self.lmask, self.gmask = masks
   
      self.ctype = self.ctype.view(-1,1)
      self.cmask = self.cmask.view(-1,1)
      self.lmask = self.lmask.view(-1,1)
   
      self.device = self.X.device
      self.ncells = int(self.X.shape[0])
   
      self.bsz = self.ncells if self.ncells < SUBSMPL else SUBSMPL

      # Format observed labels. Create one-hot encoding with label smoothing.
      oh = F.one_hot(self.label, num_classes=K).to(self.X.dtype)
      self.smooth_lab = ((.99-.01/(K-1)) * oh + .01/(K-1)).view(-1,1,K) if K > 1 else 0.

      # 1a) Define core parts of the model.
      self.output_scale_tril_unit = self.sample_scale_tril_unit
      self.output_scale_factor = self.sample_scale_factor
      self.output_log_wiggle_loc = self.sample_log_wiggle_loc
      self.output_log_wiggle_scale = self.sample_log_wiggle_scale
      self.output_global_base = self.sample_global_base
      self.output_wiggle = self.sample_wiggle
      self.output_base = self.sample_base
      self.output_shift_n = self.sample_shift_n
      self.output_x_i = self.compute_ELBO_rate_n

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
     
      if inference_mode is False:
         self.output_x_i = self.sample_rate_n
      
      # 2) Define the guide.
      self.guide = AutoGuideList(self.model)
      self.guide.append(AutoNormal(
          pyro.poutine.block(self.model, hide = ["cell_type_unobserved"])
      ))
      if self.need_to_infer_cell_type:
         self.guide.append(c_indx_autoguide(
             pyro.poutine.block(self.model, expose = ["cell_type_unobserved"]),
             device = self.device, ncells = self.ncells, cmask = self.cmask
         ))


   #  == Helper functions == #
   def zero(self, *args, **kwargs):
      return 0.


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

   def sample_log_wiggle_loc(self):
      log_wiggle_loc = pyro.sample(
            name = "log_wiggle_loc",
            # dim(log_wiggle_loc): (P x 1) x 1
            fn = dist.Normal(
                0. * torch.ones(1).to(self.device),
                1. * torch.ones(1).to(self.device)
            ),
      )
      return log_wiggle_loc

   def sample_log_wiggle_scale(self):
      log_wiggle_scale = pyro.sample(
            name = "log_wiggle_scale",
            # dim(log_wiggle_scale): (P x 1) x 1
            fn = dist.Exponential(
                3. * torch.ones(1).to(self.device),
            ),
      )
      return log_wiggle_scale

   def sample_batch_fx_scale(self):
      batch_fx_scale = pyro.sample(
            name = "batch_fx_scale",
            # dim(base): (P x 1) x B
            fn = dist.Exponential(
               3. * torch.ones(1).to(self.device),
            ),
      )
      return batch_fx_scale

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

   def sample_global_base(self): 
      global_base = pyro.sample(
            name = "global_base",
            # dim(global_base): (P) x G x 1
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
            # dim(base_0): (P) x G x 1 | C
            fn = dist.MultivariateNormal(
                torch.zeros(C).to(self.device),
                scale_tril = scale_tril
            ),
      )
      # dim(base): (P) x G x C
      base = loc + base_0.squeeze(-2)
      return base

   def sample_wiggle(self, loc, scale):
      wiggle_Gx1 = pyro.sample(
            name = "wiggle_Gx1",
            # dim(wiggle_Gx1): (P) x G x 1
            fn = dist.LogNormal(loc, scale),
      )
      # dim(wiggle): (P) x 1 x G
      wiggle = wiggle_Gx1.transpose(-1,-2)
      return wiggle

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

   def sample_c_indx(self, ctype, cmask, indx_n):
      c_indx = pyro.sample(
            name = "cell_type",
            # dim(c_indx): C x 1 x ncells x 1 | .
            fn = dist.Categorical(
               torch.ones(1,1,C).to(self.device),
            ),
            obs = subset(ctype, indx_n),
            obs_mask = subset(cmask, indx_n)
      )
      return c_indx

   def subset_c_indx(self, ctype, cmask, indx_n):
      # dim(c_indx): ncells x 1
      c_indx = subset(ctype, indx_n)
      return c_indx

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

   def compute_base_n_enum(self, c_indx, base):
      # dim(ohc): ncells x C  //  C x ncells x C
      ohc = F.one_hot(c_indx.squeeze(), num_classes=C).float()
      # dim(ohc): z x ncells x C (z = 1 or C)
      ohc = ohc.view((-1,) + ohc.shape[-2:])
      # dim(base_n): z x (P) x ncells x G (z = 1 or C)
      base_n = torch.einsum("znC,...GC->z...nG", ohc, base)
      return base_n

   def compute_base_n_no_enum(self, c_indx, base):
      # dim(ohc): ncells x C
      ohc = F.one_hot(c_indx.squeeze(), num_classes=C).float()
      # dim(base_n): (P) x ncells x G
      base_n = torch.einsum("nC,...GC->...nG", ohc, base)
      return base_n

   def compute_batch_fx_n(self, batch, batch_fx, indx_n, dtype):
      # dim(ohg): ncells x B
      ohb = subset(F.one_hot(batch).to(dtype), indx_n)
      # dim(batch_fx_n): (P) x ncells x G
      batch_fx_n = torch.einsum("...GB,nB->...nG", batch_fx, ohb)
      return batch_fx_n

   def compute_units_n(self, group, theta_n, units, indx_n):
      # dim(ohg): ncells x R
      ohg = subset(F.one_hot(group).to(units.dtype), indx_n)
      # dim(units_n): (P) x ncells x G
      units_n = torch.einsum("...noK,...GKR,nR->...nG", theta_n, units, ohg)
      return units_n

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
         mu_i = torch.clamp(mu_i - f / df, max = x_i * w2 - .01)
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

         # TODO: describe prior.

         # dim(batch_fx_scale): (P x 1) x B
         batch_fx_scale = self.output_batch_fx_scale()


      # Set up `scale_tril` from the correlation and the standard
      # deviation. This is the lower Cholesky factor of the co-
      # variance matrix (can be used directly in `Normal`).
      scale_tril = scale_factor.unsqueeze(-1) * scale_tril_unit

      # dim(log_wiggle_loc): (P x 1) x 1
      log_wiggle_loc = self.output_log_wiggle_loc()

      # dim(log_wiggle_scale): (P x 1) x 1
      log_wiggle_scale = self.output_log_wiggle_scale()

      # Per-gene sampling.
      with pyro.plate("G", G, dim=-2):
   
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

         # dim(base): (P) G x 1
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

         # dim(base): (P) G x 1 | C
         base = self.output_base(global_base, scale_tril)

         # TODO: describe prior.

         # dim(base): (P) x 1 x G
         wiggle = self.output_wiggle(log_wiggle_loc, log_wiggle_scale)
   
         # Per-batch, per-gene sampling.
         with pyro.plate("GxB", B, dim=-1):
   
            # Batch effects have a Gaussian distribution
            # centered on 0. They are weaker than 8% for
            # 95% of the genes.

            # dim(base): (P) x G x B
            batch_fx = self.output_batch_fx(batch_fx_scale)
   
         # Per-unit, per-type, per-gene sampling.
         with pyro.plate("GxKR", K*R, dim=-1):

            # TODO: describe prior.

            # dim(units): (P) x G x K x R
            units = self.output_units()


      # Per-cell sampling (on dimension -2).
      with pyro.plate("ncells", self.ncells, dim=-2,
         subsample=idx, device=self.device) as indx_n:
   
         # TODO: describe prior.

         # dim(c_indx): C x 1 x ncells x 1 | .  /// ncells x 1
         c_indx = self.output_c_indx(self.ctype, self.cmask, indx_n)

         # Proportion of the units in the transcriptomes.
         # TODO: describe prior.

         # dim(theta_n): (P) x ncells x 1 x K  ///  *
         theta_n = self.output_theta_n(self.smooth_lab, self.lmask, indx_n)
   
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
   
         # dim(units_n): (P) x ncells x G
         units_n = self.output_units_n(group, theta_n, units, indx_n)


         # Per-cell, per-gene sampling.
         with pyro.plate("ncellsxG", G, dim=-1):

            mu = base_n + batch_fx_n + units_n + shift_n
            x_i = subset(self.X, indx_n).to_dense()
            x_i_mask = subset(self.gmask, indx_n)

            self.output_x_i(x_i, mu, wiggle, x_i_mask)


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

   lmask = torch.zeros(X.shape[0], dtype=torch.bool).to(device)

   # Set the dimensions.
   B = int(batch.max() + 1)
   C = int(ctype.max() + 1)
   R = int(group.max() + 1)
   G = int(X.shape[-1])


   data = (ctype, batch, group, label, X, (cmask, lmask, None))
   data_idx = range(X.shape[0])

   data_loader = torch.utils.data.DataLoader(
         dataset = data_idx,
         shuffle = True,
         batch_size = SUBSMPL,
   )


   pyro.clear_param_store()
   stadacone = Stadacone(data)
   harnessed = plTrainHarness(stadacone)

   trainer = pl.Trainer(
      default_root_dir = ".",
#      strategy = pl.strategies.FSDPStrategy(
##         cpu_offload = torch.distributed.fsdp.CPUOffload(offload_params=True),
#         activation_checkpointing = [
#            transformers.models.bert.modeling_bert.BertLayer,
#            transformers.models.bert.modeling_camembert.CamembertLayer,
#         ],
#         mixed_precision = torch.distributed.fsdp.MixedPrecision(
#            param_dtype=torch.bfloat16,
#            reduce_dtype=torch.bfloat16,
#            buffer_dtype=torch.bfloat16,
#         ),
#         auto_wrap_policy = wrapping_policy
#      ),
      strategy = pl.strategies.DDPStrategy(find_unused_parameters=False),
      accelerator = "gpu",
      precision = "bf16-mixed",
      devices = 1 if DEBUG else -1,
      gradient_clip_val = 1.0,
#      accumulate_grad_batches = ACC,
      max_epochs = CELL_COVERAGE,
      enable_progress_bar = True,
      enable_model_summary = True,
      logger = pl.loggers.CSVLogger("."),
      # Checkpointing.
      enable_checkpointing = True,
   )

   trainer.fit(harnessed, data_loader)
   pyro.get_param_store().save(out_path)

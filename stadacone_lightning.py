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
SUBSMPL = 512
NUM_PARTICLES = 12
NUM_EPOCHS = 64

global DEBUG_COUNTER
DEBUG_COUNTER = 0

# Use only for debugging.
pyro.enable_validation(DEBUG)


def subset(tensor, idx):
   if idx is None: return tensor
   if tensor is None: return None
   return tensor.index_select(0, idx.to(tensor.device))


class plTrainHarness(pl.LightningModule):
   def __init__(self, stadacone, lr=0.01):
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
      loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, batch)
      (lr,) = self.lr_schedulers().get_last_lr()
      info = { "loss": loss, "lr": lr }
      self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
      return loss



class Stadacone(torch.nn.Module):

   def __init__(self, data, marginalize_rate_n=True):
      super().__init__()

      # Unpack data.
      self.ctype, self.batch, self.group, self.label, self.X, masks = data
      self.cmask, self.lmask, self.gmask = masks
   
      self.ctype = F.one_hot(self.ctype.view(-1,1), num_classes=C).float()
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
      self.output_log_fuzz_loc = self.sample_log_fuzz_loc
      self.output_log_fuzz_scale = self.sample_log_fuzz_scale
      self.output_global_base = self.sample_global_base
      self.output_gene_fuzz = self.sample_gene_fuzz
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
         self.output_c_indx = self.return_ctype_as_is
         self.output_base_n = self.compute_base_n_no_enum
      else:
         self.need_to_infer_cell_type = True
         self.output_c_indx = self.sample_c_indx
         self.output_base_n = self.compute_base_n_enum
     
      if marginalize_rate_n is False:
         self.output_x_i = self.sample_log_rate_n
      
      # 2) Define the guide.
      self.guide = AutoNormal(
         pyro.infer.config_enumerate(
            pyro.poutine.block(self.model, hide = ["cell_type_unobserved"])
         ),
         create_plates = self.create_ncells_plate
      )


   #  == Helper functions == #
   def zero(self, *args, **kwargs):
      return 0.

   def create_ncells_plate(self, idx=None):
      return pyro.plate("ncells", self.ncells, dim=-2,
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
                -1.0 * torch.ones(1,1).to(self.device),
                 0.5 * torch.ones(1,1).to(self.device)
            ),
      )
      return log_fuzz_loc

   def sample_log_fuzz_scale(self):
      log_fuzz_scale = pyro.sample(
            name = "log_fuzz_scale",
            # dim(log_fuzz_scale): (P) x 1 x 1
            fn = dist.Exponential(
                3. * torch.ones(1,1).to(self.device),
            ),
      )
      return log_fuzz_scale

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

   def sample_gene_fuzz(self, loc, scale):
      gene_fuzz = pyro.sample(
            name = "gene_fuzz",
            # dim(fuzz): (P) x 1 x G
            fn = dist.LogNormal(loc, scale),
      )
      return gene_fuzz

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

   def sample_c_indx(self, probs_n, ctype_i, ctype_i_mask):
      c_indx = pyro.sample(
            name = "cell_type",
            # dim(c_indx): C x (P) x ncells x 1 | C
            fn = dist.OneHotCategorical(
               probs_n
            ),
            obs = ctype_i,
            obs_mask = ctype_i_mask,
            infer = { "enumerate": "parallel" }
      )
      return c_indx

   def return_ctype_as_is(self, ctype_i, cmask_i_mask):
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
      # dim(c_indx): z x ncells x C (z = 1 or C)
      c_indx = c_indx.view((-1,) + c_indx.shape[-3:]).squeeze(-2)
      # dim(base_n): z x (P) x ncells x G (z = 1 or C)
      base_n = torch.einsum("znC,...GC->z...nG", c_indx, base)
      return base_n

   def compute_base_n_no_enum(self, c_indx, base):
      # dim(c_indx): ncells x C
      c_indx = c_indx.squeeze(-2)
      # dim(base_n): (P) x ncells x G
      base_n = torch.einsum("nC,...GC->...nG", c_indx, base)
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

   def compute_ELBO_rate_n(self, x_i, mu, sg, x_i_mask, idx):
      # Parameters `mu` and `sg` are the prior parameters of the Poisson
      # LogNormal distribution. The variational posterior parameters
      # given the observations `x_i` are `mu_i` and `w2_i`. In this case
      # we can compute the ELBO analytically and maximize it with respect
      # to `mu_i` and `w2_i` so as to pass the gradient to `mu` and `sg`.
      # This allows us to compute the ELBO efficiently without having
      # to store parameters and gradients for `mu_i` and `w2_i`.

      # FIXME: compute something during prototyping.
      if mu.dim() < 4: return

      # Detach gradient from input parameters.
      m = mu.detach()
      w2 = torch.square(sg.detach())
      # Initialize `mu_i` with dim: (P) x ncells x G.
      mu_i = (x_i * w2 - 3.) * torch.ones(m.shape[-3:]).to(self.device)
      # Perform 5 Newton-Raphson iterations.
      for _ in range(5):
         f = m + mu_i + w2 * .5 / (w2 * x_i + 1 - mu_i) - torch.log(x_i - mu_i / w2)
         df = 1 + w2 * .5 / torch.square(w2 * x_i + 1 - mu_i) + 1. / (w2 * x_i - mu_i)
         mu_i = torch.clamp(mu_i - f / df, max = x_i * w2 - .01)

      # Set the optimal `w2_i` from the optimal `mu_i`.
      s2 = torch.square(sg)
      w2_i = s2 / (s2 * x_i + 1 - mu_i)

      # Compute ELBO term as a function of `mu` and `sg`,
      # for which we kept the gradient.
      def mini_ELBO_fn(mu, sg, mu_i, w2_i):
         return - torch.exp(mu + mu_i + 0.5 * w2_i)  \
                  + x_i * (mu + mu_i) - torch.log(sg) \
                  + 0.5 * torch.log(w2_i)             \
                  - 0.5 * (mu_i * mu_i + w2_i) / (sg * sg) \
#                  - torch.lgamma(x_i+1) + .5
      mini_ELBO = mini_ELBO_fn(mu, sg, mu_i, w2_i)
   
      pyro.factor("PLN_ELBO_term", mini_ELBO)
      return x_i

   def sample_log_rate_n(self, x_i, mu, gene_fuzz, gmask, idx=None):
      delta_log_rate_n = pyro.sample(
            name = "delta_log_rate_n",
            # dim(delta_log_rate_n): (P) x ncells x G
            fn = dist.Normal(
               torch.zeros(1,1).to(self.device),
               gene_fuzz, # dim: (P) x 1 x G
            ),
      )
      # dim(log_rate_n): C x (P) x ncells x G
      rate_n = torch.clamp(torch.exp(mu + delta_log_rate_n), max=1e6)
      x_i = pyro.sample(
            name = "x_i",
            # dim(x_i): ncells x G
            fn = dist.Poisson(
               rate = rate_n # dim: C x (P) x ncells x G
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

      # The parameter `log_fuzz_loc` is the location parameter
      # for the parameter `fuzz`. The prior is Gaussian, with
      # 90% weight in the interval (-1.8, 0.2) and since `fuzz`
      # is log-normal, its median has 90% chance of being in the
      # interval (0.15, 0.85).

      # dim(log_fuzz_loc): (P) x 1 x 1
      log_fuzz_loc = self.output_log_fuzz_loc()

      # The parameter `log_fuzz_scale` is the scale parameter
      # for the parameter `fuzz`. The prior is exponential,
      # with 90% weight in the interval (.15, 1.00), which
      # indicates the typical dispersion of `fuzz` between
      # genes as a log-normal variable.

      # dim(log_fuzz_scale): (P) x 1 x 1
      log_fuzz_scale = self.output_log_fuzz_scale()

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

         # dim(base): (P) x G x 1
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

         # The parameter `gene_fuzz` is the standard deviation
         # for genes in the transcriptome. The prior is log-normal
         # with location parameter `log_fuzz_loc` and scale
         # parameter `log_fuzz_scale`. For every gene, the
         # standard deviation is applied to all the cells, so it
         # describes how "fuzzy" a gene is, or on the contrary how
         # much it is determined by the cell type and its break down
         # in transcriptional units.
   
         # dim(fuzz): (P) x G x 1
         gene_fuzz = self.output_gene_fuzz(log_fuzz_loc, log_fuzz_scale)
   
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

         # Subset data and mask.
         ctype_i = subset(self.ctype, indx_n)
         ctype_i_mask = subset(self.cmask, indx_n)
         x_i = subset(self.X, indx_n).to_dense()
         x_i_mask = subset(self.gmask, indx_n)
   
         # TODO: describe prior.

         logits_n = pyro.sample(
                 # dim(logits_n): (P) x ncells | C
                 name = "logits_n",
                 fn = pyro.distributions.Normal(
                      0  * torch.zeros(1,C).to(self.device),
                     C/2 * torch.ones(1,C).to(self.device),
                 ).to_event(1)
         )

         # dim(logits_n): (P) x ncells x 1 x C
         probs_n = logits_n.softmax(dim=-1)

         # dim(c_indx): C x (P) x ncells x 1 | C  /// (P) x ncells x 1 | C
         c_indx = self.output_c_indx(probs_n, ctype_i, ctype_i_mask)

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

            # dim(mu_n): C x (P) x ncells x G
            mu_n = base_n + batch_fx_n + units_n + shift_n
            # dim(gene_fuzz): (P) x 1 x G
            gene_fuzz = gene_fuzz.transpose(-1,-2)
            self.output_x_i(x_i, mu_n, gene_fuzz, x_i_mask, indx_n)



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
   stadacone = Stadacone(data, marginalize_rate_n=True)
   harnessed = plTrainHarness(stadacone)

   trainer = pl.Trainer(
      default_root_dir = ".",
      strategy = pl.strategies.DeepSpeedStrategy(stage=2),
      accelerator = "gpu",
      precision = "32",
      devices = [0],
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

import pyro
import re
import torch


class NegativeBinomial(torch.distributions.NegativeBinomial):
   def log_prob(self, value):
      if self._validate_args:
          self._validate_sample(value)
      log_unnormalized_prob = (self.total_count * torch.nn.functional.logsigmoid(-self.logits) +
            value * torch.nn.functional.logsigmoid(self.logits))
      log_normalization = (-torch.lgamma(self.total_count + value) +
            torch.lgamma(1. + value) + torch.lgamma(self.total_count))
      log_normalization = log_normalization.masked_fill(self.total_count + value == 0., 0.)
      return log_unnormalized_prob - log_normalization


class ZeroInflatedNegativeBinomial(pyro.distributions.ZeroInflatedNegativeBinomial):
   def __init__(self, total_count, *, probs=None, logits=None, gate=None, gate_logits=None, validate_args=None):
      base_dist = NegativeBinomial(
         total_count=total_count,
         probs=probs,
         logits=logits,
         validate_args=False,
      )
      base_dist._validate_args = validate_args
      super(pyro.distributions.ZeroInflatedNegativeBinomial, self).__init__(
         base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
      )


class warmup_scheduler(torch.optim.lr_scheduler.ChainedScheduler):
   def __init__(self, optimizer, warmup=100, decay=None):
      self.warmup = warmup
      self.decay = decay if decay is not None else 100000000
      warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.,
            total_iters=self.warmup
      )
      linear_decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.,
            end_factor=0.05,
            total_iters=self.decay
      )
      super().__init__([warmup, linear_decay])


def sc_data(data_path, header_present = True):
   """ 
   Data for single-cell transcriptome, returns a 3-tuple with
      1. a list of cell identifiers,
      2. a tensor of cell types as integers,
      3. a tensor of batches as integers,
      4. a tensor of labels as integers,
      5. a tensor with read counts,
      6. a mask tensor for labels as boolean.
   """

   list_of_identifiers = list()
   list_of_ctypes = list()
   list_of_batches = list()
   list_of_labels = list()
   list_of_exprs = list()

   # Helper function.
   parse = lambda row: (row[0], row[1], row[2], row[3], [round(float(x)) for x in row[4:]])

   with open(data_path) as f:
      if header_present:
         ignore_header = next(f)
      for line in f:
         identifier, cell, batch, label, expr = parse(line.split())
         list_of_identifiers.append(identifier)
         list_of_ctypes.append(cell)
         list_of_batches.append(batch)
         list_of_labels.append(label)
         list_of_exprs.append(torch.tensor(expr))

   unique_ctypes = sorted(list(set(list_of_ctypes)))
   list_of_ctype_ids = [unique_ctypes.index(x) for x in list_of_ctypes]
   ctype_tensor = torch.tensor(list_of_ctype_ids)

   unique_batches = sorted(list(set(list_of_batches)))
   list_of_batches_ids = [unique_batches.index(x) for x in list_of_batches]
   batches_tensor = torch.tensor(list_of_batches_ids)

   unique_labels = sorted(list(set(list_of_labels)))
   if "?" in unique_labels:
      unique_labels.remove("?")
      label_mask = [label != "?" for label in list_of_labels]
   else:
      label_mask = [True] * len(list_of_labels)
   label_mask_tensor = torch.tensor(label_mask, dtype=torch.bool)
   list_of_labels_ids = [
         unique_labels.index(x) if x in unique_labels else 0
         for x in list_of_labels
   ]
   labels_tensor = torch.tensor(list_of_labels_ids)
   labels_tensor[~label_mask_tensor] = 0

   expr_tensor = torch.stack(list_of_exprs)

   return (
      list_of_identifiers,
      ctype_tensor,
      batches_tensor,
      labels_tensor,
      expr_tensor,
      label_mask_tensor,
   )


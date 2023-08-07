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


def sc_data(data_path):
   """ 
   Data for single-cell transcriptome, returns a 3-tuple with
      1. list of cell identifiers (arbitrary),
      2. tensor of cell types as integers,
      3. tensor of batches as integers,
      4. tensor of groups as integers,
      5. tensor of labels as integers,
      6. tensor of read counts as float,
      7. tensor of label masks as boolean.
   """

   list_of_identifiers = list()
   list_of_ctypes = list()
   list_of_batches = list()
   list_of_groups = list()
   list_of_labels = list()
   list_of_exprs = list()

   # Helper functions.
   def parse_header(line):
      items = line.split()
      if not re.search(r"^[Gg]roups?", items[3]):   return 3
      if not re.search(r"^[Ll]abels?", items[4]):   return 4
      return 5
      
   parse = lambda n, row: (row[:n], [round(float(x)) for x in row[n:]])


   # Read in data from file.
   with open(data_path) as f:
      first_numeric_field = parse_header(next(f))
      for line in f:
         info, expr = parse(first_numeric_field, line.split())
         list_of_identifiers.append(info[0])
         list_of_ctypes.append(info[1])
         list_of_batches.append(info[2])
         if len(info) >= 4: list_of_groups.append(info[3])
         if len(info) >= 5: list_of_labels.append(info[4])
         list_of_exprs.append(torch.tensor(expr))

   unique_ctypes = sorted(list(set(list_of_ctypes)))
   list_of_ctype_ids = [unique_ctypes.index(x) for x in list_of_ctypes]
   ctype_tensor = torch.tensor(list_of_ctype_ids)

   unique_batches = sorted(list(set(list_of_batches)))
   list_of_batches_ids = [unique_batches.index(x) for x in list_of_batches]
   batches_tensor = torch.tensor(list_of_batches_ids)

   if list_of_groups:
      unique_groups = sorted(list(set(list_of_groups)))
      list_of_groups_ids = [unique_groups.index(x) for x in list_of_groups]
      groups_tensor = torch.tensor(list_of_groups_ids)
   else:
      groups_tensor = torch.zeros(len(list_of_identifiers)).to(torch.long)

   if list_of_labels:
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
   else:
      labels_tensor = torch.zeros(len(list_of_identifiers)).to(torch.long)
      label_mask_tensor = torch.zeros(len(list_of_identifiers)).to(torch.bool)

   expr_tensor = torch.stack(list_of_exprs)

   return (
      list_of_identifiers,
      ctype_tensor,
      batches_tensor,
      groups_tensor,
      labels_tensor,
      expr_tensor,
      label_mask_tensor,
   )


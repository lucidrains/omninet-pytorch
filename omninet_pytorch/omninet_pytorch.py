import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
	return val is not None

# classes

class Omninet(nn.Module):
	def __init__(
		self,
		*,
		dim,
		dim_head = 64,
		heads = 8
	):
		super().__init__()

	def forward(self, x):
		return x

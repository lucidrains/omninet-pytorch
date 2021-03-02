<img src="./omninet.png" width="400px"></img>

## Omninet - Pytorch

Implementation of <a href="https://arxiv.org/abs/2103.01075">OmniNet</a>, Omnidirectional Representations from Transformers, in Pytorch. The authors propose that we should be attending to all the tokens of the previous layers, leveraging recent efficient attention advances to achieve this goal.

## Install

```bash
$ pip install omninet-pytorch
```

## Usage

```python
import torch
from omninet_pytorch import Omninet

omninet = Omninet(
    dim = 512,                     # model dimension
    depth = 6,                     # depth
    dim_head = 64,                 # dimension per head
    heads = 8,                     # number of heads
    pool_layer_tokens_every = 3,   # key to this paper - every N layers, omni attend to all tokens of all layers
    attn_dropout = 0.1,            # attention dropout
    ff_dropout = 0.1               # feedforward dropout
)

x = torch.randn(1, 1024, 512)
mask = torch.ones(1, 1024).bool()

omninet(x, mask = mask) # (1, 1024, 512)
```

Causal case, just use the class `OmninetCausal`. At the moment, it isn't faithful to the paper (I am using layer axial attention with layer positional embeddings to draw up information), but will fix this once I rework the linear attention CUDA kernel.

```python
import torch
from omninet_pytorch import OmninetCausal

omninet = OmninetCausal(
    dim = 512,                     # model dimension
    depth = 6,                     # depth
    dim_head = 64,                 # dimension per head
    heads = 8,                     # number of heads
    pool_layer_tokens_every = 3,   # key to this paper - every N layers, omni attend to all tokens of all layers
    attn_dropout = 0.1,            # attention dropout
    ff_dropout = 0.1               # feedforward dropout
)

x = torch.randn(1, 1024, 512)
mask = torch.ones(1, 1024).bool()

omninet(x, mask = mask) # (1, 1024, 512)
```

## Citations

```bibtex
@misc{tay2021omninet,
    title   = {OmniNet: Omnidirectional Representations from Transformers}, 
    author  = {Yi Tay and Mostafa Dehghani and Vamsi Aribandi and Jai Gupta and Philip Pham and Zhen Qin and Dara Bahri and Da-Cheng Juan and Donald Metzler},
    year    = {2021},
    eprint  = {2103.01075},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

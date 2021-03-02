import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# use Performer, as it had the best reported numbers

from performer_pytorch import SelfAttention as PerformerAttention

# helpers

def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads =  heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, d, h, device = *x.shape, self.heads, x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            sim.masked_fill_(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = rearrange(causal_mask, 'i j -> () i j')
            sim.masked_fill_(causal_mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class Omninet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        pool_layer_tokens_every = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()

        layers = nn.ModuleList([])
        for ind in range(depth):
            num_layers = ind + 1
            should_pool = num_layers % pool_layer_tokens_every

            layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout)),
                PerformerAttention(dim = dim, heads= heads, dim_head = dim_head) if should_pool else None
            ]))

        self.layers = layers

    def forward(self, x, mask = None):
        hiddens = [x]

        for attn, ff, efficient_attn in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

            hiddens.append(x)
            if exists(efficient_attn):
                num_layers = len(hiddens)
                all_tokens = rearrange(torch.stack(hiddens), 'l b n d -> b (n l) d')

                pool_attn_mask = None
                if exists(mask):
                    pool_attn_mask = repeat(mask, 'b n -> b (n l)', l = num_layers)

                attended_tokens = efficient_attn(all_tokens, mask = pool_attn_mask)

                attended_tokens = rearrange(attended_tokens, 'b n c -> b c n')
                pooled_tokens = F.max_pool1d(attended_tokens, kernel_size = num_layers, stride = num_layers)
                x += rearrange(pooled_tokens, 'b c n -> b n c')

        return x

# causal case is sufficiently different to warrant its own class
# use layer axial attention for now, until I rewrite the linear attention cuda kernel

class OmninetCausal(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        pool_layer_tokens_every = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()

        self.layer_pos_emb = nn.Parameter(torch.randn(depth + 1, dim))

        layers = nn.ModuleList([])
        for ind in range(depth):
            num_layers = ind + 1
            should_pool = num_layers % pool_layer_tokens_every

            layers.append(nn.ModuleList([
                PreNorm(dim, Attention(causal = True, dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout)),
                Attention(dim = dim, heads= heads, dim_head = dim_head) if should_pool else None
            ]))

        self.layers = layers

    def forward(self, x, mask = None):
        b = x.shape[0]
        pos_embs = rearrange(self.layer_pos_emb, 'n d -> () n d')

        x += pos_embs[:, 0]
        hiddens = [x]

        for ind, (attn, ff, layer_axial_attn) in enumerate(self.layers):

            x = attn(x, mask = mask) + x
            x = ff(x) + x

            x += pos_embs[:, ind + 1]
            hiddens.append(x)

            if exists(layer_axial_attn):
                num_layers = len(hiddens)
                layer_tokens = rearrange(torch.stack(hiddens), 'l b n d -> (b n) l d')

                attended_tokens = layer_axial_attn(layer_tokens)
                attended_tokens = rearrange(attended_tokens, '(b n) l d -> b n l d', b = b)
                pooled_attended_tokens = attended_tokens.max(dim = -2).values
                x += pooled_attended_tokens

        return x

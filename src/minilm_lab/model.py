"""A compact decoder-only Transformer implemented with standard PyTorch layers."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class MiniLMConfig:
    vocab_size: int = 256
    context_length: int = 128
    embedding_dim: int = 192
    num_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.0
    positional_encoding: str = "learned"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniLMConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embedding_dim, config.num_heads, config.dropout, batch_first=True
        )
        self.register_buffer(
            "mask", torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1).bool()
        )

    def forward_cached(self, x: Tensor, past: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = x.shape
        heads = self.attention.num_heads
        width = x.size(-1) // heads
        qkv = F.linear(x, self.attention.in_proj_weight, self.attention.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, length, heads, width).transpose(1, 2)
        k = k.reshape(batch, length, heads, width).transpose(1, 2)
        v = v.reshape(batch, length, heads, width).transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.attention.out_proj(output), (k, v)
    def _apply_rope(self, x: Tensor) -> Tensor:
        length = x.size(2)
        half = x.size(-1) // 2
        positions = torch.arange(length, device=x.device, dtype=x.dtype)
        frequencies = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=x.dtype) / half))
        angles = positions[:, None] * frequencies[None, :]
        cos, sin = angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
        first, second = x[..., :half], x[..., half : 2 * half]
        rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
        return torch.cat((rotated, x[..., 2 * half :]), dim=-1)

    def forward(self, x: Tensor, use_rope: bool = False) -> Tensor:
        length = x.size(1)
        if use_rope:
            # This path is kept explicit for the ablation; PyTorch's attention
            # still handles masking and optimized kernels.
            qkv = F.linear(x, self.attention.in_proj_weight, self.attention.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            heads = self.attention.num_heads
            width = x.size(-1) // heads
            q = self._apply_rope(q.reshape(x.size(0), length, heads, width).transpose(1, 2))
            k = self._apply_rope(k.reshape(x.size(0), length, heads, width).transpose(1, 2))
            v = v.reshape(x.size(0), length, heads, width).transpose(1, 2)
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(x.size(0), length, -1)
            output = self.attention.out_proj(output)
        else:
            output, _ = self.attention(x, x, x, attn_mask=self.mask[:length, :length], need_weights=False)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: MiniLMConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, use_rope: bool = False) -> Tensor:
        x = x + self.attention(self.norm1(x), use_rope=use_rope)
        return x + self.feed_forward(self.norm2(x))

    def forward_cached(self, x: Tensor, past=None):
        attended, current = self.attention.forward_cached(self.norm1(x), past)
        return x + attended, current


class MiniLM(nn.Module):
    def __init__(self, config: MiniLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniLMConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.embedding_dim)
        self.position_embedding = nn.Embedding(c.context_length, c.embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.num_layers)])
        self.final_norm = nn.LayerNorm(c.embedding_dim)
        self.lm_head = nn.Linear(c.embedding_dim, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        _, length = input_ids.shape
        if length > self.config.context_length:
            raise ValueError("input sequence is longer than context_length")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids)
        if self.config.positional_encoding == "learned":
            x = x + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, use_rope=self.config.positional_encoding == "rope")
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def forward_cached(self, input_ids: Tensor, cache=None):
        past_length = 0 if cache is None else cache[0][0].size(2)
        positions = torch.arange(past_length, past_length + input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids)
        if self.config.positional_encoding == "learned":
            x = x + self.position_embedding(positions)
        next_cache = []
        for index, block in enumerate(self.blocks):
            x, layer_cache = block.forward_cached(x, None if cache is None else cache[index])
            next_cache.append(layer_cache)
        return self.lm_head(self.final_norm(x)), next_cache

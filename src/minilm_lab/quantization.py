"""Portable per-output-channel INT8 weight-only linear reference implementation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Int8WeightLinear(nn.Module):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        weight = source.weight.detach().float()
        scales = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        self.register_buffer("weight_int8", torch.round(weight / scales).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scales)
        if source.bias is not None:
            self.register_buffer("bias", source.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Reference implementation: dequantization makes this portable, but
        # it is not an accelerated INT8 kernel and may increase latency.
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, weight, self.bias)


def quantize_linear_weights(model: nn.Module) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name not in {"lm_head", "out_proj"}:
            setattr(model, name, Int8WeightLinear(child))
        else:
            quantize_linear_weights(child)
    return model


def stored_parameter_bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))

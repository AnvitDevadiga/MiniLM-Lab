import torch
from torch import nn

from minilm_lab.quantization import Int8WeightLinear


def test_weight_only_int8_preserves_linear_output_nearly() -> None:
    torch.manual_seed(4)
    layer = nn.Linear(32, 16)
    compact = Int8WeightLinear(layer)
    inputs = torch.randn(12, 32)
    assert compact.weight_int8.dtype == torch.int8
    assert torch.allclose(layer(inputs), compact(inputs), atol=0.025, rtol=0.05)

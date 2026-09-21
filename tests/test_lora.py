import torch
from torch import nn

from minilm_lab.lora import LoRALinear, trainable_parameter_count


def test_lora_freezes_base_and_trains_adapter() -> None:
    layer = LoRALinear(nn.Linear(8, 8), rank=2)
    assert trainable_parameter_count(layer) == 32
    output = layer(torch.randn(2, 8)).sum()
    output.backward()
    assert layer.base.weight.grad is None
    assert layer.up.weight.grad is not None

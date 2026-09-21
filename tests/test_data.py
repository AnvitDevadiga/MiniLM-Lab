import torch

from minilm_lab.data import split_tokens


def test_split_tokens_preserves_order_and_total() -> None:
    tokens = torch.arange(100)
    train, validation = split_tokens(tokens, validation_fraction=0.2)
    assert torch.equal(torch.cat((train, validation)), tokens)
    assert len(validation) == 20

import torch
import torch.nn.functional as F

from minilm_lab.metrics import evaluate_tokens
from minilm_lab.model import MiniLM, MiniLMConfig


def test_evaluation_counts_every_target_once() -> None:
    torch.manual_seed(9)
    model = MiniLM(MiniLMConfig(vocab_size=16, context_length=4, embedding_dim=16, num_layers=1, num_heads=4)).eval()
    ids = torch.randint(0, 16, (11,))
    expected = 0.0
    for end in range(1, len(ids), 4):
        stop = min(end + 4, len(ids))
        logits, _ = model(ids[end - 1 : stop - 1].unsqueeze(0))
        expected += F.cross_entropy(logits.reshape(-1, 16), ids[end:stop], reduction="sum").item()
    actual = evaluate_tokens(model, ids, "cpu")
    assert actual["validation_tokens"] == 10
    assert abs(actual["validation_loss"] - expected / 10) < 1e-6

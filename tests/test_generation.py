from minilm_lab import ByteTokenizer, MiniLM, MiniLMConfig
from minilm_lab.generation import generate


def test_generation_keeps_prompt() -> None:
    model = MiniLM(MiniLMConfig(vocab_size=256, context_length=16, embedding_dim=16, num_layers=1, num_heads=4))
    result = generate(model, "Hi", ByteTokenizer(), max_new_tokens=3, top_k=10, top_p=0.9)
    assert result.startswith("Hi")


def test_generation_rejects_invalid_inputs() -> None:
    model = MiniLM(MiniLMConfig(vocab_size=256, context_length=16, embedding_dim=16, num_layers=1, num_heads=4))
    tokenizer = ByteTokenizer()
    for kwargs in ({"prompt": ""}, {"prompt": "Hi", "temperature": 0}, {"prompt": "Hi", "top_p": 0}):
        try:
            generate(model, tokenizer=tokenizer, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation input was accepted")

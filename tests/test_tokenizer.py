from minilm_lab.tokenizer import BPETokenizer, ByteTokenizer


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "Hello, café 🌱"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_tokenizer_round_trip() -> None:
    tokenizer = BPETokenizer.train("hello hello hello", vocab_size=270)
    text = "hello hello"
    assert tokenizer.decode(tokenizer.encode(text)) == text

# Experiment Log

Each published experiment should record:

- run name and date
- corpus and tokenizer
- model configuration
- training steps and optimizer settings
- device and software versions
- best and final validation loss/perplexity
- bits per byte when comparing tokenizers
- generation throughput and cache mode
- checkpoint path
- one short conclusion

## Planned runs

| Run | Change | Primary comparison |
|---|---|---|
| byte-10000 | byte tokenizer baseline | reference quality and speed |
| bpe-10000 | 512-token BPE | tokenization trade-off |
| rope-10000 | RoPE positions | positional encoding |
| kv-cache | cached decoding | inference latency |
| lora | adapter fine-tuning | trainable parameter efficiency |
| int8 | dynamic quantization | CPU size and latency |

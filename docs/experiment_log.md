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

## Completed baseline runs

| Run | Change | Primary comparison |
|---|---|---|
| modern-byte-1000 | byte tokenizer, RMSNorm + SwiGLU | quality and speed |
| modern-bpe-1000 | 512-token BPE, RMSNorm + SwiGLU | tokenizer trade-off |
| modern-kv-cache | sliding-window cached decoding | 2.74× CPU speedup |

## Next controlled comparisons

| Run | Change | Primary comparison |
|---|---|---|
| rope-1000 | learned positions vs RoPE | positional encoding |
| lora | adapter fine-tuning | trainable parameter efficiency |
| int8 | dynamic quantization | CPU size and latency |

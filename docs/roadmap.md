# Roadmap

## Milestone 1 — foundations

- [x] repository structure and local-first command-line configuration
- [x] decoder-only Transformer forward pass
- [x] causal masking and next-token loss
- [x] deterministic tests for shapes and causality
- [x] byte-level tokenizer and text dataset

## Milestone 2 — training

- [x] gradient accumulation
- [x] checkpoint save/resume with optimizer state
- [x] validation loss and perplexity
- [x] first small English corpus experiment

## Milestone 3 — inference

- [x] temperature, top-k, and top-p sampling
- [x] KV cache with full-forward equivalence tests
- [x] latency benchmark with JSON output

## Milestone 4 — research-quality comparisons

- [x] learned positions and RoPE implementations
- [ ] matched learned-position versus RoPE experiment
- [ ] AdamW versus SGD
- [ ] context lengths 128/256/512
- [ ] model-size scaling
- [x] LoRA building block
- [x] portable weight-only INT8 comparison, including quality, storage, and latency
- [ ] full fine-tuning versus LoRA quality comparison

Every experiment should have a configuration, a fixed seed, recorded metrics, and a short written conclusion.

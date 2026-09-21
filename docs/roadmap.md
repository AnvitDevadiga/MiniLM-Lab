# Roadmap

## Milestone 1 — foundations

- [x] repository structure and local-first configuration
- [x] decoder-only Transformer forward pass
- [x] causal masking and next-token loss
- [x] deterministic tests for shapes and causality
- [ ] byte-level tokenizer and text dataset

## Milestone 2 — training

- [ ] gradient accumulation and mixed precision where supported
- [ ] checkpoint save/resume
- [ ] validation loss and perplexity
- [ ] first small English corpus experiment

## Milestone 3 — inference

- [ ] temperature, top-k, and top-p sampling
- [ ] KV cache
- [ ] latency and memory benchmarks

## Milestone 4 — research-quality comparisons

- [ ] learned positions versus RoPE
- [ ] AdamW versus SGD
- [ ] context lengths 128/256/512
- [ ] model-size scaling
- [ ] full fine-tuning versus LoRA
- [ ] quantization experiments

Every experiment should have a configuration, a fixed seed, recorded metrics, and a short written conclusion.

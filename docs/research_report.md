# MiniLM Lab: A From-Scratch Transformer Training and Inference Study

**Author:** Anvit Devadiga  
**Status:** Reproducible baseline and systems experiments  
**Hardware:** Apple Silicon MacBook Air using PyTorch MPS when available

## Abstract

MiniLM Lab is a small decoder-only language-model system implemented to understand modern language-model components from first principles. The project includes tokenization, causal self-attention, positional encoding, training, validation, checkpointing, sampling, evaluation, parameter-efficient fine-tuning utilities, quantization experiments, and KV-cache inference.

The central principle is measurement: every meaningful change should be evaluated with a fixed corpus, reproducible configuration, validation metrics, and an explicit explanation of trade-offs.

## System

The current baseline is a four-layer decoder Transformer with 192-dimensional representations, six attention heads, a 128-token context window, LayerNorm, GELU feed-forward blocks, tied input/output embeddings, and AdamW optimization. The default vocabulary is byte-level; a learned 512-token BPE vocabulary is also supported.

Supported components include:

- byte-level and learned BPE tokenization
- causal multi-head self-attention
- learned positional embeddings and RoPE
- gradient accumulation
- checkpoint and best-checkpoint saving
- validation loss, perplexity, and bits-per-byte evaluation
- temperature, top-k, and top-p sampling
- KV-cache incremental generation
- LoRA building block
- CPU dynamic-int8 quantization utility

## Reproduction

```bash
source .venv/bin/activate
python scripts/train_tokenizer.py data/tinyshakespeare.txt --vocab-size 512 --output artifacts/bpe.json
python scripts/train_tiny.py data/tinyshakespeare.txt --steps 10000 --eval-interval 250
python scripts/evaluate.py data/tinyshakespeare.txt --checkpoint artifacts/best_checkpoint.pt
python scripts/generate.py "To be or not to be" --checkpoint artifacts/best_checkpoint.pt --tokens 200 --top-k 40 --top-p 0.95
python scripts/benchmark_generation.py --checkpoint artifacts/best_checkpoint.pt --tokens 200
```

## Results

The preserved experiment summary is available as [experiment_summary.json](../artifacts/experiment_summary.json), with a visual comparison in [tokenizer_comparison.svg](../artifacts/tokenizer_comparison.svg).

### Byte-level baseline

Using the Tiny Shakespeare corpus and 10,000 training steps:

- best logged validation perplexity: **5.14**
- final logged validation perplexity: **5.41**
- generation throughput: approximately **523 tokens/second** on MPS in the recorded run

### BPE experiment

Using the same corpus, model shape, and training budget with a learned 512-token BPE vocabulary:

- best logged validation perplexity: **7.99**
- final logged validation perplexity: **9.40**
- full-split evaluation report: [evaluation.json](../artifacts/evaluation.json)
- full-split perplexity in the saved evaluation: **10.94**
- full-split bits per byte: **2.4814**

Raw perplexity across tokenizers must not be treated as a direct quality ranking because the token units differ. Bits per byte and qualitative generation are more appropriate cross-tokenizer comparisons. The BPE model produced more recognisable word and dialogue structure in qualitative samples, while the byte model achieved lower token-level perplexity.

## Engineering findings

1. A tiny byte-level Transformer can learn strong corpus-specific character and formatting patterns on Apple MPS.
2. Validation metrics fluctuate across stochastic batches, so full-split evaluation is more reliable than a single printed validation batch.
3. BPE preprocessing must be implemented with care: a naive repeated full-corpus merge scan was unacceptably slow, while cached chunk encoding reduced preprocessing to well under a second for this corpus.
4. Best-checkpoint selection matters because the final training step is not always the best validation step.
5. KV caching changes the inference computation pattern by reusing previous keys and values instead of recomputing the entire context for every generated token.

## Limitations

This is an educational and research baseline, not a general-purpose assistant. Tiny Shakespeare is a small, narrow corpus; the model is not expected to produce reliable factual answers or broad English knowledge. The current LoRA and quantization components are experiment utilities and require dedicated end-to-end benchmark runs before strong conclusions can be made. Cross-tokenizer comparisons require additional normalization and should not rely on perplexity alone.

## Next experiments

- compare learned positions against RoPE with matched seeds and budgets
- benchmark uncached versus KV-cached generation on identical prompts
- run full and LoRA fine-tuning comparisons
- measure dynamic-int8 CPU size, latency, and quality impact
- compare 10M, 20M, and larger models within Mac memory limits
- publish plots from `artifacts/metrics.json` and the evaluation reports

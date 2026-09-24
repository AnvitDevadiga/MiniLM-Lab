# Implementation audit

## Baseline audit

The original repository had the right breadth but several claims were ahead of the evidence. The most important gaps were correctness and reproducibility, not missing prose.

| Finding | Impact | Resolution |
| --- | --- | --- |
| Cached attention reshaped projected tensors twice | cached logits diverged from full forward | fixed tensor layout and added equivalence tests |
| Cached blocks skipped the MLP sublayer | generation used a different model than training | cache path now runs the complete block |
| RoPE was not applied in cached decoding | RoPE experiments were invalid during generation | cache path now applies the absolute position offset |
| Cache could silently run past the context window | invalid position/cache state | explicit context-limit error |
| Architecture was LayerNorm + GELU only | did not match the stated modern decoder design | RMSNorm + SwiGLU are now the default; legacy options remain |
| Benchmarks only printed terminal text | results were hard to reproduce or plot | benchmark JSON output added |
| Resume loaded after training | `--resume` did not resume the run | retained as a follow-up item until optimizer-state checkpoints are added |

## Verification status

The local project environment passes 12 tests. The key invariant is:

```text
full forward(prompt + token)[-1] == cached forward(prompt) → cached forward(token)[-1]
```

This is tested for both learned positional embeddings and RoPE. Exact throughput and quality numbers remain hardware- and seed-dependent; they must be regenerated after architecture changes.

## Honest limitations

- The bundled corpus is intentionally small and narrow.
- The BPE trainer is educational, not a production tokenizer.
- Dynamic INT8 is a CPU comparison utility. PyTorch is migrating quantization work toward torchao; the old API should not be presented as a production deployment path.
- MPS mixed precision should be benchmarked on the target machine rather than assumed. See the [PyTorch MPS notes](https://docs.pytorch.org/docs/main/notes/mps.html).

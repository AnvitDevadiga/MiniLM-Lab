# Experiment log

## Science prose baseline · 2026-09-25

Training: Darwin + Einstein, byte tokenizer, 4 layers, width 192, six heads, context 128, AdamW 3e-4, seed 42, 1,000 optimizer updates with four microbatches of eight sequences each. CPU execution. Both books have held-out final paragraphs.

| Step | Train loss | Full held-out loss |
| ---: | ---: | ---: |
| 250 | 3.0327 | 3.2751 |
| 500 | 3.0437 | 3.2797 |
| 750 | 3.0581 | 3.2734 |
| 1,000 | 2.6655 | 2.8669 |

Best checkpoint: step 1,000. Full evaluator: 17.58 perplexity, 4.136 bits per byte, 112,111 scored targets. KV-cache decode: 3.58× median speedup over seven CPU trials. Portable INT8 MLP weights: 53.6% fewer stored bytes, +0.00015 validation loss, slower forward passes.

See [the raw run record](data/science_run.json) and [source manifest](../data/science/manifest.json).

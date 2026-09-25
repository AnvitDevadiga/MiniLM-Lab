# Where MiniLM Lab stands in 2026

MiniLM Lab is a compact, inspectable systems study. It does not claim to be a state-of-the-art model, a complete training platform, or the first Transformer implementation online.

The comparison set is substantial. [nanochat](https://github.com/karpathy/nanochat) covers tokenization, pretraining, fine-tuning, evaluation and inference at much larger capability and compute budgets. [LitGPT](https://github.com/Lightning-AI/litgpt) provides validated recipes for many architectures, multiple precision modes, LoRA, quantization and deployment. [torchao](https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html) provides real quantization backends. MiniLM Lab should not claim parity with those systems.

Its portfolio value is narrower and concrete: one engineer can explain every model choice, detect numerical or measurement errors, reproduce a small public-data run, and show honest latency and storage trade-offs. The current research result is a 1 MB science-prose experiment with a held-out passage test, 15 correctness tests, raw timing samples and two figures generated from tracked data.

To move closer to research quality, priority work is:

1. Three or more seeds and uncertainty summaries for quality comparisons.
2. A test book by a third author that never appears in training.
3. M4 MPS measurements with memory and energy constraints.
4. Matched RoPE, attention, scaling and LoRA ablations with fixed token budgets.
5. An accelerated quantization backend where the hardware supports it.

Until those experiments exist, the present results are a well-documented baseline. More visual polish cannot substitute for that evidence.

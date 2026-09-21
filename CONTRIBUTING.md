# Contributing

MiniLM Lab values small, reproducible experiments.

Before opening a pull request:

```bash
source .venv/bin/activate
ruff check .
pytest -q
```

For model or training changes, include the configuration, seed, corpus, device, validation metrics, and a short explanation of the result. Do not commit checkpoints, virtual environments, or generated run directories.

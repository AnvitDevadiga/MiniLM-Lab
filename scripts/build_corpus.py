"""Assemble a reproducible MiniLM systems corpus from the repository itself."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/minilm_systems.txt"
SOURCES = (
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "configs/tiny.yaml",
    "docs/experiment_log.md",
    "docs/experiment_protocol.md",
    "docs/implementation_audit.md",
    "docs/research_report.md",
    "docs/roadmap.md",
    "src/minilm_lab/data.py",
    "src/minilm_lab/generation.py",
    "src/minilm_lab/kv_cache.py",
    "src/minilm_lab/lora.py",
    "src/minilm_lab/model.py",
    "src/minilm_lab/tokenizer.py",
)


def main() -> None:
    sections = []
    for relative_path in SOURCES:
        path = ROOT / relative_path
        sections.append(f"\n===== {relative_path} =====\n\n{path.read_text(encoding='utf-8')}\n")
    OUTPUT.write_text("".join(sections), encoding="utf-8")
    print(f"wrote {OUTPUT} from {len(SOURCES)} sources")


if __name__ == "__main__":
    main()

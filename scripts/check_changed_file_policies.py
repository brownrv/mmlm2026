from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AGENTS = "AGENTS.md"
CLAUDE = "CLAUDE.md"
MASTER_SPELLINGS = "data/TeamSpellings.csv"
RAW_M_SPELLINGS = "data/raw/march-machine-learning-mania-2026/MTeamSpellings.csv"
RAW_W_SPELLINGS = "data/raw/march-machine-learning-mania-2026/WTeamSpellings.csv"


def changed_files(base: str, head: str) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI checks for doc/data policy drift.")
    parser.add_argument("--base", required=True, help="Base git SHA.")
    parser.add_argument("--head", required=True, help="Head git SHA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = changed_files(args.base, args.head)

    errors: list[str] = []

    agents_changed = AGENTS in files
    claude_changed = CLAUDE in files
    if agents_changed ^ claude_changed:
        errors.append("AGENTS.md and CLAUDE.md must be updated together in the same change set.")

    raw_spellings_changed = RAW_M_SPELLINGS in files or RAW_W_SPELLINGS in files
    master_spellings_changed = MASTER_SPELLINGS in files
    if raw_spellings_changed and not master_spellings_changed:
        errors.append(
            "Kaggle TeamSpellings source changed without updating canonical data/TeamSpellings.csv."
        )

    if errors:
        details = "\n".join(f"- {item}" for item in errors)
        raise SystemExit(f"Policy check failed:\n{details}")

    print("Changed-file policy checks passed.")


if __name__ == "__main__":
    main()

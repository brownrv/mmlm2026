from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def staged_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    if os.getenv("ALLOW_NO_MEMORY_UPDATE") == "1":
        print("Bypass enabled via ALLOW_NO_MEMORY_UPDATE=1")
        return

    changed = staged_files()
    src_changed = any(path.startswith("src/mmlm2026/") for path in changed)
    memory_changed = any(
        path.startswith("docs/experiments/") or path.startswith("docs/decisions/")
        for path in changed
    )

    if src_changed and not memory_changed:
        raise SystemExit(
            "Memory discipline check failed:\n"
            "- Detected staged changes in src/mmlm2026/\n"
            "- No staged updates in docs/experiments/ or docs/decisions/\n\n"
            "If this is intentionally trivial, bypass with:\n"
            "ALLOW_NO_MEMORY_UPDATE=1 git commit ...\n"
            "or SKIP=memory-discipline git commit ..."
        )
    print("Memory discipline check passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "decisions" / "0000-template.md"
DECISIONS_DIR = ROOT / "docs" / "decisions"

REQUIRED_HEADINGS = [
    "## Dependencies",
    "## Invalidated by",
    "## Related Experiments",
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def next_decision_index() -> int:
    indices: list[int] = []
    for path in DECISIONS_DIR.glob("[0-9][0-9][0-9][0-9]-*.md"):
        try:
            indices.append(int(path.name[:4]))
        except ValueError:
            continue
    return (max(indices) + 1) if indices else 1


def enforce_required_fields(text: str) -> None:
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
    if missing:
        raise SystemExit(f"Template missing required fields: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new ADR from template.")
    parser.add_argument("title", help="Short decision title.")
    parser.add_argument("--owners", default="repo maintainers", help="Decision owners.")
    parser.add_argument(
        "--status",
        default="Proposed",
        choices=["Proposed", "Accepted", "Superseded", "Rejected"],
        help="Initial decision status.",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    title = args.title.strip()
    if not title:
        raise SystemExit("Title cannot be empty.")

    idx = next_decision_index()
    stem = f"{idx:04d}-{slugify(title)}"
    path = DECISIONS_DIR / f"{stem}.md"
    if path.exists():
        raise SystemExit(f"Decision file already exists: {path}")

    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.replace("# ADR 0000: <short decision title>", f"# ADR {idx:04d}: {title}")
    text = text.replace("Status: Proposed", f"Status: {args.status}")
    text = text.replace("Date: YYYY-MM-DD", f"Date: {args.date}")
    text = text.replace("Owners: <name or team>", f"Owners: {args.owners}")

    enforce_required_fields(text)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Created decision: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

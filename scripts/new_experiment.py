from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "experiments" / "experiment-template.md"
EXPERIMENTS_DIR = ROOT / "docs" / "experiments"
EXPERIMENT_LOG = ROOT / "docs" / "experiments" / "experiment-log.md"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def render_experiment_template(
    template: str,
    title: str,
    owner: str,
    date_str: str,
    status: str,
) -> str:
    text = template
    text = text.replace("<short experiment title>", title)
    text = text.replace("Date: YYYY-MM-DD", f"Date: {date_str}")
    text = text.replace("Owner: <name>", f"Owner: {owner}")
    text = re.sub(
        r"Status: Proposed \| Running \| Completed \| Revisit \| Retired",
        f"Status: {status}",
        text,
    )
    return text


def append_experiment_log(date_str: str, title: str, note_path: Path) -> None:
    block = (
        "\n---\n"
        f"## {date_str} - {title}\n\n"
        "Status: Proposed\n\n"
        "Hypothesis:\n"
        "- <one sentence>\n\n"
        "Dependencies:\n"
        "- rating_model:<version>\n"
        "- sim_engine:<version>\n\n"
        "MLflow:\n"
        "- Run name: <name>\n"
        "- Run ID: <id>\n\n"
        "Result:\n"
        "- <one or two bullets>\n\n"
        "Re-test if:\n"
        "- <trigger condition>\n\n"
        "Related:\n"
        f"- {note_path.as_posix()}\n"
        "- docs/decisions/<file>.md\n"
    )
    with EXPERIMENT_LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(block)


def maybe_create_issue(title: str, note_path: Path, create_issue: bool) -> None:
    if not create_issue:
        return
    body = (
        "## Summary\n"
        "Follow-up task for newly created experiment.\n\n"
        "## Related items\n"
        f"- Experiment note: {note_path.as_posix()}\n"
        "- Decision record:\n"
        "- MLflow run(s):\n"
        "- Git commit(s):\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--title",
                f"[Experiment] {title}",
                "--label",
                "experiment",
                "--body-file",
                str(tmp_path),
            ],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new experiment note and log stub.")
    parser.add_argument("title", help="Short experiment title.")
    parser.add_argument("--owner", default="repo maintainers", help="Experiment owner.")
    parser.add_argument(
        "--status",
        default="Proposed",
        choices=["Proposed", "Running", "Completed", "Revisit", "Retired"],
        help="Initial experiment status.",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--create-issue",
        action="store_true",
        help="Create a linked GitHub issue via gh CLI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    title = args.title.strip()
    if not title:
        raise SystemExit("Title cannot be empty.")

    slug = slugify(title)
    note_path = EXPERIMENTS_DIR / f"{args.date}-{slug}.md"
    if note_path.exists():
        raise SystemExit(f"Experiment note already exists: {note_path}")

    template_text = TEMPLATE.read_text(encoding="utf-8")
    rendered = render_experiment_template(
        template=template_text,
        title=title,
        owner=args.owner,
        date_str=args.date,
        status=args.status,
    )
    note_path.write_text(rendered, encoding="utf-8", newline="\n")
    append_experiment_log(args.date, title, note_path.relative_to(ROOT))
    maybe_create_issue(title, note_path.relative_to(ROOT), args.create_issue)
    print(f"Created experiment note: {note_path.relative_to(ROOT)}")
    print(f"Updated log: {EXPERIMENT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

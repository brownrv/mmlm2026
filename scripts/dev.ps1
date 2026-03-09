param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("sync", "hooks-install", "hooks-run", "test", "lint", "format", "typecheck", "all")]
    [string]$Command
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

switch ($Command) {
    "sync" { uv sync; break }
    "hooks-install" { uv run pre-commit install; break }
    "hooks-run" { uv run pre-commit run --all-files; break }
    "test" { uv run pytest; break }
    "lint" { uv run ruff check .; break }
    "format" { uv run ruff format .; break }
    "typecheck" { uv run mypy src; break }
    "all" {
        uv run ruff format .
        uv run ruff check .
        uv run mypy src
        uv run pytest
        uv run pre-commit run --all-files
        break
    }
}

#!/bin/sh
# Run from the repository root. A failure stops the script immediately.
set -eu
python -m pytest tests/research_ext -q
python scripts/research.py controls --output results/research_ext/exact_controls
python scripts/research.py smoke --output results/research_ext/smoke
if [ "${1:-}" = "--with-lean" ]; then
    python scripts/research.py check-formal --output results/research_ext/formal
fi

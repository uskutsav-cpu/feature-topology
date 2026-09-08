#!/usr/bin/env python3
"""Run from the repository root; no editable-package installation is required."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_ext.cli import main
if __name__=='__main__':
    raise SystemExit(main())

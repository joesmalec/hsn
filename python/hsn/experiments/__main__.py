"""``python -m hsn.experiments``: the sweep command line (:mod:`hsn.experiments.cli`)."""

from __future__ import annotations

import sys

from hsn.experiments.cli import main

if __name__ == "__main__":
    sys.exit(main())

"""Convenience wrapper: BERT-base sweep only."""
from __future__ import annotations

import sys
from . import bench_runner


if __name__ == "__main__":
    sys.argv = sys.argv[:1] + ["--model", "bert"] + sys.argv[1:]
    raise SystemExit(bench_runner.main())

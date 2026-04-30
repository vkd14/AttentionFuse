"""Convenience wrapper: GPT-2 sweep only."""
from __future__ import annotations

import sys
from . import bench_runner


if __name__ == "__main__":
    sys.argv = sys.argv[:1] + ["--model", "gpt2"] + sys.argv[1:]
    raise SystemExit(bench_runner.main())

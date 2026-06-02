"""Experimental kernel paths under active development.

Anything in this package is intentionally NOT wired into the main
runtime dispatch. It is benchmarked side-by-side against the
production kernels via standalone scripts under ``benchmarks/``.

Currently:

* ``hopper_causal_fwd`` — Hopper-targeted causal forward.
  See ``benchmarks/hopper_spike.py``.
"""

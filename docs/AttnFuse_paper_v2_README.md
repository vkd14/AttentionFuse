# AttnFuse Paper v2 — MLSys-ready Overleaf Bundle

Self-contained LaTeX source for the AttnFuse submission, integrating
the Hopper investigation (Sessions 1–8) and all six positioning
improvements:

1. **Structural-novelty as offensive claim** (Introduction, §1)
2. **DSL matches hand-CUDA on Llama-3 headline** (§10, End-to-End)
3. **Rotation Calculus as a named principle** (§8)
4. **Negative Result sidebar** (§9, with bisected counter delta)
5. **H100 ablation table** (§7.3, Table 4)
6. **Workload positioning matrix** (§11)

## Build

Tested with `pdflatex`. Two passes for cross-references and the
bibliography:

```bash
pdflatex AttnFuse_paper_v2.tex
pdflatex AttnFuse_paper_v2.tex
```

On Overleaf: New Project → Upload Project → upload this zip. Set
the main document to `AttnFuse_paper_v2.tex`. Hit Recompile.

## File manifest

| File | Purpose |
|---|---|
| `AttnFuse_paper_v2.tex` | The paper (~750 lines, twocolumn 10pt) |
| `fig_architecture.png` | Five-layer pipeline diagram (Figure 1) |

## Headline numbers (for quick reference)

| Claim | Number | Source |
|---|---|---|
| 3090 fused-RoPE causal+RoPE at N=4096 | **2.10× vs flex+rotate** | Table 9 (`tab:rope_comp`) |
| H100 plain causal closure | from 2.10× behind to **1.10× behind** | Table 4 (`tab:hopper_ablation`) |
| H100 vs flex HMMA ceiling | 29.4% measured vs 32.6% flex; spike within **3.2 pp** | §7.1, §7.3 |
| Rotation Calculus crossover on H100 | between N=4k and N=8k | Table 5 (`tab:rotation_calculus`) |
| Llama-3-8B H100 forward vs SDPA | **51 µs** gap at N=2048 (2.871 vs 2.820 ms) | Table 7 (`tab:llama3_e2e`) |
| Llama-3-8B H100 training vs SDPA | within **5%** at N=2048 | Table 7 |
| Llama-3-8B 3090 training vs SDPA | **1.06×** SDPA (flex OOMs) | Table 7 |
| Flash Decoding Llama-3-70B 32k | **17×** over unsplit AttnFuse | §5 + Table 10 |
| Block-sparse BigBird at N=4096 | **3.21×** vs flex | §6 + §12.5 |

## Notable positioning choices

- **Abstraction-level framing**: claim is "different abstraction by
  design" rather than "competitor lacks feature".
- **Negative Result is a feature**: the Triton 3.3 half-swap
  lowering finding is presented as a methodological contribution,
  not buried in a footnote.
- **Workload Positioning Matrix replaces flat speedup lists**:
  reviewers get explicit guidance about where AttnFuse fits.
- **H100 work framed as principled investigation**: five-session
  structured story (initial gap → NCU diagnosis → sweep → spike →
  RoPE extension → calculus → negative result → E2E).

## Limitations transparently surfaced

- Hopper spike covers causal MHA/GQA only at this version.
- Long-context (N≥8k) RoPE crosses over to pre-rotation as the
  Rotation Calculus predicts.
- Multi-GPU, fp8, and FA-3 producer/consumer are explicit future
  work items.

These are documented in §13.

## After submission

The repo's `docs/manuscript_updates_h100.tex` contains the working
notes from the Hopper investigation that drove this revision. Keep
both for diffability during reviewer revisions.

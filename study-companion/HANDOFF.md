# Handoff — how to continue this series

Read this first in a new session. It exists so the next run starts building within
minutes instead of re-deriving decisions already made.

## State

Branch `claude/blissful-euler-0c6xih`, latest commit carries sheets 1–3 of 20.

| # | Chapter | Folder | Published |
|---|---------|--------|-----------|
| 01 | Vectors | `ch01-vectors/` | https://claude.ai/artifact/98j4cnE4QvTKi1znhH6vh9 |
| 02 | Matrices | `ch02-matrices/` | https://claude.ai/artifact/SaJmwkXPKYmaJ6Eet8uBKd |
| 03 | Calculus | `ch03-calculus/` | https://claude.ai/artifact/DhK5wG1RE9YMnRxSHizngC |

Next up: **Chapter 04 — Statistics**.

## The reader

A mechanical engineering graduate of a tier-3 Indian college, GATE-prepared, no CS
background. Assume fluency in: engineering mechanics, strength of materials, SFD/BMD,
Mohr's circle, FEM basics, vibration, thermodynamics, fluid mechanics, heat transfer,
engineering maths. Assume **zero** programming or CS vocabulary.

## Non-negotiable rules

1. **Never trade truth for accessibility.** Every analogy carries a
   *"Where the analogy breaks"* box in amber. An analogy whose edge the reader cannot
   see is a trap.
2. **Prefer derivable claims to citable ones.** If a correspondence can be proved, prove
   it; if it is merely structural, say so. Add every new numeric claim to
   `verify_claims.py` and run it — that is what caught the pendulum error.
3. **Hedge what cannot be computed.** Research results and facts that go stale are
   marked as claims, not asserted.
4. **Provenance is mandatory.** Apache-2.0 §4(b) requires stating changes. Every chapter
   ends with Appendix A; every section chip links to the source pinned at `9850ee5`.
5. No internet research so far. If it is ever needed, use primary sources (arXiv IDs),
   never blog posts, and mark those sentences visibly as reported rather than derived.

## Build procedure

```bash
# 1. read the whole source chapter first - all five files
cd "chapter 04 - statistics" && wc -l *.md && cat *.md

# 2. list the figures it uses, copy them to the workspace
grep -rho "images/[a-z0-9_]*\.svg" *.md | sed 's|images/||' | sort -u

# 3. reuse the design system verbatim: lines 1-339 of any existing chapter are the
#    complete head + token block. Change only the <title>.
sed -n '1,339p' study-companion/ch01-vectors/index.html | sed 's|<title>.*</title>|<title>NEW NAME</title>|' > "$W/index.html"

# 4. write the body in 5-6 appended heredoc chunks (quoted delimiter - the content
#    contains backslashes and dollar signs)

# 5. screenshot ONCE with playwright, make one pass of edits, publish
```

Load these skills before writing: `artifact-design`, `artifact-diagramming`, and
`dataviz` if the chapter has any chart. Validate any categorical palette with
`dataviz/scripts/validate_palette.js` for **both** `--mode light` and `--mode dark`
(bands differ: light 0.43–0.77, dark 0.48–0.67).

## Chapter skeleton that works

`00` bridge table (GATE ↔ ML ↔ honest verdict) · `01…n` content sections, each with
break boxes and 2–3 interactive labs · `n+1` worked example solved twice, once as
mechanics once as ML · `n+2` shadow reading (10 recall prompts, `<details>`) ·
`n+3` concept map SVG + what this feeds · `A` provenance appendix.

Interactive labs are plain SVG + vanilla JS, no libraries. Maths is MathJax **SVG
output** from cdnjs (`tex-mml-svg.js`) — the CHTML build fetches fonts, which the CSP
blocks. A TeX-to-Unicode fallback and an equation auto-fitter are appended to every
chapter; copy both.

## Bridge bank — pre-worked, not yet written up

The hardest part of each chapter is finding correspondences that are *exact* rather than
decorative. These are the strongest ones found so far for the remaining chapters.

**04 Statistics.** Mean is the centroid of the density, `x̄ = ∫x dA / ∫dA` — a
calculation already done for section properties. Variance is the **second moment of
area**, and standard deviation is the **radius of gyration**, `k = √(I/A)`: exact, and
the single best bridge in the chapter. Central limit theorem ↔ RSS tolerance stack-up in
manufacturing. Sampling ↔ acceptance sampling and QC lots. Hypothesis testing ↔ Six
Sigma, control charts, Type I/II error ↔ producer's and consumer's risk.
*Break:* sample statistics are estimates with their own uncertainty; a section property
is exact.

**05 Probability.** Reliability engineering is the whole chapter: MTBF, the exponential
distribution, **Weibull** for fatigue and wear-out, the bathtub curve. Bayes ↔ condition
monitoring and fault diagnosis. Monte Carlo ↔ tolerance analysis.
*Break:* Shannon entropy and thermodynamic entropy are genuinely connected via
`S = k ln W`, but the units and the claims differ — handle carefully, do not assert they
are the same thing.

**06 Machine learning.** Regression ↔ least-squares fit to experimental data.
Overfitting ↔ a high-order polynomial through scattered points. **Bias–variance ↔
accuracy versus precision** in measurement — exact and intuitive. Regularisation ↔ the
L1/L2 norms of Ch01 §04. Cross-validation ↔ holdout test specimens.

**07 Computational linguistics.** Attention is the duality pairing of Ch01 §08.
Embeddings are the feature vectors of Ch01 §02. Positional encoding ↔ Fourier series.
RNNs ↔ difference equations and discrete state-space models.

**08 Computer vision.** Convolution ↔ the Toeplitz matrix of Ch02 §04 and ↔ impulse
response in signal processing. Pooling and downsampling ↔ **Nyquist and aliasing**.
Image gradients ↔ spatial derivatives.

**09 Audio and speech.** The strongest chapter for this reader. FFT, sampling, Nyquist,
spectrograms and windowing *are* machinery vibration analysis: order tracking, envelope
analysis for bearing faults, accelerometer FFTs. Only the mel scale is new.

**10 Multimodal.** Fusion ↔ sensor fusion; Kalman filtering of accelerometer plus gyro.

**11 Autonomous systems.** Mostly mechanical already: DH matrices (Ch02 §06), the
manipulator Jacobian (Ch02 §03, Ch03 §03), control loops, state estimation.

**12 Graph neural networks.** A finite element mesh **is** a graph, and the graph
Laplacian is assembled exactly as a stiffness matrix is. Laplacian eigenvectors ↔ mode
shapes. Message passing ↔ Gauss–Seidel and other iterative solvers. Very strong.

**13 Computing and OS.** Pipelining ↔ an assembly line. **Amdahl's law ↔ theory of
constraints** and bottleneck analysis. Caching ↔ buffer stock and material staging.

**14 Data structures and algorithms.** Big-O ↔ scaling laws. Trees ↔ bill-of-materials
hierarchies. Graphs ↔ pipe networks and assembly precedence diagrams.

**15 Production software engineering.** Version control ↔ **drawing revision control**,
ECN/ECO. Testing ↔ inspection and QA. CI/CD ↔ line automation. Containers ↔ standardised
jigs and fixtures.

**16 SIMD and GPU.** Where Ch02's matmul lands. SIMD ↔ multi-spindle machining. Memory
bandwidth ↔ material-flow bottleneck.

**17 AI inference.** Quantisation ↔ measurement resolution and significant figures.
**Batching ↔ batch production versus one-piece flow** in lean manufacturing; latency
versus throughput ↔ cycle time versus takt time. Strong.

**18 ML systems design.** Plant design and capacity planning. **Little's law** is already
used in manufacturing. Redundancy ↔ reliability engineering.

**19–20.** Outline stubs in the source (six files are empty). Either skip, or write them
as genuine surveys and mark them clearly as not derived from the compendium.

## Known issues

- The equation auto-fitter could not be verified against a real typesetter in this
  environment, only against a stub. If a reader reports a clipped equation, split it at
  source with `\begin{aligned}`.
- Inline source panels (approach C) exist in Chapter 01 only, as a trial.
- Upstream bugs found but **not** fixed, because they were out of scope: `mkdocs.yml`
  has `docs_dir: .` which aborts the build on mkdocs 1.6, and all 104 nav paths plus all
  104 `llms.txt` paths still use the pre-rename `chapter NN:` spelling. Worth raising
  upstream.

# Study Companion — for mechanical engineers

Interactive HTML companions to the *Maths, CS & AI Compendium*, rebuilt for a
specific reader: **a mechanical engineering graduate with GATE-level mathematics
and no computer science background.**

Each page is a standalone HTML file. Open it in any browser — no build step, no
server, no dependencies to install.

## The approach

The reader already owns a great deal of the material under different names. Four
years of engineering mechanics, strength of materials and engineering mathematics
built real intuition; the job is to connect it, not to start over.

So every chapter page does two things:

1. **Hands over the dictionary.** Each tool drilled for GATE, placed next to the
   name machine learning gives it — resolving forces next to feature vectors,
   Mohr's circle next to PCA, virtual work next to duality.

2. **Marks where the dictionary fails.** Every analogy carries a
   *"Where the analogy breaks"* box. This is not decoration. An analogy whose
   edge you cannot see is a trap, and the points where mechanical intuition
   stops working are exactly where machine learning starts doing something new.

**Rigour is never traded for accessibility.** Definitions, axioms and derivations
are stated in full. Where a correspondence is exact it is proved; where it is
merely structural, that is said plainly.

> **Continuing this work?** Read [](HANDOFF.md) first — it carries the build
> procedure, the rules, and a bank of pre-worked mechanical-engineering bridges for the
> remaining chapters.

## Chapters

| # | Chapter | Page | Published |
|---|---------|------|-----------|
| 01 | Vectors | [`ch01-vectors/`](ch01-vectors/index.html) | [Read online](https://claude.ai/artifact/98j4cnE4QvTKi1znhH6vh9) |
| 02 | Matrices | [`ch02-matrices/`](ch02-matrices/index.html) | [Read online](https://claude.ai/artifact/SaJmwkXPKYmaJ6Eet8uBKd) |
| 03 | Calculus | [`ch03-calculus/`](ch03-calculus/index.html) | [Read online](https://claude.ai/artifact/DhK5wG1RE9YMnRxSHizngC) |
| 04 | Statistics | [`ch04-statistics/`](ch04-statistics/index.html) | [Read online](https://claude.ai/artifact/XWetUvsAH8Mw25LRxecCKU) |
| 05 | Probability | [`ch05-probability/`](ch05-probability/index.html) | [Read online](https://claude.ai/artifact/DDbGs67DwJPTwCYWFyTUMW) |
| 06 | Machine Learning | [`ch06-machine-learning/`](ch06-machine-learning/index.html) | [Read online](https://claude.ai/artifact/AazU1pknayKewgpJQhzo3Z) |
| 07 | Computational Linguistics | [`ch07-computational-linguistics/`](ch07-computational-linguistics/index.html) | [Read online](https://claude.ai/artifact/G4qe4sW6y1Av9kVibTq11R) |
| 08 | Computer Vision | [`ch08-computer-vision/`](ch08-computer-vision/index.html) | [Read online](https://claude.ai/artifact/YGYUxoUzmFo2R2W5fb7PP7) |
| 09 | Audio and Speech | [`ch09-audio-speech/`](ch09-audio-speech/index.html) | [Read online](https://claude.ai/artifact/UeVoa1oWJcZJL8KMsbbLHq) |
| 10 | Multimodal Learning | [`ch10-multimodal/`](ch10-multimodal/index.html) | [Read online](https://claude.ai/artifact/VVW3i2z1z8dxCEUsvupMAW) |
| 11 | Autonomous Systems | [`ch11-autonomous-systems/`](ch11-autonomous-systems/index.html) | [Read online](https://claude.ai/artifact/9xX3kBS2wW6MJT8keXAvQA) |

## What's in a chapter page

- **Bridge table** — the GATE-to-ML dictionary, with an honest quality rating per row
- **Interactive panels** — drag-and-solve demonstrations built in plain SVG and JavaScript
- **Break boxes** — the failure point of every analogy, in amber
- **Worked examples** — one problem solved twice, once as mechanics and once as ML
- **Shadow reading** — active-recall prompts matching the compendium's own Phase 2 study method
- **Concept map** — which later chapter each idea feeds

Figures marked *from the compendium* are reproduced from this repository's
`images/` directory; each chapter folder carries its own copy so the page stays
self-contained.

## Provenance, and why you can check everything

This is a derivative work built on someone else's writing, so three mechanisms exist to let
you audit it rather than trust it.

**Linked sources.** Every section header carries a chip naming the source file it was built
from, and that chip is a link to the original on GitHub, pinned to commit `9850ee5` so it
cannot drift as the upstream repository changes.

**Appendix A — provenance and changes.** Each chapter ends with a table listing every
section against its source and a verdict: *faithful*, *expanded*, *added*, or *corrected*.
The `corrected` rows are the ones that matter — they are where this companion departs from
the compendium on a point of fact, with the reason stated on the page. The Apache-2.0
licence requires a statement of changes; this is it.

**Inline source panels (Chapter 01).** Every section in Chapter 01 ends with a
*Compare with the source* panel containing the original compendium text, verbatim and
unedited, for the passage that section covers. This is a trial of the approach; the other
chapters rely on the linked chips instead.

## Verification

`verify_claims.py` checks the load-bearing mathematical claims across all eleven chapters
(349 checks, zero failures)
against independent computation rather than against anybody's memory — the von Mises and
Tresca norm identities, the unit-change nearest-neighbour flip, the worked bracket example,
the `1/sqrt(d)` concentration of cosine similarity, the inertia tensor eigenvalues, polar
decomposition recovered from SVD, the rank of a truss element stiffness matrix, the pendulum
series against the exact elliptic integral, the gradient descent stability threshold, the
beam's Fundamental Theorem check, the BPE merge invariant, the RoPE relative-position
identity, the equivalence of the state-space recurrence and its convolution kernel, and the
proof that speculative decoding leaves the output distribution exactly unchanged, the
structure tensor against the Mohr's-circle principal values, Gaussian blur against an explicit
heat-equation solver, the aliasing fold frequency, the Beer-Lambert form of NeRF
transmittance, and — for Chapter 11 — the closed-form Riccati solution showing a steady-state
Kalman filter has a damping ratio of exactly `1/sqrt(2)` whatever the noise, `det J = l1 l2
sin(q2)` with the lost direction at a singularity proved to be the radial one, the
skew-symmetry of `Mdot - 2C`, and the degree-weighted limit of swarm consensus.

Several checks exist because they caught an error. The suite has overturned claims in the
source material (what neighbour-averaging consensus converges to; whether four legs are
statically stable; the size of a VLA action vocabulary) and claims written for these pages
before they were tested (whether a Kalman filter's damping ratio varies with noise; what
actually compounds in behavioural cloning; what damped least squares does near a
singularity). A test that passes is worth less than one that has failed at least once.

```
pip install numpy scipy
python3 study-companion/verify_claims.py
```

It is not decoration. Running it is what caught a real error in the Chapter 03 pendulum
table, which was printing a truncated series where it claimed to print exact values.

Claims that *cannot* be settled this way — research results such as the prevalence of saddle
points in high dimensions, or facts that go stale such as typical model sizes — are hedged in
the text rather than asserted.

## Technical notes

- Mathematics renders through MathJax (SVG output) loaded from CDN. If the CDN is
  unreachable the page degrades the TeX to readable Unicode rather than showing
  raw markup.
- Light and dark themes are both designed, with a manual toggle in the header.
- Chart palettes are validated for colour-vision deficiency separation and
  contrast in both themes.
- Pages are responsive down to phone width.

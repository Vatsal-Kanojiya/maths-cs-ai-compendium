# Handoff — how to continue this series

Read this first in a new session. It exists so the next run starts building within
minutes instead of re-deriving decisions already made.

## State

Branch `claude/blissful-euler-0c6xih`, latest commit carries sheets 1–7 of 20.

| # | Chapter | Folder | Published |
|---|---------|--------|-----------|
| 01 | Vectors | `ch01-vectors/` | https://claude.ai/artifact/98j4cnE4QvTKi1znhH6vh9 |
| 02 | Matrices | `ch02-matrices/` | https://claude.ai/artifact/SaJmwkXPKYmaJ6Eet8uBKd |
| 03 | Calculus | `ch03-calculus/` | https://claude.ai/artifact/DhK5wG1RE9YMnRxSHizngC |
| 04 | Statistics | `ch04-statistics/` | https://claude.ai/artifact/XWetUvsAH8Mw25LRxecCKU |
| 05 | Probability | `ch05-probability/` | https://claude.ai/artifact/DDbGs67DwJPTwCYWFyTUMW |
| 06 | Machine Learning | `ch06-machine-learning/` | https://claude.ai/artifact/AazU1pknayKewgpJQhzo3Z |
| 07 | Computational Linguistics | `ch07-computational-linguistics/` | *not yet published* |

Next up: **Chapter 08 — Computer Vision**.

Chapter 07 is written, probed and verified but **not published** — the artifact link is the
only thing outstanding. Publish it and fill the row in, here and in `README.md`.

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

## Format changes agreed at Chapter 05

The reader reviewed the work and rejected a proposal to shorten it. Their reasoning
was sound and is now policy: compressing to a "bridge sheet" would keep the
correspondences but discard the re-narration that makes them stick, and
optimising words-per-unit-learning risks abstracting away the truth. **Length is
not the constraint. Value is.** Four changes, starting from Chapter 05:

1. **Prediction-first labs.** Every lab opens with a `.predict` panel asking the
   reader to commit to an answer before the widget reveals it. This converts a
   passive demo into retrieval practice for almost no extra cost, and it is the
   single highest-value change made so far.
2. **A real question bank** — 24 prompts in three tiers (recall / apply / judge)
   rather than 10 flat ones. The judge tier is where the thinking is.
3. **A "what this chapter does not cover" section**, naming honestly what a full
   treatment would include and this one omits. The reader was sceptical that
   twenty chapters could cover the field; they are right, and the page should say
   so rather than imply completeness.
4. **Drop decorative labs.** A widget that teaches a mechanical engineer something
   they already know cold (the parallelogram of forces, in Chapter 01) is
   decoration. Better labs, not more of them.

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

- ~~The equation auto-fitter could not be verified against a real typesetter.~~
  **Resolved at Chapter 06** — see "Browser probe" below. Equations are now checked
  against real MathJax at five viewport widths.
- Inline source panels (approach C) exist in Chapter 01 only, as a trial.
- ~~Upstream bugs found but **not** fixed: `mkdocs.yml` has `docs_dir: .` which aborts the
  build on mkdocs 1.6, and all 104 nav paths plus all 104 `llms.txt` paths still use the
  pre-rename `chapter NN:` spelling.~~ **Both fixed** on this branch. `docs_dir` now points
  at the `docs/` symlink farm the deploy workflow already builds, and all 208 paths were
  rewritten. Verified with mkdocs 1.6.1: the build went from aborting to emitting 105 pages
  with no nav warnings. Not yet contributed upstream to `HenryNdubuaku/…`, which is a
  separate PR against a different repository.


## Browser probe — set this up before touching a chapter (added at Chapter 06)

cdnjs is blocked by the egress proxy, so MathJax will not load in a headless browser
and every equation check silently passes on unrendered TeX. npm is *not* blocked:

```bash
cd "$SCRATCH" && ln -sfn /opt/node22/lib/node_modules node_modules
(cd vendor && npm pack mathjax@3.2.2 --silent && tar xzf mathjax-3.2.2.tgz)
cp -r vendor/package/es5 chNN/vendor          # the WHOLE es5 tree, not just the
                                              # bundle: it lazy-loads extensions
python3 -c "import io,re; s=io.open('chNN/index.html').read(); \
  io.open('chNN/_probe.html','w').write(re.sub(r'https://cdnjs[^\"]+','vendor/tex-mml-svg.js',s))"
npx --offline http-server chNN -p 8607 -s &   # then point Playwright at _probe.html
```

Do not ship `vendor/` or `_probe.html`; they are probe-only. Simplest is to build the whole
probe in a scratch directory — copy `index.html` and `img/` there, add `vendor/`, and the
repository never sees either.

Environment specifics that cost time at Chapter 07:

- Chromium is at `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`. The unversioned
  `/opt/pw-browsers/chromium/` path in `PLAYWRIGHT_BROWSERS_PATH` is **not** a launchable
  binary. Playwright itself is `/opt/node22/lib/node_modules/playwright`.
- **Google Fonts is blocked too**, not just cdnjs — `fonts.googleapis.com` fails with
  `ERR_CERT_AUTHORITY_INVALID`. This is environmental: the already-published Chapter 06
  fails identically when probed. Do not chase it. The stack has `Arial Narrow`/Georgia
  fallbacks and degrades cleanly.
- `curl` to the probe server needs `--noproxy '*'`, and `http-server` needs a moment before
  its first request succeeds — a readiness loop with no delay will report connection refused
  against a server that is starting fine.

What the probe must check, at widths 1180 / 900 / 700 / 480 / 390:

1. `mjx-container` count > 100 — otherwise MathJax did not run and nothing below counts.
2. No display equation wider than its `.mathbox`.
3. `documentElement.scrollWidth == clientWidth` — no horizontal page scroll.
4. No `.ro-k` / `.lg` label with `scrollWidth > clientWidth` (clipped label).
5. Every slider swept end to end, every preset clicked, every predict button clicked,
   with `pageerror` collected throughout.

## Traps that have now bitten more than once

- **Greek letters inside uppercased labels.** `.ro-k`, `<label>`, `.lab-t`, `.note-h`
  and `thead th` all carry `text-transform:uppercase`, which turns `&beta;` into a
  capital Beta — the Latin letter **B** on screen. Likewise `&eta;`→H, `&kappa;`→K,
  `&lambda;`→Λ. Sliders read "MOMENTUM B", "STEP H". Every Greek entity in those
  elements must be wrapped in `<span class="gk">`, and `.gk{text-transform:none}`
  must be in the stylesheet. This was live in all five published chapters before
  Chapter 06 caught it; 31 labels were fixed retrospectively.
- **The head-extraction recipe drops `</style>`.** `sed -n '1,339p'` ends *inside*
  the first `<style>` block. **The correct boundary is `sed -n '1,385p'`** — line 352
  closes the token block and 353–385 is a second, equally necessary block (`.srcref`,
  `.predict`, `.pbtn`, `.flag`, `.gap`). Every chapter on disk is balanced now, so the
  earlier damage has been repaired; keep asserting
  `count('<style>') == count('</style>')` before publishing anyway.
- **Long *inline* math forces page scroll.** The auto-fitter only ever looked at
  `display="true"` containers inside `.mathbox`. Inline math cannot wrap and is not in
  a scrollable box, so one long `\(...\)` pushes the whole page sideways. `fitInline()`
  now handles it — keep it when copying the tail into a new chapter. Better still,
  promote any inline expression longer than about 40 characters to a display equation.
- **A `\qquad`-joined pair of equations overflows below ~480px.** Split with
  `\begin{aligned}` and `&=` alignment; it reads better on desktop anyway.
- **An `<img>` outside `.paper` overflows the page.** There is no bare `img{max-width:100%}`
  rule in the design system — only `.paper img`. A compendium figure written as
  `<figure><img …></figure>` renders at its intrinsic SVG width and forces horizontal page
  scroll at 700px and below. The pattern is
  `<figure><div class="paper"><img … width="410"></div><figcaption>…</figcaption></figure>`,
  and the caption convention is `<b>Fig. N.M</b> — text <i>Figure from the compendium.</i>`
  This cost a full probe cycle at Chapter 07.
- **`.mono` is table-only.** It is defined as `td.mono,th.mono`, so `<span class="mono">` in
  prose renders unstyled. There is zero precedent for it in chapters 01–06. Inline code in
  prose is `<code>`, which is styled and which MathJax skips.
- **The uppercased set is 17 selectors, not five.** The full list that carries
  `text-transform:uppercase`, and therefore needs `.gk` around any Greek entity, is:
  `.btn`, `.ctl label`, `.defn-h`, `.eyebrow`, `.flag`, `.gap-h`, `.lab-tag`, `.lab-t`,
  `.note-h`, `.predict-h`, `.pv`, `.rail-h`, `.ro-k`, `.tb-k`, `.themebtn`, `h4`, `thead th`.
  `.pbtn` is *not* uppercased, so Greek in a predict button is safe.
- **A list of numbers in inline math will overflow a phone.** Eight six-decimal values render
  about 434px wide and cannot wrap. The ~40-character rule is about *rendered* width, not
  source length — `\min\!\left(1,p_t/p_d\right)` is 78 source characters and renders short,
  while a number list is short in source and wide on screen. Promote number lists to display.
- **Ship only the figures the page uses.** Chapters 05 and 06 both ship exactly as many SVGs
  as they reference. Copy the chapter's full figure set into the workspace while drafting,
  then prune before committing.

## The Chapter 06 lesson worth generalising

The momentum lab was built to illustrate "momentum cancels the transverse oscillation",
which is how every textbook draws it. Simulating it showed the claim is **false at the
optimal tuning**: the momentum path flips the sign of its stiff coordinate on every
single step, exactly as plain descent does, and overshoots *further* (1.71 vs 1.00). The
gain is a faster decay per crossing (0.62 vs 0.90) bought by a step size that plain
descent could not survive.

Nothing in the prose was wrong in a way a reader could catch. It took building the thing
and running it. **If a page asserts a picture, make the lab draw that picture, and if the
lab draws something else, the prose is what is wrong.** The corrected version is in
§05 and break item 5, and the numbers are in `verify_claims.py` under `[Ch06 s05]`.

## The Chapter 07 lesson worth generalising

Chapter 07 looked, going in, like the chapter with no mechanical content. It turned out to
have the most exact correspondence in the series so far: a state-space model *is* a
state-space model, and the SSM convolution kernel is the discrete impulse response, verified
to `7e-18`. The lesson is not "look harder for analogies" — it is that the strongest bridges
were found by reading the *newest* material, not the oldest. Mamba (2023) landed on control
theory; the 1950s linguistics in source file 01 has no bridge at all.

The second lesson is sharper and is now the chapter's own summary: **the bridge is exact
right up to the point where the architecture becomes good, and then it ends.** Everything in
§07 is linear and time-invariant, so poles, transfer functions and superposition all apply —
until selectivity makes `B`, `C` and `Δ` input-dependent, at which point every one of those
tools fails. A bridge that holds only for the linear, non-learned special case is worth
stating *and* worth bounding, in the same breath. The break box is not a disclaimer attached
to a good analogy; it is half the content.

Third, the correction in §13 is the kind the numeric checks exist to catch. The tempting
claim — that the SSM kernel is the textbook impulse response `h(t)` sampled and scaled by
`Δ` — is false, because a discrete unit sample is a pulse held for one step, not a Dirac
delta. The kernel is non-zero at `j=0` where `h(0)=0`, and the ratio only converges to 1
after several steps. Nothing in the prose would have caught this; computing both columns did.
The true statement (kernel == *discrete* impulse response, exactly) is stronger and more
useful than the false one, which is usually how these go.

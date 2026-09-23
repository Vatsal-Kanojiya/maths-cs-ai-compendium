# Handoff — how to continue this series

Read this first in a new session. It exists so the next run starts building within
minutes instead of re-deriving decisions already made.

## State

Branch `claude/blissful-euler-0c6xih`, latest commit `ccdbfbc` carries sheets 1–12 of 20.
**All twelve are published.** Working tree clean, local matches remote.

| # | Chapter | Folder | Published |
|---|---------|--------|-----------|
| 01 | Vectors | `ch01-vectors/` | https://claude.ai/artifact/98j4cnE4QvTKi1znhH6vh9 |
| 02 | Matrices | `ch02-matrices/` | https://claude.ai/artifact/SaJmwkXPKYmaJ6Eet8uBKd |
| 03 | Calculus | `ch03-calculus/` | https://claude.ai/artifact/DhK5wG1RE9YMnRxSHizngC |
| 04 | Statistics | `ch04-statistics/` | https://claude.ai/artifact/XWetUvsAH8Mw25LRxecCKU |
| 05 | Probability | `ch05-probability/` | https://claude.ai/artifact/DDbGs67DwJPTwCYWFyTUMW |
| 06 | Machine Learning | `ch06-machine-learning/` | https://claude.ai/artifact/AazU1pknayKewgpJQhzo3Z |
| 07 | Computational Linguistics | `ch07-computational-linguistics/` | https://claude.ai/artifact/G4qe4sW6y1Av9kVibTq11R |
| 08 | Computer Vision | `ch08-computer-vision/` | https://claude.ai/artifact/YGYUxoUzmFo2R2W5fb7PP7 |
| 09 | Audio and Speech | `ch09-audio-speech/` | https://claude.ai/artifact/UeVoa1oWJcZJL8KMsbbLHq |
| 10 | Multimodal Learning | `ch10-multimodal/` | https://claude.ai/artifact/VVW3i2z1z8dxCEUsvupMAW |
| 11 | Autonomous Systems | `ch11-autonomous-systems/` | https://claude.ai/artifact/9xX3kBS2wW6MJT8keXAvQA |
| 12 | Graph Neural Networks | `ch12-graph-neural-networks/` | https://claude.ai/artifact/6PoKo8b3Hq3QzVQvRDJT8u |

`verify_claims.py` stands at **398 checks, 0 failures**, covering chapters 01–12. Run it first
in any new session — it is the fastest way to confirm nothing has rotted. It needs numpy and
scipy, which a fresh container will not have: `pip install numpy scipy`.

Next up: **Chapter 13 — Computing and Operating Systems**. Source is
`chapter 13 - computing and OS/`, 1,243 lines across 5 files — one run, no split.

**Chapter 13 inherits a promise, the way 12 inherited one from 11.** Chapter 12 §03 argued that
a CNN is to a GNN what a finite-difference stencil on a structured grid is to finite element
assembly on an unstructured mesh, and §27 says explicitly that Chapter 13 "inherits the
sparse-matrix problem" — indirect indexing, cache misses, irregular access. Chapter 12 also
promised Sheet 13 would be "where the sparse-matrix bill for everything above comes due", in
its closing endnote. Pick that up rather than starting cold.

### Session hygiene that matters

- **Artifact watch limit.** A session can hold at most 10 artifact wake-subscriptions. This
  session hit the cap, so Chapter 10 is published but unwatched. Harmless, but do not claim
  a chapter is being watched without checking the publish result.
- **Container restarts wipe the filesystem but not the branch.** This happened twice. Both
  times the local clone still had the commits and the remote matched; the only casualty was
  the remote-tracking ref, which `git fetch origin <branch>` restores. Push after every
  chapter and nothing is ever at risk.
- **Background http-servers do not survive a restart** and do not need to be restarted unless
  you are mid-probe.

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

# 3. reuse the design system verbatim: lines 1-385 of any existing chapter are the
#    complete head + BOTH token blocks. 339 is wrong - it ends inside the first
#    <style>. Change only the <title>. Assert count('<style>')==count('</style>').
sed -n '1,385p' study-companion/ch11-autonomous-systems/index.html | sed 's|<title>.*</title>|<title>NEW NAME</title>|' > "$W/index.html"

# 4. write the body in 5-6 appended heredoc chunks. USE A QUOTED DELIMITER (<<'EOF').
#    An unquoted one turns every $$ into the shell's PID and every \\ into \, which
#    silently destroys all display maths. See the trap of that name below. If you
#    need a variable in the body, substitute it afterwards in Python.

# 5. probe in a real browser (recipe below), screenshot EACH LAB individually,
#    fix, re-probe, publish
```

Load these skills before writing: `artifact-design`, `artifact-diagramming`, and
`dataviz` if the chapter has any chart **or any lab that colours more than two things**.
Validate any categorical palette with `dataviz/scripts/validate_palette.js` for **both**
`--mode light --surface "#FBFCFB"` and `--mode dark --surface "#161A1C"` (bands differ:
light 0.43–0.77, dark 0.48–0.67). Chapter 12 carries a validated `--c1…--c5` block that
passes both — copy it rather than re-deriving one, and read the trap about the stock
`--s1`/`--brk`/`--s3` tokens before using them.

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

**08 Computer vision.** ~~Convolution ↔ the Toeplitz matrix of Ch02 §04. Pooling ↔ Nyquist
and aliasing. Image gradients ↔ spatial derivatives.~~ **Written.** All three held, but none
was the headline. The strongest bridge was not in this bank: the **Harris structure tensor is
a plane stress tensor** — symmetric 2×2, eigenvalues are principal values via Mohr's circle,
and `det`/`trace` are exactly the invariants `I₁,I₂`. It appears twice in the chapter, as
corner detection and again as the matrix Lucas–Kanade inverts for optical flow. Also exact and
not in the bank: **Gaussian blur is the heat equation** with σ²=2t, **a ResNet block is
forward Euler** with h=1, **smooth-L1 is an elastic–perfectly-plastic law**, **U-Net is a
multigrid V-cycle**, **the optical-flow constraint is the material derivative**, and **NeRF
transmittance is Beer–Lambert**. Lesson for the remaining chapters: the bank is a floor, not a
ceiling — read the source before trusting it.

**09 Audio and speech.** ~~FFT, sampling, Nyquist, spectrograms and windowing *are* machinery
vibration analysis. Only the mel scale is new.~~ **Written.** The bank was right, and for once
it was right about the *scale* of the thing: most of the bridge table reads "identical
operation" rather than "analogous", which had not happened before. Accelerometer FFT and
magnitude spectrum are the same transform; a Hanning window on a vibration spectrum is a Hann
window before an FFT for the same reason; a waterfall plot **is** a spectrogram; order tracking
and pitch detection are the same method with the same octave-error failure. Added beyond the
bank: **streaming is causal filtering** (a unidirectional encoder is a causal filter, a
bidirectional one is not, and `filtfilt` cannot run on a live signal for the same reason
bidirectional models cannot stream); **RTF < 1** is the control-loop deadline; **statistics
pooling is an overall-level reading**; **beamforming is a phased array** with the √M array gain;
**MVDR is a Lagrange-multiplier problem** with a closed form; **the LMS filter is live gradient
descent**. The two things that genuinely do not transfer: perceptual weighting (mel, log, phase
discard — all facts about a listener, and a bearing has no cochlea) and the alignment problem.

**10 Multimodal.** ~~Fusion ↔ sensor fusion; Kalman filtering of accelerometer plus gyro.~~
**Written.** The bank's one line was correct and was not the interesting part. Early/middle/late
fusion **are** raw-data/feature/decision-level fusion — the same taxonomy arrived at twice — but
the chapter's spine turned out to be what is *missing*: no observation model, no noise
covariance, no innovation to monitor, no observability test. Structures transfer completely;
guarantees do not. Added beyond the bank: **residual quantisation is a SAR converter**
(the headline, below); **codebook quantisation is Ch06 k-means and Ch09 amplitude quantisation
at once**; **codebook collapse is Ch07 MoE load balancing**; **classifier-free guidance is
successive over-relaxation** with a negative weight on the unconditional branch; **FID is a
centroid and a second moment**; **Flamingo's zero-initialised gate is a bumpless transfer**;
**token compression is static condensation**; **staged training is a commissioning sequence**.

**11 Autonomous systems.** Mostly mechanical already: DH matrices (Ch02 §06), the
manipulator Jacobian (Ch02 §03, Ch03 §03), control loops, state estimation. **Chapter 10 has
already done half the setup**: its §01 break box enumerates exactly what a Kalman filter has
that learned fusion lacks — `H`, `R`, the innovation, observability — so Chapter 11 can pick
those up as things the reader has now been told twice they will meet properly here. Treat that
as a promise already made to the reader.

**12 Graph neural networks.** ~~A finite element mesh **is** a graph, and the graph
Laplacian is assembled exactly as a stiffness matrix is. Laplacian eigenvectors ↔ mode
shapes. Message passing ↔ Gauss–Seidel and other iterative solvers. Very strong.~~
**Written.** The bank was right about the strength and **wrong about the solver**, which turned
out to be the chapter's best result — see below. The Laplacian identity held completely: `L =
D - A` *is* the direct-stiffness assembly with element matrix `[[1,-1],[-1,1]]`, to zero error,
for one scalar DOF per node. From it: `x'Lx = 2U` (strain energy), path-graph eigenvectors ==
free–free bar modes `cos(πk(j+½)/n)`, and the ring's graph Fourier transform **is** the DFT
(same matrix — and the same objects a bladed disc's nodal-diameter modes). Added beyond the
bank: **message passing must be Jacobi, not Gauss–Seidel** (the headline — Gauss–Seidel is
order-dependent, so permutation equivariance forbids it; the symmetry requirement picks the
solver); **equivariance IS material frame indifference** and an invariant architecture is an
isotropic constitutive law written in the invariants; **the ℓ=0/1/2 irrep hierarchy is the
velocity-gradient split** into dilatation ⊕ vorticity ⊕ deviatoric strain rate, `9 = 1+3+5`,
and **Mohr's circle's factor of 2 IS the ℓ=2 label**; **cyclomatic number == degree of static
indeterminacy**; **normalisation is mass-normalisation** and lumped mass really is proportional
to degree on a uniform mesh; **DiffPool is the Galerkin coarse-grid operator** `R'KR`, inheriting
multigrid's rigid-body-mode rule; **self-loops are a stability condition**; **a CNN is to a GNN
what a finite-difference stencil is to FE assembly**. The bridge ends at **attention**: GAT is
not symmetric, so Maxwell–Betti fails and strain energy, real modes and the graph Fourier
transform all go at once — Chapter 07's lesson in the same shape.

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
- **Two upstream SVGs were malformed XML and rendered as nothing.** Fixed on this branch at
  Chapter 09: `ctc_alignment.svg` used `&mdash;` (an HTML entity XML does not define) and
  `feature_store.svg` had a bare `<` in text content. All 263 SVGs in `images/` now parse.
  Same upstream-PR caveat as the mkdocs fix above.
- Chapter 10 is published but **not watched** — that session hit the 10-artifact
  wake-subscription cap. Cosmetic; it only means no notification if someone republishes it.
  Chapter 12's publish reported its subscription as "arming in the background, not registered
  yet", which is not a confirmed watch either. Do not state that a chapter is being watched
  unless the publish result says so.
- **Nine of the twelve published sheets have display equations that overflow their `.mathbox`
  at 390px, and Chapter 10 has one clipped `.ro-k` label.** Measured at Chapter 12; details in
  the series-sweep note under the browser-probe section. Pre-existing, not regressions, and
  not fixed — they need eight artifacts re-probed and republished.


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
  against a server that is starting fine. At Chapter 12 `npx --offline http-server` took long
  enough that the readiness loop gave up and a `python3 -m http.server` fallback was started on
  the same port, which then failed with `Address already in use` — because http-server had in
  fact come up. **If the fallback reports the port is taken, the first server is running; just
  curl it.** Also note `npm pack mathjax@3.2.2` works but is slow; budget for it.
- **Anatomy of the probe.** Two scripts are worth keeping separate: one that walks the five
  widths asserting the structural invariants, and one that scrolls to each lab and screenshots
  it. The first proves nothing is broken; only the second shows whether anything is
  *legible*. At Chapter 12 the first passed clean while the second revealed a colour collision,
  two label overlaps, an invisible arc, and a wrong readout.

What the probe must check, at widths 1180 / 900 / 700 / 480 / 390:

1. `mjx-container` count > 100 — otherwise MathJax did not run and nothing below counts.
2. No display equation wider than its `.mathbox`.
3. `documentElement.scrollWidth == clientWidth` — no horizontal page scroll.
4. No `.ro-k` / `.lg` label with `scrollWidth > clientWidth` (clipped label).
5. Every slider swept end to end, every preset clicked, every predict button clicked,
   with `pageerror` collected throughout.
6. Every readout id still holding its placeholder `—` after that sweep — a readout that never
   got written is a lab that never ran, and it looks identical to one that ran fine.
7. Any pair of readouts that should agree (a measured value and its closed form) actually
   agreeing. Chapter 12 shipped a draft where they differed by 12% and nothing else noticed.

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
- **A design-system fix does not travel backwards on its own.** Each chapter inlines its own
  copy of the token block, so a rule repaired while writing chapter N stays broken in
  chapters 1..N-1 until someone backports it. This bit `.ro-k`: the `white-space:nowrap`
  that clipped long readout keys was fixed at Chapter 06 and carried forward to 07 and 08,
  while Chapters 01-05 kept the broken rule and Chapters 03, 04 and 05 shipped with visibly
  clipped labels (`sigma - radius of gyra...`, `H(p,q) - cross-ent...`). Fixed everywhere now.
  **After repairing anything in the shared head block, grep the earlier chapters for the old
  rule and backport it**, then re-probe them. A one-line diff per chapter; the alternative is
  a series that drifts apart chapter by chapter.
- **An SVG that is not well-formed XML renders as nothing at all.** SVG is XML, which
  predefines only five entities. `&mdash;` inside a `<text>` element is an *undefined entity*,
  the parse fails, and the browser draws a blank box — not a partial figure, nothing. A bare
  `<` in text content (`(<5ms)`) fails the same way. Two of the compendium's 263 SVGs had this
  (`ctc_alignment.svg`, `feature_store.svg`); both are fixed on this branch with numeric
  references (`&#8212;`) and `&lt;`. **Before shipping a chapter, assert every SVG it uses
  parses:** `python3 -c "import xml.dom.minidom as m; m.parse('f.svg')"` over the set. The probe
  catches it too — `brokenImg` counts images where `naturalWidth === 0` — but only if you look.
- **A lab's parameter range can produce a degenerate result that looks like success.** The
  Chapter 10 residual-quantisation lab ran k-means with a 512-entry codebook over 260 sample
  points, so every point got its own centroid and the reconstruction error read exactly
  `0.0000`. That is not a great result, it is the absence of a result. **Sweep each slider to
  both ends and ask whether the extreme values are still meaningful**, and where a lab fits
  clusters, keep the codebook well below the sample count.
- **Any softmax in a lab needs the log-sum-exp trick.** `exp(0.9/0.01)` overflows a float64
  and yields `NaN`. Subtract the row maximum before exponentiating — it leaves the result
  unchanged and is exactly why every real implementation does it. This bit both the Chapter 10
  verification script and the lab that mirrors it.
- **Splitting a chapter silently breaks cross-references.** Every `&sect;NN` written in an
  early part points at a number that later parts shift. Chapter 09 renumbered the mel-scale
  correction three times (§10 → §16 → §22); Chapter 10 merged two parts and moved seven
  sections. **After any renumber, verify that every `&sect;NN` resolves to a section that
  exists and says what the sentence claims** — parse the `secnum` spans into a map and print
  each reference against its resolved title. Both chapters came out clean, but only because
  the check was run; nothing else would have caught it.
- **The provenance appendix is end-matter and gets dropped when a chapter is split.** Rule 4
  says every chapter ends with Appendix A. Chapter 09 part 1 shipped without it because the
  end-of-chapter material was deferred to a later part. If you split, either write the
  appendix in part 1 covering part 1, or put an explicit note in the endnote saying it lands
  with the final part — and then actually land it.
- **A `<<HEREDOC` without quotes destroys LaTeX. This cost a whole repair pass at Chapter 12.**
  The build procedure already said "quoted delimiter"; here is *why*, so the next run does not
  rationalise its way past it. In an **unquoted** heredoc the shell expands `$$` to its own
  **process ID** — so all 23 `<div class="mathbox">$$…$$</div>` blocks came out as
  `<div class="mathbox">940…940</div>` — and it collapses `\\` to `\`, so every
  `\begin{bmatrix}a&b\\c&d\end{bmatrix}` row separator vanished. Neither breaks the HTML, so
  tag-balance checks pass and the damage is silent until MathJax renders nonsense. If you need
  a variable (the source-URL prefix), use a **quoted** delimiter and substitute afterwards with
  Python, or write a placeholder and `sed` it. Detection, if it has already happened:
  `grep -c '<div class="mathbox col">\$\$'` must equal `grep -c '<div class="mathbox col">'`.
- **The design system has only three series colours, and two pairs of them are duplicates.**
  `--s1` is `#BF5B00` and `--brk` is `#A8500F` — both orange, indistinguishable on screen.
  `--s3` and `--bridge` are the *same hex*. There is no `--s4`, so any `var(--s4)` silently
  renders as an invalid paint. Chapter 12 shipped a lab colouring positive and negative mode
  amplitudes `--s1` against `--brk`, which made a sign change invisible — in a lab whose entire
  subject is where the sign changes. Chapter 12 now defines validated `--c1…--c5` in its own
  style block (light and dark, both passing the `dataviz` validator on lightness band, chroma
  floor, adjacent-pair CVD separation, normal-vision floor and contrast). **Copy that token
  block forward; never pair `--s1` with `--brk`, and never use more than three of the stock
  series tokens at once.**
- **A readout can be numerically wrong in a way no structural check sees.** Chapter 12's
  over-smoothing lab printed a measured decay ratio of 0.914 beside a predicted `|μ₂|` of 0.811.
  Both numbers rendered, nothing overflowed, no error was thrown — but the lab measured spread
  about the plain *mean* while the verified claim projects out the rigid-body direction
  `√deg/‖√deg‖`, which is not the same subtraction on a graph with unequal degrees. **Put the
  measured value and the predicted value next to each other in the readout row**, and then a
  screenshot shows the disagreement immediately. That is worth doing deliberately in every lab
  that has a closed form to compare against.
- **Draw disconnected components separately.** Laying every node of a 6-node graph on one
  circle made "two triangles" look like a tangle of crossed chords, in the one lab whose point
  is that it is *two loose pieces*. Detect components and give each its own circle.
- **Screenshots catch what the probe cannot.** The probe's numeric checks passed on all of the
  above. Colour collisions, label overlaps, arcs drawn in the same hue as the circle under
  them, and a legend sitting on top of an axis title are all invisible to `scrollWidth`
  comparisons. **Screenshot each lab individually, not just the page**, and select with
  `#section .lab` before `#section figure` — a section that opens with a compendium figure will
  otherwise hand you the figure.
- **Probe the whole series occasionally, not just the chapter you are writing.** The clipped
  labels above were invisible for three chapters because nobody re-ran the earlier pages. A
  sweep over every `ch*/index.html` at 1180 and 390 catches this in about a minute:
  horizontal scroll, `scrollWidth > clientWidth` on `.ro-k`/`.ro-v`/`.lab-t`, display
  equations wider than their `.mathbox`, broken images, and page errors.
  **Result of that sweep at Chapter 12, recorded so nobody has to re-run it to know:** no page
  ever scrolls horizontally, no broken images, no JS errors, in any of the twelve. But **nine
  of twelve have display equations wider than their `.mathbox` at 390px** — ch01 (5), ch02 (3),
  ch03 (1), ch04 (2), ch05 (2), ch07 (4), ch08 (1), ch11 (1). Those scroll inside the box
  (`.mathbox{overflow-x:auto}`) so they are readable with a sideways swipe rather than broken,
  which is why they have survived; ch06, ch09 and ch12 are the only clean ones. **ch10 has one
  genuinely clipped label at 390px** ("extrapolating"), which is the `.ro-k` bug class the
  notes above say was fixed everywhere — it was not, or it regressed. Both are pre-existing and
  outside Chapter 12's scope; neither was touched. Fixing them means re-probing and
  republishing eight artifacts, so it is a deliberate piece of work for a session that chooses
  it, not something to fold into the next chapter.

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

## The Chapter 08 lesson worth generalising

Three things came out of Chapter 08 that apply to everything after it.

**The bridge bank undersold the chapter badly.** It listed three correspondences for computer
vision; the chapter shipped with eight, and the strongest one — the Harris structure tensor as
a plane stress tensor — was not in the bank at all. The bank was written from the chapter
titles. The real bridges were found by reading the source and noticing that a symmetric 2×2
matrix classified by `det` and `trace` is a thing this reader has done by hand a hundred times.
**Read the source before trusting the bank.** Budget for the bank being a floor.

**The same object appearing twice is the strongest possible signal.** The structure tensor
turns up in file 01 as corner detection and again in file 05 as the matrix Lucas–Kanade
inverts. That repetition is what made it the headline rather than one bridge among several,
and it also supplied the chapter's best single insight: flow is recoverable exactly where the
tensor is well-conditioned, which is why trackers track corners. When a chapter reuses one
piece of mathematics under two names, that is the spine.

**The discrete-versus-continuum trap has now bitten twice, in the same shape.** Chapter 07:
the SSM kernel is the discrete impulse response, not the sampled continuous one, because a
unit sample is a held pulse and not a delta. Chapter 08: a blurred step matches the error
function, but only once σ exceeds the pixel spacing, and an asymmetric discrete step
`(x>=0)` matches `erf(x+½)` rather than `erf(x)` because its jump sits half a sample off
centre. Both were invisible in prose and obvious the moment two columns of numbers were put
side by side. **When a page claims a discrete object equals a continuous one, compute both and
print the ratio.** State which identity is exact and which is a limit; the exact one is always
the more useful claim, and it is usually not the one you first reached for.

One process note: the traps recorded after Chapter 07 — the `.paper` wrapper, block-level
`ro-k`/`ro-v` divs, the corrected `sed -n '1,385p'` head boundary, the 17-selector uppercase
list — all held. Chapter 08 needed no layout rework at the probe stage beyond widening one
diagram for legibility. The notes are worth keeping current; they paid for themselves in one
chapter.

## The Chapter 09 lesson worth generalising

**The source can be wrong, and its own formula will usually prove it.** File 01 states that the
mel scale "is the reason that musical semitones are equally spaced on a log-frequency axis".
Computing octave widths from the formula the source itself gives, `m = 2595 log₁₀(1 + f/700)`,
returns 241.6, 367.8, 499.0 and 608.2 mel — rising steadily, with the third octave 1.36× the
second. If the claim were true these would be equal. Two different perceptual facts had been
merged into one sentence: musical intervals are equal on a *log-frequency* axis, which is about
pitch **ratios**, while the mel scale measures perceived pitch **distance** and is deliberately
near-linear below about 700 Hz. **Where a claim is arithmetic, do the arithmetic.** A sentence
that joins two true facts can still be false, and prose review will never catch it.

**The verification test is as likely to be wrong as the page.** The Chapter 09 autocorrelation
check failed three times before it was right: too few periods, then missing unbiased
normalisation, then taking the global maximum instead of the first peak. That third failure
landed on **twice the period** — which is precisely the octave error the page describes two
sections earlier. The test now asserts both the naive wrong answer and the correct one, because
reproducing the failure mode is worth more than hiding it. When a check fails, the first
question is whether the test or the claim is wrong, and it is roughly even odds.

**Upstream bugs show up as blank figures, not as errors.** See the SVG trap above. The only
reason it was caught is that the probe counts broken images.

## The Chapter 10 lesson worth generalising

**The best result in the chapter came from a test that failed.** The source calls residual
quantisation "analogous to successive approximation" and moves on. The check written to confirm
that showed the error was *not* monotone: with a fixed-scale codebook it stalls at 82% of the
original and then rises, because once the residual is smaller than the codebook vectors the
nearest entry is farther away than zero. Fit each stage to its own residual and it falls
geometrically — per-stage ratio 0.793 to 0.799, constant to 1%, so `log(error)` is linear in
stage count. That is exactly why a SAR converter halves its test voltage every bit rather than
using a fixed step.

The generalisable form: **when a verification test fails on an analogy the source asserts, ask
first whether the analogy carries an unstated condition** — not whether the code is buggy. A
bridge that predicts a failure mode is load-bearing; one that only renames things is
decoration. That is Chapter 08's own test, and this is the clearest instance of it so far.

**Two test errors in one check, and the second would have shipped a false claim.** Measuring
FID's finite-sample bias, the first suspicion was numerical (`sqrtm` of a non-symmetric
product). Both the naive and the stable form agreed exactly, which ruled that out. The actual
bug was mine: samples were generated with covariance `AAᵀ` while being compared against
`AAᵀ + I`, so the apparent bias floor of 0.9 was simply the true FID between two genuinely
different distributions. Had the first hypothesis been accepted, the page would have asserted
that FID never converges — confidently, with a table. **When a measurement plateaus where
theory says it should converge, check what you actually sampled from before blaming the
metric.**

**Cross-references are the tax on splitting.** See the trap above.

## The Chapter 11 lesson worth generalising

**Three of the chapter's best results came from a test that failed, and the test was mine, not
the page's.** This kept happening and is now the most reliable way to find something worth
writing:

1. I asserted the steady-state Kalman filter's damping ratio falls as `q/r` rises. The test
   "passed" — on floating-point noise in the last digits. It does not fall; solving the
   Riccati equation in closed form gives `ζ = 1/√2` **exactly**, for every `q`, `ρ`, `Δt` and
   both standard `Q` discretisations. Only `ωₙ = (q/ρ)^(1/4)` moves. The filter *derives* the
   0.707 a mechanical engineer is taught to pick off a design chart. That became the headline
   result of the whole sheet.
2. I asserted behavioural cloning compounds because injected noise accumulates. The test
   failed, correctly: a contracting closed loop absorbs noise **forever** — deviation at
   T=160 came out smaller than at T=10. What compounds is *losing the contraction*, and
   leaving the training distribution is how it is lost. Replaced with a Monte Carlo of the
   actual argument (cost quadratic in horizon; doubling T costs 3.4×, not 2×).
3. I asserted damped least squares "flattens at ‖Δx‖/λ". The browser probe showed otherwise:
   the demand **peaks** exactly where `σ_min = λ` (measured 0.118, predicted 0.123) and then
   **falls towards zero**. It does not cap the demand; it withdraws from the direction being
   lost. Better result than the one I had written.

**Lesson: write the claim, then try to break it before you believe it.** Also beware a test
that passes *vacuously* — `max(0.0, -λ_min)` can never go negative, so a positive-definiteness
assertion built that way always passes and checks nothing.

**The browser probe found a bug invisible in source, again.** LAB 6 declared `CX0` as an array
at module scope, colliding with LAB 2's `CX0` number in the same IIFE — so LAB 2 threw on every
interaction once LAB 6 loaded. Fix: after merging a new lab into an existing page, scan the
shared script for module-scope `var` collisions (2-space indentation = module scope in these
files) before publishing. A one-line script does it.

**Corrections found in the source this chapter (five).** All stated on the page with the
computation:
- Swarm consensus converges to the **degree-weighted** mean, not "the global average" — 8.1%
  of the range off on the worked graph. And the source's own lab builds neighbourhoods by
  k-nearest-neighbours, which is not symmetric, so even that formula fails there. *Best framing
  found:* the claim is exactly right for a **regular** graph (the lab's Ring preset shows error
  0.0000) and was generalised to graphs that are not. Say that — it is more useful than "wrong".
- "No single point of failure" is true for crashes, false for value consensus: freeze one node
  and the whole swarm adopts its value exactly.
- A quadruped is not statically stable on three legs — lifting a corner foot leaves a margin of
  **exactly zero at every aspect ratio**, because a rectangle's centre lies on its diagonal.
- The VLA action vocabulary is 256 shared bins, not 7×256=1792 — the source's *own* RT-2
  example emits seven tokens all below 256.
- Earth–Mars one-way light time derived from orbital radii is 3.03–22.31 min, not 4–24.

**What made this chapter unusual: almost everything transferred.** Twelve of seventeen bridge
rows read *identical*, not *analogous* — and that is a literal claim, verified. `det J = l₁l₂
sin q₂`, and at `q₂ = 0` the lost singular direction **is** the radial direction to ten
decimals, which makes "a singularity is a linkage dead centre" an identity. So the sheet moved
fast through the identities and spent its length on the four places intuition misleads. When a
chapter is this close to the reader's training, the risk inverts: the few non-transferring
items become easy to miss, and they are the dangerous ones.

**A result worth reusing:** `ζωₙ = (b + K_d)/2m` — `K_p` cancels completely. So `K_d` alone
sets the decay envelope and `K_p` alone sets ringing; raising `K_p` cannot speed up settling.
Verified across eight gain pairs. This is the kind of one-line consequence that is worth more
than a page of qualitative tuning advice, and it was found by multiplying two definitions
together and noticing a cancellation.

## The Chapter 12 lesson worth generalising

**The bank was not merely incomplete this time — it was wrong, and the error was the
chapter.** Every previous entry undersold its chapter; this one contained a false claim.
It said "message passing ↔ Gauss–Seidel and other iterative solvers". Message passing cannot
be Gauss–Seidel: Gauss–Seidel uses values already updated within the sweep, so its answer
depends on the node ordering, and permutation equivariance — the defining property of a graph
network, set up in the chapter's own file 01 — forbids exactly that. Measured, the mismatch
under relabelling is `1.1e-16` for Jacobi and `1.9e-1` for Gauss–Seidel. **The symmetry
requirement picks the solver, and the slower solver is the only admissible one.** That became
the best thing in the sheet, and it was found by asking "which relaxation scheme is this,
exactly?" rather than accepting the family name. **When a bank entry names a category
("an iterative solver", "an optimiser", "a transform"), ask which member of the category and
why that one — the constraint that selects it is usually the content.**

**Five corrections in one chapter, and one of them is visible by running the source's own
code.** File 03 claims mean cannot distinguish `{1,1}` from `{2,2}` — the means are 1 and 2.
Worse, its coding task builds `{1,1,1,1}` and `{2,2}`, prints both aggregations, and captions
the output "Sum distinguishes these multisets; mean does not!" — while the printed means are
1.0 and 2.0 (separated) and the printed sums are 4.0 and 4.0 (collided). It demonstrates the
exact opposite of its caption. The conclusion it is attached to is nonetheless correct, which
is the interesting part: **a sound conclusion can carry a broken argument for years, because
readers check the conclusion against what they already believe and never run the example.**
The real argument needs no example at all — mean and max are killed by *duplication*, for any
feature map, and that is a proof. **Run the source's code. It takes a minute and it is the
highest-yield check available.**

**The same degree-normalisation trap has now appeared three times.** Ch11 §19: swarm consensus
reaches the degree-weighted mean, not the global average. Ch12 §13: over-smoothing converges to
`√(1+dᵢ)`, not to a uniform value. Both are the same mistake — assuming a degree-normalised
iteration has a flat fixed point, which is true only on a regular graph, which is exactly what
every textbook figure draws. **Whenever an iteration is normalised by degree, compute its fixed
point; never assume it is flat.** What makes the Ch12 instance worth more than a correction is
that the mechanical reading *predicts* it: `L̂` is the mass-normalised stiffness matrix, so it
acts on `q = M^(1/2)x`, and the rigid-body mode `x = 1` written in those coordinates simply *is*
`√degree`. A bridge that predicts the correction has earned more trust than one that survives it.

**My own claim broke, and the break became the worked example.** §13 originally said `μ₂` is why
GNNs stop at 2–4 layers. Computing the depth budget for an actual sparse molecular graph gives
`|μ₂| = 0.967`, hence *68* layers to lose 90% of the distinguishing signal, against a diameter
of 8 — the window is wide open, and the rule of thumb is nowhere near either bound. Isolating
the pieces showed the learned `W` barely matters (0.736 → 0.722 at four layers) while the
**ReLU** roughly doubles the collapse rate, and even with both, four layers leave 38% of the
spread intact. So the standard explanation does not hold at the scale it is usually asserted;
the window closes only for *dense* graphs (it is shut for `G(24, p=0.8)` and `K₂₄`). §13 was
corrected, §24 is the calculation, and the superseded claim is named in the provenance
appendix rather than quietly deleted. **Distinguish "this explanation fails" from "there is no
problem" — the calculation rules out one mechanism at one scale and does not establish what
the operating mechanism is.** Saying so is the honest stopping point.

**When almost everything transfers, put the effort into bounding it.** Sixteen of twenty bridge
rows are identities. The risk of a hit rate that high is that the exceptions get less attention
exactly when they need more, so the sheet states the scope condition on the front page — *one
scalar degree of freedom per node* — and repeats it wherever it bites. The single most valuable
paragraph is the amber box in §17: attention breaks the symmetry, and strain energy, real mode
shapes, the graph Fourier transform and every variational argument go at once. That is
Chapter 07's lesson for the third time: **the bridge is exact right up to the point where the
architecture becomes good, and then it ends.** Say it in the same breath as the bridge.

## On splitting a chapter into parts

Chapter 09 (3,250 source lines) was built in four runs and Chapter 10 (1,900) in two, against
single-run chapters 06, 07 and 08. The user asked directly whether splitting hurt quality. It
did not — measured:

| | Ch06 (1 run) | Ch07 (1 run) | Ch08 (1 run) | Ch09 (4 runs) | Ch10 (2 runs) |
|---|---|---|---|---|---|
| words | 8,025 | 8,935 | 8,463 | **13,613** | 10,361 |
| sections | 18 | 17 | 18 | **26** | 19 |
| labs | 3 | 3 | 3 | **4** | 3 |
| bridge + break boxes | 19 | 10 | 7 | **41** | 35 |
| questions | 16 | 24 | 24 | 24 | 24 |

Words per source line came out at 4.19 for Ch09 and 5.45 for Ch10, against 4.22 and 4.63 for
the single-run chapters, so splitting did not pad — Ch10 is the shortest split chapter and
still the second-densest in bridge boxes. What splitting bought was depth in exactly the
dimension that carries this project's value, plus one probe cycle per part — which caught
three lab bugs in Ch09 §06 and one in Ch10 §06 that a single pass would likely have missed.

The costs are real and both are avoidable: the dropped provenance appendix and the
cross-reference renumbering, both covered in the traps above. **Rule of thumb: split above
~2,000 source lines, and write the part boundaries at a natural seam** — Ch09 split at
signal → features → recognition → synthesis, Ch10 at representation → generation. Commit and
push every part; never leave a part uncommitted at the end of a run.

## What is actually left

Chapters 13–20, of which 19 and 20 are outline stubs (20 and 21 lines respectively — five
near-empty files each) and may not be worth writing as chapters at all. That makes the real
remaining work chapters 13–18, six sheets. The bridge bank above has a pre-worked entry for
each; Chapters 08, 09, 10, 11 **and 12** all found their headline bridge *outside* the bank, so
treat it as a floor, not a plan. Chapter 12 went further and found the bank's single line for
it was *wrong* (Gauss–Seidel), which became the chapter's best result.

Source sizes, measured, so the split decision is already made:

| # | Chapter | Source lines | Expect |
|---|---------|--------------|--------|
| ~~12~~ | ~~Graph neural networks~~ | ~~1,209~~ | **written, one run** |
| 13 | Computing and OS | 1,243 | one run |
| 14 | Data structures and algorithms | 2,695 | **split** — above the ~2,000 line |
| 15 | Production software engineering | 1,474 | one run |
| 16 | SIMD and GPU programming | 3,722 | **split**, probably three parts |
| 17 | AI inference | 1,276 | one run |
| 18 | ML systems design | 1,278 | one run |
| 19 | Applied AI | 20 | stub — five near-empty files |
| 20 | Bleeding edge AI | 21 | stub — five near-empty files |

All of 13–20 are now measured, so every split decision is already made. Chapter 16 is the
largest source in the series after Chapter 09 and will want three parts; Chapter 14 wants two.
Everything else is a single run.

# Chapter 16 part 2 of 3 — SIMD and GPU Programming: the GPU (source files 04–05)

## 1. State row

| 16 | SIMD and GPU Programming *(pt 2: the GPU)* | `ch16-simd-gpu-part2/` | https://claude.ai/artifact/8byqc279ofge6NJ7k7h4NU |

## 2. README entry

For the Verification paragraph: "and — for Chapter 16 part 2, with no GPU in the container — the
GPU roofline from lanes x clock x 2, a cycle-by-cycle simulation of one SM against Little's bound
(and with latency jitter), divergence over 20,000 simulated warps against `1-(1-q)^32`, an
output-stationary systolic array checked wave by wave against `a_k b_k^T`, float32 sums of one
array in 200 arrival orders, the online-softmax identity, and the source's Pallas kernel and JAX
chain actually run on the CPU (JAX 0.11.2, Pallas interpreter mode)."

Sheet entry: the GPU read through part 1's two shop-floor laws, paying its seven debts by
computation. Headline: **warps are kanban cards and the register file is the card rack** — the
source's 400 cycles and 3.3 TB/s ask for 39.5 warp-loads in flight per SM; 65,536 registers hold
40 warps only at <= 51 registers per thread, and at 128 registers the law caps an SM at 40% of its
bandwidth (simulated: 40.1%) unless each warp keeps several loads in flight. The roofline moved:
ridge 20.3 FLOP/B (part 1's core: 2.56), so the source's 16x16 tile (4 FLOP/B) is delivery-bound
at 13.2 TFLOP/s, and the 1,024-thread block caps any one-output-per-thread tile at 32 (8 FLOP/B).
Exact correspondences: PCIe as a receiving dock (a third station in the same min; streams as
Sheet 13's pipeline, at most 2.02x for the ReLU task); divergence as **Sheet 15's c = 0 sampling
plan** (a 1% branch is run by 27.5% of warps) and a loop as the deepest hole; coalescing as
pallets; shared memory as a planned bin (exact at kit size, where LRU needed slack) and double
buffering as the two-bin system; the systolic array as a line filling and draining,
`K/(K+2n-2)` (3.1% at batch 8); fusion priced at the bottleneck, ceiling `1 + ksI*/d` = 1.29 on
a real bf16 layer against a 5:1 traffic cut; online softmax as a datum shift; summation order as a
tolerance stack (14 different totals in 200 orders). Fourteen corrections to the source, six labs.

## 3. Bridge bank update

**16 SIMD and GPU.** Part 1 entry stands. **Part 2 written.** Memory bandwidth ↔ material flow held
again, now as a three-station line (dock, stores, machines), and the ridge moved past every tile
the source teaches. Little's law held exactly in a deterministic SM simulation and broke on who
sets the card count (the register file and the compiler, in whole blocks). The drilling head held
for branch counts and broke at per-lane addressing, warp switching and shuffles. Outside the bank:
the c = 0 sampling plan for divergence, the two-bin system for double buffering, and fusion's
ceiling `1 + ksI*/d`.

## 4. The lesson worth generalising

**When there is no hardware, simulate the law, not the machine, and say which inputs are borrowed.**
No GPU existed here, yet every headline is computed: a ~40-line SM model reproduces Little's bound
to the third figure, the systolic array is checked wave by wave, and float32 order effects are
exact in the browser (`Math.fround(a+b)` is IEEE float32 addition, since double holds 2p+2 bits).
Separating source figures from four reported constants (SM count, lanes, clock, warp limit) let
every derived number stay honest. Second: **price a claimed saving at the bottleneck.** "Fusion
removes 4 of 5 passes" is true and buys 1.29x on a compute-bound layer; the formula
`1 + ksI*/d` says when it buys more. Same move as part 1's moving ridge.

## 5. New traps, and verifier count

- **Replay the lab's RNG in Python** (`mul32` mirrors the page's `rng()` bit for bit) so the
  verifier asserts the exact rest values of stochastic labs, not tolerances around them.
- **Optional dependencies in the verifier.** The two checks that run the source's JAX/Pallas code
  are skipped with a NOTE when JAX is absent (system python here has none; it was run from a venv).
- **Round from the formula, not memory**: `1-0.999^32` is 3.15%, which prints as 3.2%; the draft said 3.1%.
- **Per-warp completion-interval throughput** avoids window-quantisation bias in short simulations
  (a 10,000-cycle window over a 404-cycle loop read 1.2% above the bound until changed).
- Verifier after this block: **1,157 PASS, 0 `*** FAIL ***`, rc 0** with system python (JAX checks
  skipped); **1,159 PASS** with JAX installed. `_ch16p2` prints 99–100 `[Ch16p2 sNN]` lines, 85–87
  PASS. Probe: five widths clean, 28 rest-value expectations met. `audit_classes.py` rc 0.

## 6. Forward promises (the page's §19, "What part 3 owes you")

Part 3 (files 06–07) owes, each as a calculation:
1. **RVV as predicates again** (part 1 §16, pointing to its §09; and Triton's mask, §11 here): run file
   06's `vsetvl` strip-mining loop; show `ceil(n/VL)` strokes with a partly filled last stroke (part 1
   §05's count) across the source's VLEN range 128–65,536 bits; say what `vsetvl` adds over a mask.
2. **Vulkan subgroups as lanes** (part 1 §16): map invocation/workgroup/subgroup onto thread/block/warp
   (§02 here); recompute §06's divergence count and `1-(1-q)^s` at subgroup sizes 32 and 64.
3. **File 07's tree reduction, counted**: barriers, idle lanes and idle warps per step (as §06 counts
   divergence), and whether its total repeats bit for bit in §14's sense (fixed within a workgroup;
   the second dispatch's order).
4. **File 07's GLSL tiled matmul, priced**: its `TILE_SIZE` by `b/4` against a ridge from the source's
   device figures where given; which station binds.
5. **The edge in FLOPs per joule**: file 06's server GPU / smartphone / microcontroller table, compute
   divided by power from the source's own numbers, and what it implies about where computing is cheapest.
6. **A chapter-wide concept map** joining all three parts, and part 3's own Appendix A for files 06–07
   plus the chapter's end matter.

# Chapter 16 part 1 of 3 — SIMD and GPU Programming: the CPU (source files 00–03)

## 1. State row

| 16 | SIMD and GPU Programming *(pt 1: the CPU)* | `ch16-simd-gpu/` | https://claude.ai/artifact/PkFJKQedyFn3SNLNqWGKCy |

## 2. README entry

For the Verification paragraph: "and — for Chapter 16 part 1 — the source's own C++ compiled and
run (its bandwidth benchmark printing ~1,000 GB/s because GCC ran 1 of 100 trials, its alignment
benchmark deleted as dead code), `min(K/4, 2)` against measured FMA chains, Little's law against a
K-chain pointer chase, exact LRU traffic of naive and tiled matmul by stack distance against direct
simulation, `16x16x32 = 8,192` MACs for AMX run with and without its setup, and the cube-law
equal-power Amdahl crossover `s* = (n^-1/3 - 1/n)/(1 - 1/n)`."

Sheet entry: the CPU read through **two shop-floor laws**. Headline: **the closed-line bound
`TH(w) <= min(w/T0, r_b)`**, derived in two lines from Little's law, **runs the core at two scales**:
FMA chains with K independent accumulators measured 0.26, 0.50, 1.02, 1.48, 1.86, 2.04 vector FMAs
per cycle (knee at `4 cycles x 2 units = 8`), and a K-chain pointer chase rose as `K/210 ns` and
flattened near 15 lines in flight. The roofline is the `r_b` half: **a two-station serial line**
(delivery BW x I, machining Peak), the ridge is the balanced product mix, and adding cores moves it
(Goldratt's fifth step, 2.56 -> 20.48 FLOP/B). Also exact: **SIMD is a multi-spindle drilling head**
(strokes, masked spindles, both-branches divergence count) while **a multi-spindle automatic is a
pipeline**; **cache lines as pallets** with the split fraction `W/64`; **tiling is kitting** with
intensity `b/4` = volume over surface (cube optimal by Lagrange); the affinity laws' exponents
for the cube law; the 2-bit predictor as hysteresis; fusion as done-in-one machining (`1+2k` to 1).
Fifteen corrections to the source, including the 10 ms matmul (needs 13.7 TFLOP/s), the peak that
counts an FMA as one FLOP, the NEON "4 accumulators" explanation (one register is one chain:
measured 0.30), "restrict prevents vectorisation" (GCC versions the loop), 31 not 32 AArch64
registers, SMMLA's 32 MACs, the uncompilable SME kernel, AMX's "hundreds" and its SIGILL as printed.
Six labs.

## 3. Bridge bank update

**16 SIMD and GPU.** ~~Where Ch02's matmul lands. SIMD ↔ multi-spindle machining. Memory bandwidth ↔
material-flow bottleneck.~~ **Part 1 written.** Bandwidth ↔ material flow held and was undersold:
the roofline **is** the two-station line's throughput formula, the ridge is the balance point, and
the ridge moves with cores. The SIMD entry named a family: only the **drilling head** is SIMD; the
**multi-spindle automatic is a pipeline**. Ch02's matmul lands as **kitting** with intensity
`b/4` (exact LRU counts). The headline was outside the bank: **Little's law / the CONWIP best case**
sets accumulator counts and per-core memory bandwidth, measured. Parts 2–3 still to write.

## 4. The lesson worth generalising

**Measurements are claims about a machine, and the machine changed under me twice.** The container
switched between a Cascade Lake and a Sapphire Rapids-class CPU across cut-offs. Constants moved
(float-add latency 4 -> 2 cycles; a 260 MiB L3 held arrays that were DRAM-bound on the other), and
three verifier checks that encoded one machine's constants failed on the other. The fix was to
assert the **law, not the constant**: test `unrolled ~ AVX2` rather than "all three equal",
derive floors from a measured add latency, take the best of several runs, and widen bounds that
rest on a clock calibration under turbo. Report both machines on the page. The structural result
(the knee at latency x units, the plateau of misses in flight) held on both.

## 5. New traps

- **The optimiser is a participant in every benchmark.** Two of the source's six x86 programs
  measured nothing (a hoisted repeat, a dead loop). Use every result, or pass it through an empty
  `asm volatile`; compute the floor first; a result that beats the floor is wrong.
- **Shared-VM timing noise is 20–30% per run** and other parallel runs load the same CPUs: any
  measured assertion needs best-of-N and a bound with room for it, or it flakes in the coordinator's run.
- **Lab SVGs at 390px scale text to ~6px.** Wrapping each in `.chartwrap{overflow-x:auto}` with
  `.chartwrap > .svgfig{min-width:500px}` keeps labels legible and passes the probe.
- **Sheet 13 already named the staging vocabulary** (line-side bin, cell supermarket, plant store;
  bench and stores). Grep the earlier sheets for a bridge before coining new words for it.
- Verifier after this block: **983 PASS, 0 `*** FAIL ***`, rc 0** (whole file, with the Ch15/17/18
  blocks present); `_ch16p1` contributes 84 `[Ch16 sNN]` lines, 83 of them PASS checks.

## 6. Forward promises (the page's §16, "What part 2 owes you")

Part 2 (files 04–05) owes seven calculations:
1. The roofline with a GPU's numbers: peak and ridge from the source's own GPU figures (FMA = 2 FLOP),
   placing ReLU 0.125, layer norm 0.625, naive matmul 0.25, tiled matmul `b/4`, conv 139.6 FLOP/B; say which change station.
2. Little's law at GPU scale: warps in flight = latency x issue rate from the source's latency figures;
   where warps-as-cards holds and breaks (hardware switches warps; the compiler allocates accumulators).
3. Shared-memory tiles: the CUDA tiled matmul as §07's kitting with a programmer-managed bin; intensity
   of the source's tile by `b/4`, checked against the GPU ridge.
4. Warp divergence: the drilling head with drills removed on a 32-lane warp; the exact cost count, what
   SIMT adds beyond a CPU mask, where the head picture fails.
5. Matrix engines: MACs per instruction for a tensor core and a TPU systolic array, beside AMX's 8,192 and
   SME's 256; the array computing `sum_k a_k b_k^T`.
6. Fusion priced: the `(1+2k)`-to-1 traffic ratio for a fused Triton kernel on a real layer shape.
7. The third station: PCIe 4.0 x16 = 31.5 GB/s each way as a delivery station ahead of GPU memory; when it binds.

Part 3 (files 06–07) owes: RVV as SVE's vector-length-agnostic predicates again; Vulkan subgroups as
lanes; and a chapter-wide concept map joining all three parts. Each later part carries its own Appendix A.

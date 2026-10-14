# Chapter 16 part 3 of 3 — SIMD and GPU Programming: portability and the edge (source files 06–07)

## 1. State row

| 16 | SIMD and GPU Programming *(pt 3: portability and the edge)* | `ch16-simd-gpu-part3/` | https://claude.ai/artifact/SHS1r9AzoMUwqskvgMc4G7 |

## 2. README entry

For the Verification paragraph: "and — for Chapter 16 part 3 — RVV strip-mining simulated for every
n up to 3,000 at seven widths under both of the spec's `vsetvl` policies, the source's RVV loop
compiled for RISC-V with clang (11 instructions per pass), file 06's three C++ tasks compiled and
run, all six of file 07's GLSL shaders compiled to SPIR-V with glslang's WebAssembly build, energy
per FLOP from the source's own device table against part 1's cube law, file 07's reduction tree
counted and run in float32 through both dispatches, and its softmax shader simulated past one
workgroup."

Sheet entry: portability and the edge, closing Chapter 16 and paying part 2's six debts.
Headline: **the edge is smaller, not cheaper** — by the source's own table the server GPU,
smartphone and microcontroller spend 0.70, 0.50 and 1.00 pJ per FLOP across a 700,000-fold range
of power, and part 1's cube law would have put the microcontroller at 4.57 fJ, 219x lower: part 1
§03's leakage warning, quantified. Exact: **`vsetvl` strip-mining is part 1's drilling head with the
width read off the machine** — ceil(n/VLMAX) passes whether the hardware fills greedily or balances
the last two (exhaustive, n <= 3,000); subgroups as heads of vendor-set width, with Sheet 15's c = 0
count at sample s (27.5% / 47.4% / 72.4% at 32/64/128 lanes for a 1% branch); file 07's reduction as
a **whiffletree** (255 beams, 8 levels, 9 barriers, 12.5% of lane-slots, bit-repeatable where part
2's atomics gave 14 totals); Vulkan's minimum limits as **design allowables** (largest guaranteed
one-output tile 11, at 2.75 FLOP/B); energy per FLOP as **specific fuel consumption**, rising at part
load; task 2's quantisation error as reading uncertainty (0.2496 steps against 1/4). Twelve
corrections, four labs, a chapter-wide map (six ideas across three parts) and the chapter's end matter.

## 3. Bridge bank update

**16 SIMD and GPU.** **Part 3 written; Chapter 16 complete.** The bank's two entries held across all
three parts: bandwidth as material flow became a line with two stations (CPU), three (GPU, with the
PCIe dock) and a phone ridge near 195 FLOP/B (reported bandwidth); "SIMD ↔ multi-spindle machining"
was right for the drilling head only, and that head counted strokes on every machine, ending as
`vsetvl` and Vulkan subgroups. Outside the bank, per part: Little's law (accumulators, warps, a job too
small to fill the pipe), kitting, the c = 0 plan, the whiffletree, design allowables, and specific fuel
consumption with its part-load rise.

## 4. The lesson worth generalising

**Divide the source's table by itself.** The three-device table was printed to show how extreme the
edge is; dividing compute by power, a one-line operation on numbers already on the page, gave the
sheet's headline and contradicted the impression the table was meant to give. Then test the series'
own earlier law against it (part 1's cube law, off by 219x) rather than citing the law. Second: when
there is no hardware, **compile what can be compiled**: clang's RISC-V target and an npm WebAssembly
glslang turned "cannot run" into instruction counts and 6/6 shaders checked.

## 5. New traps, and verifier count

- **Tools for absent hardware exist in the package managers.** `clang --target=riscv64-unknown-elf
  -march=rv64gcv -ffreestanding -S` compiles RVV intrinsics with no sysroot; `npm pack @webgpu/glslang`
  gives a working GLSL-to-SPIR-V compiler in Node. The verifier skips both with a NOTE when absent
  (the glslang path is in this session's scratchpad, so a fresh session will skip it).
- **A unit slip that keeps its units survives review**: task 1's "MCU cycles" are host time x MCU clock.
  Check a printed count against the work it must contain (50,816 multiply-adds).
- **The worktree guard refuses shell lines with heredocs plus pipes**: write scripts to the scratchpad.
- Verifier after this block: **1,206 PASS, 0 `*** FAIL ***`, rc 0** (system python; part 2's JAX
  checks skipped); `_ch16p3` prints 56 `[Ch16p3 sNN]` lines, 49 PASS. Task 3's printed timing is
  reported, not asserted. Probe: five widths clean, 22 rest-value expectations. `audit_classes.py` rc 0.

## 6. Forward promises

None: Chapter 16 is complete. The endnote points back to parts 1–2 and on to Sheet 17, which already
cites "Sheet 16" for the roofline and bandwidth. Paid here, all six of part 2 §19: RVV as predicates
(§02), subgroups as lanes (§07), the tree reduction counted (§08), the GLSL kit priced (§09), FLOPs per
joule (§04), and the chapter-wide map (§14).

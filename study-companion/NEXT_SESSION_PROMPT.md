# Starter prompt for the next session

Paste everything below the line into a new Claude Code session on this repository.

---

Finish the study-companion series in `Vatsal-Kanojiya/maths-cs-ai-compendium`, on branch
`claude/nifty-pascal-e4h18c` (if the session is bound to another branch name, fast-forward that
branch to this one first and say so in HANDOFF). Do not open a PR.

**Read `study-companion/HANDOFF.md` first, all of it** (about 1,200 lines; two or three Read
calls). It is the continuation spec: state table, reader profile, non-negotiable rules, build
procedure, the server-less browser probe (`study-companion/tools/probe.js`), the traps, and one
lesson per chapter. Then run:

```
pip install numpy scipy
python3 study-companion/verify_claims.py   # ~3 minutes: use a long timeout. Expect 1,072 PASS, 0 "*** FAIL ***", rc 0
python3 study-companion/audit_classes.py   # exit code 0
```

Check the verifier's return code and stderr, not only its stdout. A FAIL tagged `[Ch16 s01/s08/
s10/s11]` is a known flaky timing check (Known issues), not a real error — fixing it is part of
this session's work.

**The reader.** A mechanical engineering graduate from an Indian tier-3 college with no CS/IT
background, who prepared seriously for GATE. **Governing constraint, in the reader's words:**
"but be caucious, sometimes analogy looses the truth so never ever sacrifice concepts and truth
for ease of understanding". Every analogy gets an honest verdict and an amber "where it breaks" box.

**State.** Everything is published except **Sheet 16 parts 2 and 3**: chapters 01–15, Sheet 16
part 1 (the CPU), 17, 18 and a combined 19–20 coda. URLs are in the HANDOFF state table.

**Next, in order:**
1. **Sheet 16 part 2 — the GPU** (source files 04–05, 991 lines) into `ch16-simd-gpu-part2/`,
   verifier stub `_ch16p2()`. Its contract is Sheet 16 part 1 §16 — seven calculations, also listed
   in `study-companion/handoff-notes/ch16-part1.md` §6 — plus Sheet 15's promise that this sheet
   explains why a GPU run is not bit-exactly repeatable. No GPU in the container: simulate what can
   be simulated, mark hardware figures as the source's. End with a forward-promise section to part 3
   and its own Appendix A.
2. **Sheet 16 part 3 — portability and the edge** (files 06–07, 1,096 lines) into
   `ch16-simd-gpu-part3/`, stub `_ch16p3()`: RVV as SVE-style predicates, Vulkan subgroups as lanes,
   the chapter's end matter and a concept map joining all three parts.
3. Make Sheet 16 part 1's timing checks deterministic (best-of-N and wider bounds, or a `--timing`
   flag) without weakening what they assert.

**Workflow, per part:** read the whole source and run its code; find exact bridges; verify every
numeric claim in `verify_claims.py` **before** writing prose; build from Sheet 16 part 1's page
(same skeleton and palette); probe with `tools/probe.js --shots --expect` and read every screenshot;
run `audit_classes.py`; publish; update README, HANDOFF and this prompt; commit and push. **Commit
and push after every checkpoint** (verifier block, each page chunk, probe fixes, publish) — usage
limits have stopped runs mid-page several times, and only pushed work survives.

Commit trailers:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: <this session's URL>
```

**Habits that matter:** write the claim, then try to break it — including your own corrections to
the source; run the source's code; look for hand procedures from the reader's training (they make
the best bridges, and a confirmation is as valuable as a correction); state every source correction
with the computation that shows it; no internet research — mark anything not derived here as
"reported".

Start by reading the HANDOFF, then run the verifier, then begin Sheet 16 part 2.

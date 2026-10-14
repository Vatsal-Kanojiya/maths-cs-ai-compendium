# Starter prompt for a follow-up session

The study-companion series is **complete**: every chapter of the compendium has a published sheet
(01–15, 16 in three parts, 17, 18 and a combined 19–20 coda). Use this prompt only for optional
clean-up. Paste everything below the line into a new Claude Code session on this repository.

---

Work on the study-companion series in `Vatsal-Kanojiya/maths-cs-ai-compendium`, branch
`claude/nifty-pascal-e4h18c` (if the session is bound to another branch name, fast-forward that branch
to this one first and say so in HANDOFF). Do not open a PR.

Read `study-companion/HANDOFF.md` first, all of it — especially **Known issues** and **What is actually
left**. Then run:

```
pip install numpy scipy
python3 study-companion/verify_claims.py   # 3-4 minutes: long timeout. Expect 1,206 PASS (one timing NOTE possible), 0 "*** FAIL ***", rc 0
python3 study-companion/audit_classes.py   # exit code 0
```

The reader: a mechanical engineering graduate (tier-3 Indian college, GATE-prepared, no CS background).
Their rule: "never ever sacrifice concepts and truth for ease of understanding".

Optional clean-ups, in the order I would do them (each one: edit, probe with
`study-companion/tools/probe.js --shots`, republish the SAME artifact URL after reading it, commit, push):
1. Fix the two wrong cross-references (Sheet 06 §06 → no EWMA in Sheet 04; Sheet 09 §19 → Chapter 03 §07).
2. Soften Sheet 13 part 2 §10's "runs identically everywhere" per Sheet 15 §19.
3. Make Sheet 16 part 1's ReLU unrolling check use the per-run ratio (see Known issues).
4. Re-probe and fix the nine sheets whose display equations are wider than their box at 390 px.

Commit and push after every fix. Commit trailers:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: <this session's URL>
```

# Starter prompt for the next session

Paste everything below the line into a new Claude Code session on this repository.

---

Continue the study-companion series in `Vatsal-Kanojiya/maths-cs-ai-compendium`, on branch
`claude/blissful-euler-0c6xih`. Develop and push there only. Do not open a PR.

**Read `study-companion/HANDOFF.md` first, all of it.** It is the continuation spec: state
table, reader profile, non-negotiable rules, build procedure, browser-probe recipe, the traps
that have bitten before, and one lesson per chapter. Then run the checks it asks for:

```
pip install numpy scipy
python3 study-companion/verify_claims.py        # expect 612 PASS, 0 "*** FAIL ***", return code 0
python3 study-companion/audit_classes.py        # every sheet should pass (exit code 0)
```

Check the verifier's return code and stderr, not only its stdout.

**The reader.** A mechanical engineering graduate from an Indian tier-3 college with no CS/IT
background, who prepared seriously for GATE. Every page is written for that one person.

**Governing constraint, in the reader's words:** "but be caucious, sometimes analogy looses the
truth so never ever sacrifice concepts and truth for ease of understanding". Every analogy gets
an honest verdict and an amber "where it breaks" box.

**State.** Sheets 1–14 of 20 are built and published. Chapter 13 is in three parts and
Chapter 14 in two. `verify_claims.py` has 612 checks with zero failures.

**Next: Chapter 15, production software engineering.** The source is
`chapter 15 - production software engineering/`, five files and 1,474 lines, so it is a single
run. The bridge bank's entry (version control ↔ drawing revision control and ECN/ECO, testing ↔
inspection and QA, CI/CD ↔ line automation, containers ↔ jigs and fixtures) is a floor, not a
ceiling. Chapter 14 part 2 made two promises to Sheet 15 that must be paid:
- §18: the longest common subsequence "becomes version control";
- §22: hashing is how version-control systems name what they store.

After that come Chapters 16 (three parts, the largest source after Chapter 09), 17 and 18.
Chapters 19–20 are outline stubs; decide deliberately whether to write them.

**Already done:** the Chapter 09–13 skeleton repair. It is fixed in the repo and republished at new
URLs, which are in the HANDOFF state table. `audit_classes.py` passes on every sheet; keep it that
way.

**Workflow, per chapter or part:**
1. Read the whole source.
2. Find exact bridges.
3. Verify every numeric claim in `verify_claims.py` **before** writing prose. Put each new
   block inside its own function, like `_ch14p1()`, so its names cannot collide.
4. Build the page from **Chapter 14's skeleton**, not from Chapters 09–13. Use a quoted heredoc
   and an `@SRC@` placeholder, plain-SVG labs with a prediction panel first, and the validated
   `--c1…--c5` palette.
5. Probe at five widths in a real browser, checking readout values against the verified
   numbers, and screenshot every lab. Run `audit_classes.py`.
6. Publish the artifact.
7. Update README and HANDOFF.
8. Commit and push.

Commit trailers:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: <this session's URL>
```

**Habits that matter:**
- Write the claim, then try to break it before you believe it. That includes your own
  corrections to the source: two of Chapter 14's draft corrections were refuted by computation.
- Run the source's code.
- Look for hand procedures from the reader's training, such as Fulkerson's rule, the method of
  joints or the slip-gauge rule. They make the best bridges, and a confirmation is as valuable
  as a correction.
- Report honestly. State every source correction together with the computation that shows it.
- Do no internet research. Mark anything not derived here as "reported".

Start by reading the HANDOFF, then run the verifier, then begin Chapter 15.

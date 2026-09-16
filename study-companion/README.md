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

## Chapters

| # | Chapter | Page | Published |
|---|---------|------|-----------|
| 01 | Vectors | [`ch01-vectors/`](ch01-vectors/index.html) | [Read online](https://claude.ai/artifact/98j4cnE4QvTKi1znhH6vh9) |

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

## Technical notes

- Mathematics renders through MathJax (SVG output) loaded from CDN. If the CDN is
  unreachable the page degrades the TeX to readable Unicode rather than showing
  raw markup.
- Light and dark themes are both designed, with a manual toggle in the header.
- Chart palettes are validated for colour-vision deficiency separation and
  contrast in both themes.
- Pages are responsive down to phone width.

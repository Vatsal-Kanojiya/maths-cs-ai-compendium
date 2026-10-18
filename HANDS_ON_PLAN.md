# Hands-On ML Engineering Plan

A plan for adding practical, real-data projects to the compendium. The goal is to close the gaps between "knows the theory" and "can do the job of an ML engineer":

1. Real data and real frameworks (pandas on messy data, PyTorch, Hugging Face).
2. End-to-end projects: data → training → deployment → monitoring.
3. Data engineering: SQL, pipelines, orchestration, Spark.
4. Tooling habits: experiment tracking, data versioning, GPU cloud workflows.
5. Interview preparation: ML system design and take-home exercises.

## Principles

- **Fundamentals first.** Milestones apply what a file already teaches; they do not replace or re-explain the theory.
- **Learn → do → do better.** Two running projects grow through the chapters. Each milestone improves the previous version, so the reader sees why each technique matters.
- **No forced projects.** Only files where a project step fits naturally get a milestone. Theory-heavy files (learning theory, probabilistic graphical models, most of chapters 01-05, 13, 14 and 20) get none.
- **No datasets in Git.** Data is downloaded in code from stable public sources and cached locally. This keeps the repo small and respects dataset licences. Pin dataset versions for reproducibility.
- **Laptop first.** Everything runs on a modest Linux laptop (tested target: i5 5th gen, 12 GB RAM, SSD) except GPU fine-tuning, which uses free Google Colab or Kaggle notebooks. Run one heavy tool at a time.
- **Short milestones.** A few paragraphs plus one code block that builds on the previous milestone. Shared setup code lives once in the chapter 21 hub.

## The two running projects

| Project | Dataset | How to load | Runs on |
|---|---|---|---|
| Tabular: income / risk prediction | Adult income (census, ~49k rows; missing values, categoricals, imbalance; includes sex and race, useful for the fairness audit) | `sklearn.datasets.fetch_openml("adult", version=2, as_frame=True)` | Laptop |
| Text: news or review classification | AG News or IMDB reviews | `datasets.load_dataset("ag_news")` or `load_dataset("imdb")` (Hugging Face) | Colab GPU for fine-tuning, laptop for serving |

## Format of a milestone section

Each milestone is a section added after a file's `## Coding Tasks` block:

```
## Project Milestone N: <title>

- What this milestone adds and why, in terms of what this file just taught (2-4 bullets).
- What to look for in the results.

<one ```python block that builds on the previous milestone>

- "Make it better" extensions: 2-3 ideas for the reader to try on their own.
```

House style for the repo: British spelling, bullet-point prose, `$$` maths, images as hand-written SVGs in `images/`, nested lists indented by **four** spaces (two spaces renders as flat bullets). Every code block should run top to bottom on its own.

## Steps

Do these in order and commit after each one.

| Step | What | Where | Notes |
|---|---|---|---|
| 1 | **Hub**: laptop setup (Linux, Python env with `uv` or `venv`, Docker, Colab for GPU), a `load_adult()` / data-cache helper, the full roadmap with links to every milestone, and the order to do them in | New `chapter 21 - hands-on ML engineering/01. setup and roadmap.md`; add to `mkdocs.yml` nav, `llms.txt` and the README outline | Define the shared helper here once |
| 2 | **Tabular M1**: load Adult, clean it, explore it, first baseline (dummy + logistic regression) | `chapter 06 - machine learning/08. practical data work.md` | Reuse that file's pipeline ideas |
| 3 | **Tabular M2**: gradient boosting, then proper cross-validation, then fix what it reveals (leakage, threshold choice, calibration) | `chapter 06/06. tree ensembles and gradient boosting.md` and `chapter 06/07. model evaluation and validation.md` | |
| 4 | **Tabular M3**: explain the model (permutation importance, SHAP-style), then a fairness audit by sex and race, then simulate drift | `chapter 06/13. interpretability.md` and `chapter 06/15. fairness, robustness and distribution shift.md` | |
| 5 | **Tooling M4**: MLflow experiment tracking and DVC data versioning | `chapter 15 - production software engineering/05. deployment and devops.md` (or `04. testing and quality assurance.md` for the tests part) | |
| 6 | **Serving M5**: FastAPI + Docker, a CI test (GitHub Actions), monitoring (prediction logging, drift check) and a retraining trigger | `chapter 17 - AI inference/03. serving and batching.md` and `chapter 18 - ML systems design/04. ML systems design.md` | Plays to backend experience |
| 7 | **Data pipeline M6**: SQL features with DuckDB, a small Prefect or Airflow pipeline, Spark in local mode | `chapter 18/03. large scale infrastructure.md`, or a chapter 21 file if nothing fits naturally | |
| 8 | **PyTorch on CPU**: training loop, `Dataset`/`DataLoader`, checkpointing, debugging a model that will not learn | `chapter 06 - machine learning/03. deep learning.md` | Small model; CPU is fine |
| 9 | **Text project**: fine-tune a small transformer (e.g. DistilBERT) on Colab, push to the Hugging Face Hub, then serve it on the laptop (optionally quantised) | `chapter 07 - computational linguistics/04. transformers and language models.md` and `chapter 17/01. quantisation.md` | GPU part marked clearly as Colab |
| 10 | **Interview prep**: worked ML system design walkthroughs, a take-home exercise template, a checklist | New `chapter 21/02. interview preparation.md` | |
| 11 | **Final pass**: README, nav, `llms.txt`, full site build | — | |

Placements in steps 5-7 are provisional: check the target file first, and put the milestone in chapter 21 instead if it would be forced into a theory file.

## Checking your work

- Run each code block top to bottom in a fresh environment.
- Build the site locally to check rendering: create the `docs/` symlinks as in `.github/workflows` (or copy what the deploy workflow does), then `mkdocs serve`. The `social` plugin needs cairo; remove it from a local copy of `mkdocs.yml` if it fails.
- Check that nested lists render as nested (four-space indent) and that maths renders.

## Status

- Planned on 18 October 2026 on branch `claude/hopeful-maxwell-ixge0f`. No step has been started.
- Note: the cloud sandbox used to write this plan could not reach OpenML or Hugging Face, so code there could only be tested on stand-in data. On a normal internet connection the loaders above work directly.

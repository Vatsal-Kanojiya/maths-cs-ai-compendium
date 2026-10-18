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

## Levels

The hands-on work is organised into three levels so it reads as a path, not a pile (chapter 21's roadmap has the full reading order and "done when" checklists):

| Level | Goal | Milestones |
|---|---|---|
| 1. Foundations | Messy data to a trustworthy, explained model: "knows ML" | 1-3 |
| 2. Production | Ship it: tracking, serving, CI, monitoring, pipelines, PyTorch, fine-tuning; plus interview prep: "can ship ML" | 4-8 |
| 3. Frontier | Build, scale and post-train language models; evaluate LLM systems; reproduce research: "can build what the labs build" | 9-13 |

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
| 6 | **Serving M5**: FastAPI + Docker, a CI test (GitHub Actions), monitoring (prediction logging, drift check) and a retraining trigger | `chapter 18 - ML systems design/04. ML systems design.md` (chapter 17's serving file is about LLM serving, so the milestone went to chapter 18 only) | Plays to backend experience |
| 7 | **Data pipeline M6**: SQL features with DuckDB, a small Prefect or Airflow pipeline, Spark in local mode | `chapter 18/03. large scale infrastructure.md`, or a chapter 21 file if nothing fits naturally | |
| 8 | **PyTorch on CPU**: training loop, `Dataset`/`DataLoader`, checkpointing, debugging a model that will not learn | `chapter 06 - machine learning/03. deep learning.md` | Small model; CPU is fine |
| 9 | **Text project**: fine-tune a small transformer (e.g. DistilBERT) on Colab, push to the Hugging Face Hub, then serve it on the laptop (optionally quantised) | `chapter 07 - computational linguistics/04. transformers and language models.md` and `chapter 17/01. quantisation.md` | GPU part marked clearly as Colab |
| 10 | **Interview prep**: worked ML system design walkthroughs, a take-home exercise template, a checklist | New `chapter 21/02. interview preparation.md` | |
| 11 | **Final pass**: README, nav, `llms.txt`, full site build | — | |
| 12 | **Level 3, M9: GPT from scratch**: character and BPE tokenisation, a decoder-only transformer, a training loop with evaluation, sampling, and a small scaling experiment (model size against loss), on TinyShakespeare or TinyStories | `chapter 07 - computational linguistics/04. transformers and language models.md` | Laptop CPU for a tiny model; Colab GPU for the scaling experiment |
| 13 | **Level 3, M10: efficient and distributed training**: mixed precision, gradient accumulation, activation checkpointing, real data-parallel training with `torch.distributed` across CPU processes, and profiling, applied to the M9 model | `chapter 06 - machine learning/05. distributed deep learning.md` | DDP with the gloo backend runs on a laptop |
| 14 | **Level 3, M11: post-training a small open model**: SFT with LoRA on an instruction dataset, then GRPO-style RL with a verifiable reward (arithmetic or code tests), with a before-and-after eval | `chapter 07 - computational linguistics/05. advanced text generation.md` | Colab GPU; a model of about 0.5B parameters |
| 15 | **Level 3, M12: an LLM application with evals**: retrieval or tool use over a document set, a hand-labelled eval set, an LLM judge calibrated against human labels, regression tests, cost and latency budgets | `chapter 19 - applied AI/04. agentic systems.md` | Works with a hosted API or a local model |
| 16 | **Level 3, M13: reproducing a paper**: a guide to reading and reproducing papers, with one worked example small enough for a laptop | New `chapter 21 - hands-on ML engineering/03. reproducing a paper.md` | |
| 17 | **Final pass for Level 3**: roadmap statuses, README, nav, `llms.txt`, site build | — | |

Placements in steps 5-7 are provisional: check the target file first, and put the milestone in chapter 21 instead if it would be forced into a theory file.

## Checking your work

- Run each code block top to bottom in a fresh environment.
- Build the site locally to check rendering: create the `docs/` symlinks as in `.github/workflows` (or copy what the deploy workflow does), then `mkdocs serve`. The `social` plugin needs cairo; remove it from a local copy of `mkdocs.yml` if it fails.
- Check that nested lists render as nested (four-space indent) and that maths renders.

## Status

- Steps 1-11 (Levels 1 and 2) are complete. Steps 12-17 (Level 3) are planned, in the order above: each builds on the one before.

- All eleven steps are complete on branch `claude/hopeful-maxwell-ixge0f` (18 October 2026). Chapter 21's roadmap links every milestone.
- Placements that differ from the table above: milestone 5 (serving and monitoring) went to chapter 18's ML systems design file only, because chapter 17's serving file is about LLM serving; the text project's quantisation uses ONNX Runtime rather than PyTorch's deprecated `torch.ao.quantization`.
- Testing: every code block ran in the cloud sandbox, but on a stand-in dataset with the same columns as Adult (OpenML and the Hugging Face Hub were unreachable there), and the transformer code ran with a tiny locally built model. The Docker build could not be tested there. **First real-data runs happen on your machine**; the text gives expectations for the real data as rough ranges.

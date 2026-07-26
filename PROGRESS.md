# factr.ai — Project Progress

## Overarching goal

Build a working multimodal misinformation detector for social media content
(originally scoped as Instagram): given an image + caption, produce a
misinformation-risk score by combining multiple AI signals — visual-text
consistency, deepfake/manipulation detection, and text credibility — into a
weighted ensemble. Two research questions drive the model choices:

- **OpenCLIP** for cheap, zero-shot image/text alignment and as a frozen
  feature extractor for two supervised probes.
- **LLaVA** for natural-language visual reasoning (does the image actually
  support what the caption claims; does the image look manipulated).
- Calibrate/train against **LIAR** (political statement truthfulness) and
  **DFDC** (deepfake video) — the two most relevant public labeled datasets,
  even though neither is a perfect match for "Instagram post" data.

Delivery form factor is intentionally undecided long-term — started as a
browser extension, currently an API/CLI tool (see decision log below).

## Decision log

- **2026-07-26**: Repo had accumulated ~10 abandoned attempts at this
  (`main_broken.py`, `main_no_clip.py`, multiple stripped `requirements_*.txt`)
  all hitting the same wall: LLaVA + a large OpenCLIP checkpoint don't fit
  free-tier hosting (Railway). Decided to resolve this by running LLaVA via
  cloud inference (Replicate) instead of loading it locally, keeping only
  OpenCLIP (small) local.
- **2026-07-26**: Paused the browser-extension direction. Building an
  API/CLI first to prove the detection pipeline actually works end to end,
  before taking on extension-store review and Instagram DOM-scraping
  fragility. Revisit once the ensemble is calibrated and accurate.
- **2026-07-26**: LIAR and DFDC don't share a schema with each other or with
  "Instagram post" data (LIAR = text-only statements, DFDC = video, no
  captions). Resolved by training two *separate* linear probes on frozen
  CLIP embeddings — one per dataset — rather than pretending there's one
  unified training set. See `factr/heads.py`.

## What's implemented (this session)

New `factr/` package, verified working end-to-end on this machine
(Python 3.14, see `requirements-factr.txt`):

- `factr/clip_scorer.py` — OpenCLIP wrapper (image/text embeddings,
  zero-shot consistency score). Confirmed working: model downloads and runs.
- `factr/heads.py` — `ProbeHead`: logistic regression on frozen CLIP
  embeddings, with save/load/train/evaluate. Confirmed working on a
  synthetic true/false example.
- `factr/llava_client.py` — calls LLaVA via Replicate, prompts for
  consistency + authenticity scores, parses the response, falls back
  cleanly if `REPLICATE_API_TOKEN` is unset or the call fails.
- `factr/ensemble.py` — `MisinformationEnsemble`: fuses
  `clip_consistency` + `clip_visual_head` + `clip_text_head` + `llava` with
  configurable weights (`models/ensemble_weights.json`, defaults to equal).
  **Confirmed it degrades gracefully** — ran with only `clip_consistency`
  available (no trained heads, no Replicate token) and correctly
  renormalized weight to 1.0 for that signal.
- `factr/data/liar.py`, `factr/data/dfdc.py` — kagglehub-based download +
  parsing (LIAR tsv → statement/label/risk_score; DFDC metadata.json →
  video labels + OpenCV frame extraction).
- `factr/train/train_text_head.py`, `factr/train/train_visual_head.py` —
  CLI training scripts, `--download` or `--data-dir`, print train/test
  accuracy + AUC, save to `models/*.joblib`.
- `factr/api.py` — FastAPI service (`/health`, `/analyze`,
  `/analyze/upload`). Confirmed working: booted the server, fetched a live
  image URL, ran real CLIP inference, returned a scored JSON response.
- `factr/cli.py` — one-off CLI analysis, no server needed. Confirmed
  working against a local test image.
- `requirements-factr.txt` — pinned to exactly what installed and ran
  successfully in this session.
- `README_FACTR.md` — architecture rationale + setup/train/run commands.

Legacy root-level files (`main*.py`, `instagram_ensemble_detector.py`,
`training_pipeline.py`, `instagram_data_collector.py`, the various
deployment scripts) are untouched — superseded by `factr/` but left in case
anything (e.g. the Instagram-pattern regex heuristics, the scraper) is worth
salvaging later.

## What's NOT done / next session

1. **Neither head is actually trained yet.** `train_text_head.py` and
   `train_visual_head.py` were only smoke-tested on tiny synthetic data —
   nobody has run `--download` against the real LIAR/DFDC datasets. DFDC
   requires manually accepting the competition rules at
   https://www.kaggle.com/c/deepfake-detection-challenge first (the API
   can't do this step). Next session: get Kaggle credentials set up, run
   both training scripts for real, record the accuracy/AUC numbers here.
2. **No ensemble-weight calibration.** Weights are currently equal by
   default. Need a small labeled validation set (image + caption +
   misinformation label) to grid-search/fit weights across all four
   signals — `instagram_training_data.db` schema from the old
   `instagram_data_collector.py` could be reused/adapted for this, or a
   fresh small hand-labeled set could work as a first pass.
3. **No real accuracy numbers on Instagram-like data.** Everything so far
   has been validated for *plumbing correctness* (does it run, does it
   degrade gracefully), not for detection quality. Need an actual eval set
   of real Instagram posts (or similar) with ground-truth labels.
4. **`REPLICATE_API_TOKEN` never exercised against the real API** — the
   LLaVA path has only been tested via its fallback (token unset). Get a
   token and verify the prompt/parsing actually works against live LLaVA
   output, tune `PROMPT` in `factr/llava_client.py` if parsing is flaky.
5. **Form factor decision deferred, not resolved.** Once the ensemble is
   calibrated and shows it's actually accurate, revisit whether to build
   the browser extension, keep it a web app, or something else.
6. **Repo cleanup deferred.** Once `factr/` is confirmed to be the
   long-term direction, consider removing/archiving the superseded
   root-level files and old deployment configs.

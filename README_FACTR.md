# factr — OpenCLIP + LLaVA misinformation ensemble

A working, from-scratch reimplementation of the detection core. Ships as an
**API/CLI tool**, not a browser extension — see "Why" below.

## Why this architecture

The repo's history (`main_broken.py`, `main_no_clip.py`, several stripped-down
`requirements_*.txt`) shows the same problem hit repeatedly: LLaVA (7B+) and
a large OpenCLIP checkpoint do not fit on free/cheap hosting. This version
resolves that by:

- Running **OpenCLIP locally** (it's small enough — a few hundred MB — to run
  on CPU) for embeddings, zero-shot image/caption consistency, and as the
  frozen feature extractor for two trained linear probes.
- Calling **LLaVA via Replicate** (cloud GPU inference) instead of loading it
  in-process. Requires `REPLICATE_API_TOKEN`; if unset, the ensemble just
  drops that signal and renormalizes the remaining weights.
- Skipping the browser extension for now — an API + CLI is enough to prove
  the detection pipeline actually works end to end, without also taking on
  extension store review and Instagram DOM scraping fragility.

## Ensemble signals

| Signal | Source | Trained on |
|---|---|---|
| `clip_consistency` | Zero-shot CLIP image/text cosine similarity | nothing (no training needed) |
| `clip_visual_head` | Logistic regression on frozen CLIP image embeddings | DFDC (real/fake video frames) |
| `clip_text_head` | Logistic regression on frozen CLIP text embeddings | LIAR (6-way truthfulness, binarized) |
| `llava` | Replicate-hosted LLaVA rating consistency + authenticity in natural language | zero-shot, not fine-tuned |

Weights live in `models/ensemble_weights.json` (falls back to equal weights
in `factr/config.py` if that file doesn't exist). Any signal that isn't
available at request time (head untrained, no Replicate token, network
failure) is dropped and the rest are renormalized — the ensemble degrades
gracefully instead of failing.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-factr.txt
```

Optional environment variables:

```bash
export REPLICATE_API_TOKEN=...        # enables the llava signal
export KAGGLE_USERNAME=... KAGGLE_KEY=...   # or ~/.kaggle/kaggle.json — needed to --download datasets
```

## Train the two heads

DFDC requires accepting the competition rules on kaggle.com first (the API
can't bypass that): https://www.kaggle.com/c/deepfake-detection-challenge

```bash
python -m factr.train.train_text_head --download            # LIAR
python -m factr.train.train_visual_head --download --limit 400   # DFDC sample (400 videos)
```

Each prints train/test accuracy + AUC and saves to `models/*.joblib`. Without
these, `clip_consistency` (and `llava`, if configured) still work — the
ensemble just runs with fewer signals.

## Run it

```bash
# API
uvicorn factr.api:app --reload
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' \
  -d '{"image_url": "https://.../photo.jpg", "caption": "..."}'

# CLI
python -m factr.cli --image-url https://.../photo.jpg --caption "..."
python -m factr.cli --image-path ./photo.jpg --caption "..." --no-llava
```

## What's not done

- `models/ensemble_weights.json` calibration script (grid-search/logistic
  regression over labeled validation data combining all four signals) isn't
  built yet — weights are currently equal by default. Worth doing once
  you have a labeled Instagram-style validation set.
- The legacy root-level files (`main*.py`, `instagram_ensemble_detector.py`,
  `training_pipeline.py`, `instagram_data_collector.py`, the various
  `requirements_*.txt` / deployment scripts) are untouched. They're
  superseded by `factr/` but left alone in case you want to salvage
  anything (e.g. the Instagram-pattern heuristics or scraper).

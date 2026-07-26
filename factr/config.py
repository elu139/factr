"""Central configuration, all overridable via environment variables."""

import json
import os
from pathlib import Path

MODEL_DIR = Path(os.getenv("FACTR_MODEL_DIR", "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# OpenCLIP backbone. ViT-B-32 keeps the download (~600MB) and CPU inference
# time reasonable; override to a bigger checkpoint (e.g. ViT-L-14 / laion2b_s32b_b82k)
# once you have GPU capacity.
OPEN_CLIP_MODEL = os.getenv("FACTR_CLIP_MODEL", "ViT-B-32")
OPEN_CLIP_PRETRAINED = os.getenv("FACTR_CLIP_PRETRAINED", "laion2b_s34b_b79k")

# LLaVA is served via Replicate instead of loaded locally -- a 7B+ VLM does not
# fit free/cheap hosting. Requires REPLICATE_API_TOKEN in the environment.
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_LLAVA_MODEL = os.getenv("FACTR_LLAVA_MODEL", "yorickvp/llava-13b")

VISUAL_HEAD_PATH = MODEL_DIR / "visual_head.joblib"
TEXT_HEAD_PATH = MODEL_DIR / "text_head.joblib"
ENSEMBLE_WEIGHTS_PATH = MODEL_DIR / "ensemble_weights.json"

# Default ensemble weights, used until calibrate_ensemble.py produces tuned
# weights from labeled validation data. Any signal that fails to load/run
# (missing head, missing API token, network error) is dropped and the
# remaining weights are renormalized -- see ensemble.py.
DEFAULT_WEIGHTS = {
    "clip_consistency": 0.25,
    "clip_visual_head": 0.25,
    "clip_text_head": 0.25,
    "llava": 0.25,
}


def load_ensemble_weights() -> dict:
    if ENSEMBLE_WEIGHTS_PATH.exists():
        with open(ENSEMBLE_WEIGHTS_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_WEIGHTS)

"""Download and load the DFDC (DeepFake Detection Challenge) sample dataset
for training the visual-manipulation head.

The full training set is ~470GB; `train_sample_videos.zip` (~4GB, 400 clips
with a metadata.json of REAL/FAKE labels) is what most people actually train
against and is what `download()` pulls by default. Requires having accepted
the competition rules on kaggle.com (the API can't bypass that).
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_COMPETITION = "deepfake-detection-challenge"


def download(competition: str = DEFAULT_COMPETITION) -> Path:
    import kagglehub

    path = kagglehub.competition_download(competition)
    logger.info("DFDC dataset available at %s", path)
    return Path(path)


def load_metadata(dataset_dir: Path) -> dict:
    """Returns {filename: {"label": "REAL"|"FAKE", ...}}."""
    dataset_dir = Path(dataset_dir)
    candidates = list(dataset_dir.rglob("metadata.json"))
    if not candidates:
        raise FileNotFoundError(f"No metadata.json found under {dataset_dir}")
    with open(candidates[0]) as f:
        metadata = json.load(f)
    return metadata, candidates[0].parent


def extract_frame(video_path: Path, frame_fraction: float = 0.5) -> Optional[Image.Image]:
    """Grab a single representative frame from a video as a PIL Image."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = max(0, int(total_frames * frame_fraction) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def iter_labeled_frames(dataset_dir: Path, limit: Optional[int] = None) -> Iterator[Tuple[Image.Image, int, str]]:
    """Yields (frame, is_fake, filename) for each video that has both a
    metadata entry and a readable frame."""
    metadata, video_dir = load_metadata(dataset_dir)
    count = 0
    for filename, info in metadata.items():
        if limit is not None and count >= limit:
            return
        video_path = video_dir / filename
        if not video_path.exists():
            continue
        frame = extract_frame(video_path)
        if frame is None:
            continue
        is_fake = 1 if info.get("label", "").upper() == "FAKE" else 0
        count += 1
        yield frame, is_fake, filename

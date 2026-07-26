"""Download and load the LIAR fake-news dataset (political statements with
6-way truthfulness labels) for training the text-credibility head.

Original schema (tab-separated, no header, 14 columns):
id, label, statement, subject, speaker, speaker_job, state, party,
barely_true_counts, false_counts, half_true_counts, mostly_true_counts,
pants_on_fire_counts, context
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_HANDLE = "doanquanvietnamca/liar-dataset"

COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
    "context",
]

# Higher risk = less truthful. Used both as a continuous target and,
# thresholded at 50, as the binary label the text head is trained on.
LABEL_RISK = {
    "pants-fire": 100,
    "false": 80,
    "barely-true": 60,
    "half-true": 40,
    "mostly-true": 20,
    "true": 0,
}


def download(handle: str = DEFAULT_KAGGLE_HANDLE, dest_dir: Optional[str] = None) -> Path:
    import kagglehub

    path = kagglehub.dataset_download(handle)
    logger.info("LIAR dataset available at %s", path)
    return Path(path)


def _read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, quoting=3)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(LABEL_RISK)]
    df["risk_score"] = df["label"].map(LABEL_RISK)
    df["is_misinformation"] = (df["risk_score"] >= 50).astype(int)
    return df.reset_index(drop=True)


def load(dataset_dir: Path, split: str = "train") -> pd.DataFrame:
    """Load one split ('train', 'test', or 'valid') as a DataFrame with
    columns: statement, label, risk_score, is_misinformation (+ metadata)."""
    dataset_dir = Path(dataset_dir)
    candidates = list(dataset_dir.rglob(f"{split}.tsv"))
    if not candidates:
        found = [p.name for p in dataset_dir.rglob("*.tsv")]
        raise FileNotFoundError(
            f"Could not find {split}.tsv under {dataset_dir}. Files present: {found}"
        )
    return _read_split(candidates[0])

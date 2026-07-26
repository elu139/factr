"""Train the text-credibility linear probe on the LIAR dataset.

    python -m factr.train.train_text_head --download
    python -m factr.train.train_text_head --data-dir /path/to/liar --limit 2000

Encodes each statement with the OpenCLIP text tower and fits logistic
regression against the binarized truthfulness label (see data/liar.py for
the label -> risk mapping), then saves the probe to models/text_head.joblib.
"""

import argparse
import logging
import sys

import numpy as np

from .. import config
from ..clip_scorer import get_scorer
from ..data import liar
from ..heads import ProbeHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def encode_statements(statements, log_every: int = 200) -> np.ndarray:
    scorer = get_scorer()
    embeddings = []
    for i, statement in enumerate(statements):
        embeddings.append(scorer.encode_text(statement))
        if (i + 1) % log_every == 0:
            logger.info("Encoded %d/%d statements", i + 1, len(statements))
    return np.vstack(embeddings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download LIAR via kagglehub first")
    parser.add_argument("--data-dir", type=str, default=None, help="Local LIAR dataset directory")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of training rows (for a quick run)")
    args = parser.parse_args()

    if args.download:
        data_dir = liar.download()
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        parser.error("Pass --download or --data-dir")

    train_df = liar.load(data_dir, "train")
    test_df = liar.load(data_dir, "test")

    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=42)

    logger.info("Train rows: %d, test rows: %d", len(train_df), len(test_df))

    train_embeddings = encode_statements(train_df["statement"].tolist())
    train_labels = train_df["is_misinformation"].to_numpy()

    head = ProbeHead()
    train_metrics = head.fit(train_embeddings, train_labels)
    logger.info("Train metrics: %s", train_metrics)

    test_embeddings = encode_statements(test_df["statement"].tolist())
    test_labels = test_df["is_misinformation"].to_numpy()
    test_metrics = head.evaluate(test_embeddings, test_labels)
    logger.info("Test metrics: %s", test_metrics)

    head.save(config.TEXT_HEAD_PATH)
    logger.info("Saved text head to %s", config.TEXT_HEAD_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

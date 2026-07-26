"""Train the visual-manipulation linear probe on DFDC video frames.

    python -m factr.train.train_visual_head --download
    python -m factr.train.train_visual_head --data-dir /path/to/dfdc --limit 400

Extracts one frame per video, encodes it with the OpenCLIP image tower, and
fits logistic regression against the REAL/FAKE label from DFDC's
metadata.json, then saves the probe to models/visual_head.joblib.
"""

import argparse
import logging
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from .. import config
from ..clip_scorer import get_scorer
from ..data import dfdc
from ..heads import ProbeHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download DFDC sample via kagglehub first")
    parser.add_argument("--data-dir", type=str, default=None, help="Local DFDC dataset directory")
    parser.add_argument("--limit", type=int, default=400, help="Max number of videos to process")
    args = parser.parse_args()

    if args.download:
        data_dir = dfdc.download()
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        parser.error("Pass --download or --data-dir")

    scorer = get_scorer()

    embeddings, labels = [], []
    for i, (frame, is_fake, filename) in enumerate(dfdc.iter_labeled_frames(data_dir, limit=args.limit)):
        embeddings.append(scorer.encode_image(frame))
        labels.append(is_fake)
        if (i + 1) % 50 == 0:
            logger.info("Processed %d videos (%s)", i + 1, filename)

    if len(embeddings) < 20:
        logger.error("Only found %d usable video/frame pairs; need more data to train a probe", len(embeddings))
        return 1

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    logger.info("Total videos: %d, fake ratio: %.2f", len(labels), labels.mean())

    train_emb, test_emb, train_labels, test_labels = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )

    head = ProbeHead()
    train_metrics = head.fit(train_emb, train_labels)
    logger.info("Train metrics: %s", train_metrics)

    test_metrics = head.evaluate(test_emb, test_labels)
    logger.info("Test metrics: %s", test_metrics)

    head.save(config.VISUAL_HEAD_PATH)
    logger.info("Saved visual head to %s", config.VISUAL_HEAD_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

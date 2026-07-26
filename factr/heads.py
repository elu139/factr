"""Linear-probe heads trained on frozen OpenCLIP embeddings.

Both heads are plain logistic regression on top of L2-normalized CLIP
embeddings -- the standard, cheap way to get a supervised signal out of a
frozen foundation model without fine-tuning it. `visual_head` is trained on
DFDC video frames (real=0 / fake=1); `text_head` is trained on LIAR
statements (truthful=0 / misinformation=1).
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


class ProbeHead:
    def __init__(self, classifier: Optional[LogisticRegression] = None):
        self.classifier = classifier

    @property
    def is_trained(self) -> bool:
        return self.classifier is not None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.classifier.fit(embeddings, labels)
        preds = self.classifier.predict(embeddings)
        probs = self.classifier.predict_proba(embeddings)[:, 1]
        return {
            "train_accuracy": accuracy_score(labels, preds),
            "train_auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        }

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        probs = self.classifier.predict_proba(embeddings)[:, 1]
        preds = self.classifier.predict(embeddings)
        return {
            "accuracy": accuracy_score(labels, preds),
            "auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        }

    def risk_score(self, embedding: np.ndarray) -> float:
        """Probability the input is misinformation/manipulated, scaled to 0-100."""
        if not self.is_trained:
            raise RuntimeError("Head has not been trained/loaded")
        prob = self.classifier.predict_proba(embedding.reshape(1, -1))[0, 1]
        return float(prob * 100.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, path)

    @classmethod
    def load(cls, path: Path) -> Optional["ProbeHead"]:
        if not Path(path).exists():
            return None
        return cls(joblib.load(path))

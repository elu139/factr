"""Weighted ensemble of OpenCLIP signals + LLaVA cloud reasoning.

Four independent signals, each producing a 0-100 "misinformation risk" score:

  clip_consistency  - zero-shot cosine similarity between image and caption
                       (no training needed)
  clip_visual_head  - logistic-regression probe on CLIP image embeddings,
                       trained on DFDC real/fake video frames
  clip_text_head    - logistic-regression probe on CLIP text embeddings,
                       trained on LIAR truthfulness labels
  llava             - Replicate-hosted LLaVA rating image/caption consistency
                       and image authenticity via natural-language reasoning

Any signal that isn't available (head not trained yet, no Replicate token)
is dropped and the remaining weights are renormalized, so the ensemble
degrades gracefully rather than failing.
"""

import logging
from typing import Any, Dict, Optional

from PIL import Image

from . import config, llava_client
from .clip_scorer import get_scorer
from .heads import ProbeHead

logger = logging.getLogger(__name__)


class MisinformationEnsemble:
    def __init__(self):
        self.scorer = get_scorer()
        self.visual_head = ProbeHead.load(config.VISUAL_HEAD_PATH)
        self.text_head = ProbeHead.load(config.TEXT_HEAD_PATH)
        self.weights = config.load_ensemble_weights()

    async def analyze(
        self,
        image: Image.Image,
        caption: str,
        image_url: Optional[str] = None,
        use_llava: bool = True,
    ) -> Dict[str, Any]:
        image_emb = self.scorer.encode_image(image)
        text_emb = self.scorer.encode_text(caption)

        signals: Dict[str, Dict[str, Any]] = {}

        cosine = float((image_emb * text_emb).sum())
        consistency = (cosine + 1.0) / 2.0
        signals["clip_consistency"] = {
            "risk_score": (1.0 - consistency) * 100.0,
            "confidence": 0.9,
            "detail": {"cosine_similarity": cosine},
        }

        if self.visual_head and self.visual_head.is_trained:
            signals["clip_visual_head"] = {
                "risk_score": self.visual_head.risk_score(image_emb),
                "confidence": 0.8,
            }
        else:
            logger.info("visual_head not trained yet (run train_visual_head.py); skipping")

        if self.text_head and self.text_head.is_trained:
            signals["clip_text_head"] = {
                "risk_score": self.text_head.risk_score(text_emb),
                "confidence": 0.8,
            }
        else:
            logger.info("text_head not trained yet (run train_text_head.py); skipping")

        if use_llava:
            llava_result = await llava_client.analyze(image, caption, image_url=image_url)
            if llava_result["available"]:
                signals["llava"] = llava_result

        return self._fuse(signals)

    def _fuse(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        active_weights = {name: self.weights.get(name, 0.0) for name in signals}
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            # Nothing configured with weight; fall back to equal weighting.
            active_weights = {name: 1.0 for name in signals}
            total_weight = float(len(signals))
        normalized_weights = {name: w / total_weight for name, w in active_weights.items()}

        final_score = sum(
            signals[name]["risk_score"] * normalized_weights[name] for name in signals
        )

        return {
            "misinformation_score": round(final_score, 1),
            "risk_level": self._risk_level(final_score),
            "confidence_level": self._confidence_level(signals),
            "signal_scores": {name: round(s["risk_score"], 1) for name, s in signals.items()},
            "signal_weights": {name: round(w, 3) for name, w in normalized_weights.items()},
            "signal_details": signals,
        }

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 30:
            return "low"
        if score < 70:
            return "medium"
        return "high"

    @staticmethod
    def _confidence_level(signals: Dict[str, Dict[str, Any]]) -> str:
        if not signals:
            return "none"
        avg_conf = sum(s.get("confidence", 0.5) for s in signals.values()) / len(signals)
        if avg_conf > 0.75:
            return "high"
        if avg_conf > 0.5:
            return "medium"
        return "low"

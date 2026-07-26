"""Thin wrapper around an OpenCLIP checkpoint: shared embeddings used for
zero-shot image-text consistency scoring and as features for the trained
linear-probe heads (visual manipulation / text credibility)."""

import io
import threading
from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

from . import config

_lock = threading.Lock()


class OpenClipScorer:
    """Lazily loads one OpenCLIP model/tokenizer and reuses it for every call."""

    def __init__(self, model_name: str = config.OPEN_CLIP_MODEL, pretrained: str = config.OPEN_CLIP_PRETRAINED):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text[:512]]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def consistency_score(self, image: Image.Image, text: str) -> float:
        """Cosine similarity between image and text embeddings, in [0, 1]."""
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(text)
        cosine = float(np.dot(image_emb, text_emb))
        return (cosine + 1.0) / 2.0

    @staticmethod
    def load_image_bytes(data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")


@lru_cache(maxsize=1)
def get_scorer() -> OpenClipScorer:
    with _lock:
        return OpenClipScorer()

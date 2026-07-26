"""LLaVA reasoning via Replicate's hosted API.

A 13B vision-language model has no business running on a free-tier box, so
instead of loading it locally we call it as a cloud endpoint. This is the
one signal in the ensemble that reasons about the image and caption
together in natural language rather than via embeddings.
"""

import asyncio
import base64
import io
import logging
import re
from typing import Optional

from PIL import Image

from . import config

logger = logging.getLogger(__name__)

PROMPT = (
    "You are checking an Instagram-style post for misinformation. "
    "Caption: \"{caption}\"\n"
    "Look at the image and answer two questions, each as a 0-100 score:\n"
    "1) CONSISTENCY: how well does the image actually match what the caption claims?\n"
    "2) AUTHENTICITY: how likely is the image to be unedited/real, as opposed to "
    "AI-generated, staged, or digitally manipulated?\n"
    "Reply with exactly this format and nothing else: "
    "CONSISTENCY=<0-100> AUTHENTICITY=<0-100> REASON=<one short sentence>"
)

_SCORE_RE = re.compile(r"CONSISTENCY\s*=\s*(\d+(?:\.\d+)?).*?AUTHENTICITY\s*=\s*(\d+(?:\.\d+)?)", re.S | re.I)
_REASON_RE = re.compile(r"REASON\s*=\s*(.+)", re.S | re.I)


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _fallback_result(reason: str) -> dict:
    return {
        "model": "llava",
        "available": False,
        "risk_score": 50.0,
        "confidence": 0.0,
        "reasoning": reason,
    }


def _call_replicate(image_ref: str, caption: str) -> str:
    import replicate

    output = replicate.run(
        config.REPLICATE_LLAVA_MODEL,
        input={"image": image_ref, "prompt": PROMPT.format(caption=caption)},
    )
    if isinstance(output, (list, tuple)):
        return "".join(str(chunk) for chunk in output)
    return str(output)


async def analyze(image: Image.Image, caption: str, image_url: Optional[str] = None) -> dict:
    """Ask LLaVA to rate image/caption consistency and image authenticity.

    Returns a risk_score in [0, 100] (higher = more likely misinformation),
    derived from (100 - average(consistency, authenticity)).
    """
    if not config.REPLICATE_API_TOKEN:
        return _fallback_result("REPLICATE_API_TOKEN not set; LLaVA signal skipped")

    image_ref = image_url or _image_to_data_uri(image)

    try:
        raw_text = await asyncio.to_thread(_call_replicate, image_ref, caption)
    except Exception as exc:  # network/model errors shouldn't take down the ensemble
        logger.warning("LLaVA call failed: %s", exc)
        return _fallback_result(f"LLaVA call failed: {exc}")

    match = _SCORE_RE.search(raw_text)
    if not match:
        logger.warning("Could not parse LLaVA response: %r", raw_text)
        return _fallback_result("Could not parse LLaVA response")

    consistency = min(100.0, max(0.0, float(match.group(1))))
    authenticity = min(100.0, max(0.0, float(match.group(2))))
    reason_match = _REASON_RE.search(raw_text)
    reason = reason_match.group(1).strip() if reason_match else raw_text.strip()

    risk_score = 100.0 - ((consistency + authenticity) / 2.0)

    return {
        "model": "llava",
        "available": True,
        "risk_score": risk_score,
        "consistency_score": consistency,
        "authenticity_score": authenticity,
        "confidence": 0.75,
        "reasoning": reason,
        "raw_response": raw_text.strip(),
    }

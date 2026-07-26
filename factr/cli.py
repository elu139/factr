"""Command-line entry point for one-off analysis, no server required.

    python -m factr.cli --image-url https://example.com/photo.jpg --caption "..."
    python -m factr.cli --image-path ./photo.jpg --caption "..." --no-llava
"""

import argparse
import asyncio
import json
import sys

import httpx
from PIL import Image

from .clip_scorer import OpenClipScorer
from .ensemble import MisinformationEnsemble


async def run(args: argparse.Namespace) -> dict:
    if args.image_url:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(args.image_url)
            response.raise_for_status()
        image = OpenClipScorer.load_image_bytes(response.content)
    else:
        image = Image.open(args.image_path)

    ensemble = MisinformationEnsemble()
    return await ensemble.analyze(
        image, args.caption, image_url=args.image_url, use_llava=not args.no_llava
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-url", type=str, help="URL of the image to analyze")
    source.add_argument("--image-path", type=str, help="Local path to the image to analyze")
    parser.add_argument("--caption", type=str, required=True, help="Post caption / claimed text")
    parser.add_argument("--no-llava", action="store_true", help="Skip the LLaVA cloud call")
    args = parser.parse_args()

    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

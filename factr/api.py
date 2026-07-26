"""FastAPI service exposing the OpenCLIP + LLaVA misinformation ensemble.

    uvicorn factr.api:app --reload
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from .clip_scorer import OpenClipScorer
from .ensemble import MisinformationEnsemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ensemble models...")
    state["ensemble"] = MisinformationEnsemble()
    logger.info("Ready.")
    yield
    state.clear()


app = FastAPI(title="factr.ai", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    image_url: HttpUrl
    caption: str
    use_llava: bool = True


@app.get("/health")
async def health():
    ensemble: MisinformationEnsemble = state.get("ensemble")
    return {
        "status": "ok" if ensemble else "loading",
        "visual_head_trained": bool(ensemble and ensemble.visual_head and ensemble.visual_head.is_trained),
        "text_head_trained": bool(ensemble and ensemble.text_head and ensemble.text_head.is_trained),
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    ensemble: MisinformationEnsemble = state["ensemble"]

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            response = await client.get(str(request.image_url))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=400, detail=f"Could not fetch image_url: {exc}")

    image = OpenClipScorer.load_image_bytes(response.content)
    return await ensemble.analyze(
        image, request.caption, image_url=str(request.image_url), use_llava=request.use_llava
    )


@app.post("/analyze/upload")
async def analyze_upload(caption: str = Form(...), use_llava: bool = Form(True), file: UploadFile = File(...)):
    ensemble: MisinformationEnsemble = state["ensemble"]
    data = await file.read()
    try:
        image = OpenClipScorer.load_image_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}")

    return await ensemble.analyze(image, caption, use_llava=use_llava)

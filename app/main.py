"""
Honest Lister web app (FastAPI).

Endpoints:
  GET  /                 -> the single-page demo UI
  POST /api/appraise     -> image (+ optional seller fields, demo scenario) -> grounded listing
  GET  /api/demo         -> run a built-in mock scenario (no upload needed)
  GET  /api/benchmark    -> the Honesty Benchmark results (for the in-app chart)
  GET  /api/health       -> liveness

Open-weight model backend is selected by HONEST_LISTER_VLM (default: mock),
so this runs end-to-end with zero model download for the demo/CI path.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from lister import __version__
from lister.benchmark import run as run_benchmark
from lister.grounding import Source, build, publishability
from lister.pricing import recommend
from lister.vlm import appraise

STATIC = Path(__file__).resolve().parent.parent / "static"
app = FastAPI(title="Honest Lister", version=__version__)


def _result_dict(image_bytes: bytes, hint: str, seller: dict[str, str]) -> dict:
    vlm = appraise(image_bytes, hint)

    # Fold any seller-confirmed fields in as authoritative SELLER claims.
    claims = list(vlm.claims)
    for attr, value in seller.items():
        if value:
            from lister.grounding import Claim
            claims.append(Claim(attr=attr, value=value, confidence=1.0,
                                evidence="entered by seller", source=Source.SELLER))

    listing = build(claims, vlm.tags, vlm.defects, seller_values=seller)
    gate = publishability(listing, seller_values=seller)

    # Price from VERIFIED attributes only.
    verified_attrs = {c.attr: c.value for c in listing.claims if c.stateable}
    price = recommend(verified_attrs)

    return {
        "title": listing.title,
        "description": listing.description,
        "tags": listing.tags,
        "honesty_score": listing.honesty_score,
        "claims": [asdict(c) for c in listing.claims],
        "defects": [asdict(d) for d in listing.defects],
        "unreadable": vlm.unreadable,
        "publishability": asdict(gate),
        "price": asdict(price),
        "backend_error": vlm.backend_error,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


@app.get("/api/demo")
def demo(scenario: str = "jacket_no_label") -> JSONResponse:
    return JSONResponse(_result_dict(b"", scenario, {}))


@app.post("/api/appraise")
async def api_appraise(
    image: UploadFile | None = File(default=None),
    scenario: str = Form(default=""),
    brand: str = Form(default=""),
    size: str = Form(default=""),
    material: str = Form(default=""),
) -> JSONResponse:
    data = await image.read() if image is not None else b""
    seller = {k: v.strip() for k, v in
              {"brand": brand, "size": size, "material": material}.items() if v.strip()}
    return JSONResponse(_result_dict(data, scenario, seller))


@app.get("/api/benchmark")
def benchmark() -> JSONResponse:
    return JSONResponse(run_benchmark())

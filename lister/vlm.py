"""
Vision backend: look at the item photo, return CLAIMS and DEFECTS (not prose).

The model is asked to behave like a cautious appraiser: for each attribute give a
value, a confidence 0-1, and the visual evidence for it -- or abstain. It also
reports visible defects and unreadable/missing areas. We never ask it for
marketing copy; grounding.py turns claims into copy.

Three interchangeable, all open-weight backends so the app runs end-to-end today
and swaps to a real model with one env var:

    HONEST_LISTER_VLM=mock          deterministic, offline (default in dev/tests)
    HONEST_LISTER_VLM=hf_space      an open-weight VLM on a HF Space (demo)
    HONEST_LISTER_VLM=transformers  moondream2 / Qwen2-VL running locally

No closed API is ever in the path -- that is the "why open innovation matters"
argument: a reseller doing hundreds of listings pays ~0 per listing locally.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .grounding import Claim, Defect, Source

APPRAISER_PROMPT = """You are a cautious product appraiser looking at ONE photo of a
second-hand clothing item for resale. Report ONLY what the PHOTO shows. If the photo
does not clearly show an attribute, set its value to "unknown". Never guess a brand,
size, or fabric you cannot actually see. Also report any visible defects and any area
that is unreadable or missing.

Return STRICT JSON:
{
  "category":  {"value":"...","confidence":0.0-1.0,"evidence":"what shows this"},
  "brand":     {"value":"...","confidence":0.0-1.0,"evidence":"..."},
  "color":     {"value":"...","confidence":0.0-1.0,"evidence":"..."},
  "material":  {"value":"...","confidence":0.0-1.0,"evidence":"..."},
  "size":      {"value":"...","confidence":0.0-1.0,"evidence":"..."},
  "condition": {"value":"...","confidence":0.0-1.0,"evidence":"..."},
  "defects":   [{"note":"...","confidence":0.0-1.0,"evidence":"..."}],
  "unreadable":["size label","care label"],
  "tags":      ["lowercase","search","tags"]
}
Confidence is how sure you are from the PHOTO ALONE. Evidence must cite the image."""

ATTRS = ["category", "brand", "color", "material", "size", "condition"]


@dataclass
class VlmResult:
    claims: list[Claim]
    tags: list[str]
    defects: list[Defect] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)  # best-guess, for pricing
    backend_error: str = ""


def _parse(payload: dict) -> VlmResult:
    claims, attrs = [], {}
    for attr in ATTRS:
        node = payload.get(attr) or {}
        value = str(node.get("value", "unknown"))
        claims.append(Claim(
            attr=attr, value=value,
            confidence=float(node.get("confidence", 0.0) or 0.0),
            evidence=str(node.get("evidence", "")), source=Source.IMAGE,
        ))
        attrs[attr] = value
    defects = [Defect(note=str(d.get("note", "")),
                      confidence=float(d.get("confidence", 0.0) or 0.0),
                      evidence=str(d.get("evidence", "")))
               for d in (payload.get("defects") or [])]
    return VlmResult(
        claims=claims, tags=list(payload.get("tags", []) or []),
        defects=defects, unreadable=list(payload.get("unreadable", []) or []),
        attrs=attrs, backend_error=str(payload.get("_backend_error", "")),
    )


# --- mock scenarios (deterministic, exercise every status path) --------------

_SCENARIOS = {
    # Default: sneakers. brand is a weak guess (-> UNVERIFIED), size not visible.
    "sneakers": {
        "category": {"value": "sneakers", "confidence": 0.94,
                     "evidence": "lace-up athletic shoe with rubber sole"},
        "color":    {"value": "white/green", "confidence": 0.91,
                     "evidence": "white upper with a green side panel"},
        "condition":{"value": "used - good", "confidence": 0.86,
                     "evidence": "light toe creasing, clean sole"},
        "brand":    {"value": "Nike", "confidence": 0.62, "evidence": ""},
        "size":     {"value": "unknown", "confidence": 0.0,
                     "evidence": "no size tag visible"},
        "material": {"value": "unknown", "confidence": 0.3,
                     "evidence": "cannot tell leather vs synthetic from this angle"},
        "defects":  [{"note": "light creasing on the toe box", "confidence": 0.8,
                      "evidence": "visible fold lines across the toe"}],
        "unreadable": ["size label"],
        "tags": ["sneakers", "shoes", "white", "green", "used"],
    },
    # The "wow" demo: a jacket with NO visible brand or size label.
    "jacket_no_label": {
        "category": {"value": "jacket", "confidence": 0.93,
                     "evidence": "collared outerwear with full-length zip"},
        "color":    {"value": "navy", "confidence": 0.9,
                     "evidence": "dark blue body under even lighting"},
        "condition":{"value": "used - very good", "confidence": 0.82,
                     "evidence": "no visible wear on cuffs or collar"},
        "brand":    {"value": "unknown", "confidence": 0.0,
                     "evidence": "no logo or brand tag visible in the photo"},
        "size":     {"value": "unknown", "confidence": 0.0,
                     "evidence": "no size label visible"},
        "material": {"value": "unknown", "confidence": 0.35,
                     "evidence": "looks like a cotton blend but no care label shown"},
        "defects":  [],
        "unreadable": ["brand tag", "size label", "care label"],
        "tags": ["jacket", "navy", "outerwear", "used"],
    },
}


def _mock(image_bytes: bytes, hint: str = "") -> dict:
    return dict(_SCENARIOS.get(hint or "sneakers", _SCENARIOS["sneakers"]))


def _hf_space(image_bytes: bytes, hint: str = "") -> dict:
    from gradio_client import Client, handle_file
    import tempfile
    space = os.environ["HONEST_LISTER_SPACE"]
    token = os.environ.get("HF_TOKEN")
    client = Client(space, hf_token=token) if token else Client(space)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        path = f.name
    out = client.predict(handle_file(path), APPRAISER_PROMPT, api_name="/appraise")
    return out if isinstance(out, dict) else json.loads(out)


# --- real open-weight VLM (moondream2), cached and queried per-attribute -----

_MODEL_CACHE: dict = {}

# Targeted questions. Each asks for "VALUE because EVIDENCE", and to answer
# exactly "unknown" when the photo doesn't show it -- so abstention is a
# first-class answer, not a failure.
_QUESTIONS = {
    "category": "What type of clothing or accessory is this? Answer in the form 'VALUE because EVIDENCE'.",
    "brand": "What brand is this item? Look only for a visible logo or brand tag. If none is clearly visible, answer exactly 'unknown'. Otherwise answer 'BRAND because EVIDENCE'.",
    "color": "What is the main color of this item? Answer 'COLOR because EVIDENCE'.",
    "material": "What material is this item made of? Only answer if a care label or obvious texture shows it; otherwise answer exactly 'unknown'. Answer 'MATERIAL because EVIDENCE'.",
    "size": "What size is this item? Only answer if a size label is visible in the photo; otherwise answer exactly 'unknown'. Answer 'SIZE because EVIDENCE'.",
    "condition": "What is the condition of this item (like new, used - very good, used - good, used - fair)? Answer 'CONDITION because EVIDENCE'.",
}

_HEDGE = ("maybe", "possibly", "appears", "looks like", "could be", "might be",
          "probably", "seems", "hard to tell", "not sure")
_ABSTAIN = ("unknown", "cannot tell", "can't tell", "not visible", "no ", "none",
            "unclear", "not clear")


def _load_moondream():
    if "m" not in _MODEL_CACHE:
        import torch
        device = os.environ.get("HONEST_LISTER_DEVICE", "cpu")
        # moondream's remote code auto-selects MPS on Macs and then mixes it with
        # CPU tensors ("Passed CPU tensor to MPS op"). On the CPU path we hide MPS
        # from torch BEFORE the model loads so everything stays on one device.
        if device == "cpu":
            def _mps_false():
                return False
            # keep attributes torch/transformers may read off the original
            for name in ("is_available", "is_built"):
                orig = getattr(torch.backends.mps, name, None)
                if orig is not None:
                    _mps_false.__wrapped__ = getattr(orig, "__wrapped__", orig)
                    setattr(torch.backends.mps, name, _mps_false)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = os.environ.get("HONEST_LISTER_MODEL", "vikhyatk/moondream2")
        rev = os.environ.get("HONEST_LISTER_MODEL_REV", "2025-01-09")  # pinned for reproducibility
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev, trust_remote_code=True,
            torch_dtype=torch.float32)
        model = model.to(device).eval()
        _MODEL_CACHE["m"] = model
        _MODEL_CACHE["t"] = AutoTokenizer.from_pretrained(model_id, revision=rev)
    return _MODEL_CACHE["m"], _MODEL_CACHE["t"]


def _ask(model, tok, img, question: str) -> str:
    # Newer moondream exposes .query(); older uses encode_image + answer_question.
    try:
        return str(model.query(img, question)["answer"]).strip()
    except (AttributeError, TypeError, KeyError):
        enc = model.encode_image(img)
        return str(model.answer_question(enc, question, tok)).strip()


def _to_node(attr: str, answer: str) -> dict:
    """Map a free-text answer to {value, confidence, evidence} honestly."""
    a = answer.strip()
    low = a.lower()
    if any(low.startswith(t) or f" {t}" in f" {low}" for t in ("unknown",)) or low in {"no", "none"}:
        return {"value": "unknown", "confidence": 0.0, "evidence": a}
    # split "VALUE because EVIDENCE"
    value, evidence = a, ""
    for sep in (" because ", " - ", ": "):
        if sep in a:
            value, evidence = a.split(sep, 1)
            break
    value = value.strip().strip(".").strip()
    if any(t in low for t in _ABSTAIN) and attr in ("brand", "size", "material"):
        return {"value": "unknown", "confidence": 0.0, "evidence": a}
    conf = 0.6 if any(h in low for h in _HEDGE) else 0.92
    return {"value": value, "confidence": conf, "evidence": evidence.strip() or value}


def _transformers(image_bytes: bytes, hint: str = "") -> dict:
    import io
    from PIL import Image
    model, tok = _load_moondream()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    payload: dict = {}
    for attr, q in _QUESTIONS.items():
        payload[attr] = _to_node(attr, _ask(model, tok, img, q))

    defects_ans = _ask(model, tok, img,
                       "Are there any visible defects, stains, holes, fading or wear? "
                       "Describe them briefly, or answer exactly 'none'.")
    defects = []
    if defects_ans.strip().lower() not in {"none", "no", "no defects", ""}:
        defects = [{"note": defects_ans.strip(), "confidence": 0.7, "evidence": "described from the photo"}]

    unreadable = [a for a in ("brand", "size", "material")
                  if payload[a]["value"] == "unknown"]
    tags = [payload[k]["value"] for k in ("category", "color", "brand")
            if payload[k]["value"] != "unknown"]
    payload["defects"] = defects
    payload["unreadable"] = unreadable
    payload["tags"] = [t.lower() for t in tags]
    return payload


_BACKENDS = {"mock": _mock, "hf_space": _hf_space, "transformers": _transformers}


def appraise(image_bytes: bytes, hint: str = "") -> VlmResult:
    backend = os.environ.get("HONEST_LISTER_VLM", "mock")
    fn = _BACKENDS.get(backend, _mock)
    try:
        payload = fn(image_bytes, hint)
    except Exception as exc:  # never let a model hiccup fabricate a listing
        payload = _mock(image_bytes, hint)
        payload["_backend_error"] = str(exc)
    return _parse(payload)

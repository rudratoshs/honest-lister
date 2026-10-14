"""
A baseline listing generator -- what an ordinary 'AI listing writer' does.

It states every non-empty attribute the vision model produced as fact, with no
abstention and no evidence check. This is the honest control we measure Honest
Lister against. It is NOT a claim about any specific competitor; it is the
default behaviour you get when you ask a model to 'write a listing' and publish
what it returns.
"""

from __future__ import annotations

from .vlm import VlmResult

_EMPTY = {"", "unknown", "unclear", "n/a", "none", "not visible"}


def generate(vlm: VlmResult) -> dict:
    """Return the attributes this baseline would STATE as fact in the listing."""
    stated = {
        c.attr: c.value
        for c in vlm.claims
        if (c.value or "").strip().lower() not in _EMPTY
    }
    order = ["brand", "category", "color", "material", "size", "condition"]
    title = " ".join(stated[a] for a in order if a in stated)
    desc = " ".join(f"{a.capitalize()}: {v}." for a, v in stated.items())
    return {"title": title, "description": desc, "stated": stated, "tags": vlm.tags}

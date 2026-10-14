"""Tests for the honesty core: the ledger must never state a guess as fact."""

from lister.grounding import (Claim, Source, Status, build, classify,
                              publishability)
from lister.vlm import _parse


def _c(attr, value, conf, ev):
    return Claim(attr=attr, value=value, confidence=conf, evidence=ev)


def test_guess_without_evidence_is_unverified():
    c = classify(_c("brand", "Nike", 0.62, ""))
    assert c.status is Status.UNVERIFIED


def test_high_stakes_needs_high_confidence():
    # brand visible but only 0.78 -> below the 0.90 high-stakes bar
    c = classify(_c("brand", "The North Face", 0.78, "small logo on chest"))
    assert c.status is Status.UNVERIFIED


def test_observed_requires_confidence_and_evidence():
    c = classify(_c("color", "navy", 0.9, "dark blue body"))
    assert c.status is Status.OBSERVED


def test_inferred_middle_band_for_ordinary_attr():
    c = classify(_c("color", "teal", 0.6, "bluish-green hue"))
    assert c.status is Status.INFERRED


def test_abstain_passes_through_as_unverified():
    c = classify(_c("size", "unknown", 0.0, "no size tag visible"))
    assert c.status is Status.UNVERIFIED
    assert "photo" in c.reason


def test_seller_photo_conflict_is_contradicted():
    c = _c("brand", "Adidas", 0.95, "three stripes clearly visible")
    out = classify(c, seller_values={"brand": "Nike"})
    assert out.status is Status.CONTRADICTED


def test_unverified_high_stakes_never_enters_title():
    vlm = _parse({
        "category": {"value": "sneakers", "confidence": 0.95, "evidence": "lace-up shoe"},
        "brand": {"value": "Nike", "confidence": 0.6, "evidence": ""},
        "color": {"value": "white", "confidence": 0.9, "evidence": "white upper"},
        "size": {"value": "unknown", "confidence": 0.0, "evidence": "no tag"},
        "material": {"value": "unknown", "confidence": 0.0, "evidence": ""},
        "condition": {"value": "used - good", "confidence": 0.85, "evidence": "light wear"},
        "tags": [],
    })
    listing = build(vlm.claims, vlm.tags, vlm.defects)
    assert "Nike" not in listing.title              # the unverified brand is kept out
    assert "sneakers" in listing.title.lower()


def test_contradiction_blocks_publish():
    claims = [_c("category", "dress", 0.95, "A-line garment"),
              _c("condition", "like new", 0.9, "no wear"),
              _c("brand", "Zara", 0.95, "ZARA tag readable")]
    listing = build(claims, [], seller_values={"brand": "Mango"})
    gate = publishability(listing, seller_values={"brand": "Mango"})
    assert not gate.ok
    assert any("contradiction" in b for b in gate.blockers)

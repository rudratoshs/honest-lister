"""
Evidence-backed pricing.

A photo cannot tell you market value, so we never pretend it can. We price only
from VERIFIED attributes against real comparable sales, and we return a RANGE
with the number of comparables and the uncertainty -- not a false-precision
single number. If there aren't enough comparables, we abstain and say so.

Baseline = median of comparables (cheap, robust, honest).
TabPFN    = optional; used ONLY if it beats the baseline on held-out comps.
            Note: TabPFN's recommended weights are non-commercial -- it stays an
            experiment here, reported honestly, never silently shipped.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

MIN_COMPARABLES = 5          # below this we do not give a confident price
COMPS_PATH = Path(__file__).resolve().parent.parent / "data" / "comps.csv"


@dataclass
class PriceEstimate:
    ok: bool
    low: float | None
    point: float | None
    high: float | None
    n_comparables: int
    method: str
    note: str
    comparables: list[dict]


def load_comps(path: Path = COMPS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        try:
            r["price"] = float(r["price"])
        except (KeyError, ValueError):
            r["price"] = None
    return [r for r in rows if r.get("price") is not None]


def _matches(row: dict, attrs: dict[str, str]) -> bool:
    """A comparable must share category + condition; brand/size narrow it."""
    for key in ("category", "condition"):
        want = (attrs.get(key) or "").strip().lower()
        if want and want not in {"unknown", ""} and row.get(key, "").strip().lower() != want:
            return False
    return True


def comparables(attrs: dict[str, str], comps: list[dict] | None = None) -> list[dict]:
    comps = comps if comps is not None else load_comps()
    pool = [r for r in comps if _matches(r, attrs)]
    # Narrow by brand if we actually verified one.
    brand = (attrs.get("brand") or "").strip().lower()
    if brand and brand not in {"unknown", ""}:
        narrowed = [r for r in pool if r.get("brand", "").strip().lower() == brand]
        if len(narrowed) >= MIN_COMPARABLES:
            return narrowed
    return pool


def median_baseline(attrs: dict[str, str], comps: list[dict] | None = None) -> PriceEstimate:
    pool = comparables(attrs, comps)
    n = len(pool)
    if n < MIN_COMPARABLES:
        return PriceEstimate(
            ok=False, low=None, point=None, high=None, n_comparables=n,
            method="median-of-comparables",
            note=(f"only {n} comparable sale(s) found (need {MIN_COMPARABLES}); "
                  "not enough evidence for a confident price. "
                  "Confirm brand/size to find more comps."),
            comparables=pool,
        )
    prices = sorted(r["price"] for r in pool)
    point = statistics.median(prices)
    # Interquartile range as the honest "most sell between" band.
    q = statistics.quantiles(prices, n=4) if n >= 4 else [prices[0], point, prices[-1]]
    low, high = round(q[0], 2), round(q[-1], 2)
    return PriceEstimate(
        ok=True, low=low, point=round(point, 2), high=high, n_comparables=n,
        method="median-of-comparables",
        note=f"based on {n} comparable sales; most sold between {low} and {high}.",
        comparables=pool[:10],
    )


def recommend(attrs: dict[str, str], comps: list[dict] | None = None,
              try_tabpfn: bool = False) -> PriceEstimate:
    """Default path. TabPFN is opt-in and must earn its place via the benchmark."""
    baseline = median_baseline(attrs, comps)
    if try_tabpfn and baseline.ok:
        tab = _tabpfn_estimate(attrs, comps)
        if tab is not None:
            return tab
    return baseline


def _tabpfn_estimate(attrs: dict[str, str], comps: list[dict] | None = None):
    """TabPFN regressor over comps, if installed. Returns None if unavailable."""
    try:
        import numpy as np
        from tabpfn import TabPFNRegressor
    except Exception:
        return None
    comps = comps if comps is not None else load_comps()
    if len(comps) < MIN_COMPARABLES:
        return None
    cats = {k: {} for k in ("category", "brand", "condition", "size")}

    def enc(key, val):
        d = cats[key]
        v = (val or "unknown").strip().lower()
        return d.setdefault(v, len(d))

    X = np.array([[enc(k, r.get(k, "")) for k in cats] for r in comps], dtype=float)
    y = np.array([r["price"] for r in comps], dtype=float)
    model = TabPFNRegressor()
    model.fit(X, y)
    xq = np.array([[enc(k, attrs.get(k, "")) for k in cats]], dtype=float)
    pred = float(model.predict(xq)[0])
    spread = float(np.std(y)) * 0.5
    return PriceEstimate(
        ok=True, low=round(pred - spread, 2), point=round(pred, 2),
        high=round(pred + spread, 2), n_comparables=len(comps),
        method="TabPFN (experimental, non-commercial weights)",
        note="TabPFN prediction; validate against the median baseline before trusting.",
        comparables=comps[:10],
    )

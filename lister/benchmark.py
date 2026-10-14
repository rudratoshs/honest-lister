"""
The Honesty Benchmark.

Run the SAME vision-model output through two systems and measure the trade-off
that actually matters for a reseller:

  1. Unsupported claims / listing  -- facts stated that the photo cannot establish (LOWER better)
  2. Correct-abstention rate        -- of the unknowable attrs, how many were held back (HIGHER)
  3. Supported attrs retained       -- of the knowable attrs, how many were still stated (HIGHER)
  4. Human corrections / listing    -- claims a human must fix before publishing (LOWER better)

The goal is NOT maximum abstention (a system that says nothing is safe but
useless). It's the balance: fewer false claims while still writing a usable
listing. Results are preliminary evidence on a small annotated set, not proof of
universal reliability -- the sample size is printed with every run.

    python -m lister.benchmark            # pretty table
    python -m lister.benchmark --json     # machine-readable (for the article chart)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from . import baseline as baseline_gen
from .grounding import Status, build
from .vlm import ATTRS, VlmResult, _parse

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures.json"


@dataclass
class Metrics:
    system: str
    n_items: int
    unsupported_per_listing: float
    correct_abstention_rate: float
    supported_retained_rate: float
    corrections_per_listing: float


def _truth(fixture: dict) -> dict:
    return fixture["ground_truth"]


def _score(stated: dict[str, str], gt: dict) -> tuple[int, int, int, int, int, int]:
    """
    stated: attr -> value the system asserted as fact.
    Returns raw counts: (unsupported, knowable_total, knowable_stated,
                         unknowable_total, unknowable_abstained, corrections)
    """
    unsupported = knowable_total = knowable_stated = 0
    unknowable_total = unknowable_abstained = 0
    for attr in ATTRS:
        info = gt.get(attr, {"knowable": False})
        knowable = bool(info.get("knowable"))
        asserted = attr in stated
        if knowable:
            knowable_total += 1
            if asserted:
                knowable_stated += 1
        else:
            unknowable_total += 1
            if asserted:
                unsupported += 1          # stated a fact the photo can't establish
            else:
                unknowable_abstained += 1
    corrections = unsupported             # at minimum, every unsupported claim needs a human fix
    return (unsupported, knowable_total, knowable_stated,
            unknowable_total, unknowable_abstained, corrections)


def _honest_stated(vlm: VlmResult) -> dict[str, str]:
    listing = build(vlm.claims, vlm.tags, vlm.defects)
    return {c.attr: c.value for c in listing.claims
            if c.status in (Status.OBSERVED, Status.INFERRED)}


def run(fixtures_path: Path = FIXTURES) -> dict:
    data = json.loads(fixtures_path.read_text())
    items = data["items"]
    agg = {"baseline": [0, 0, 0, 0, 0, 0], "honest": [0, 0, 0, 0, 0, 0]}

    per_item = []
    for it in items:
        vlm = _parse(it["vlm"])
        gt = _truth(it)
        b_stated = baseline_gen.generate(vlm)["stated"]
        h_stated = _honest_stated(vlm)
        b = _score(b_stated, gt)
        h = _score(h_stated, gt)
        for i in range(6):
            agg["baseline"][i] += b[i]
            agg["honest"][i] += h[i]
        per_item.append({"id": it["id"],
                         "baseline_unsupported": b[0], "honest_unsupported": h[0]})

    n = len(items)

    def metrics(name, a) -> Metrics:
        unsupported, k_tot, k_stated, u_tot, u_abst, corr = a
        return Metrics(
            system=name, n_items=n,
            unsupported_per_listing=round(unsupported / n, 2),
            correct_abstention_rate=round(u_abst / u_tot, 2) if u_tot else 1.0,
            supported_retained_rate=round(k_stated / k_tot, 2) if k_tot else 1.0,
            corrections_per_listing=round(corr / n, 2),
        )

    return {
        "sample_size": n,
        "disclaimer": "Preliminary evidence on a small annotated set; not proof of universal reliability.",
        "baseline": asdict(metrics("baseline", agg["baseline"])),
        "honest": asdict(metrics("honest", agg["honest"])),
        "per_item": per_item,
    }


def _print(result: dict) -> None:
    b, h = result["baseline"], result["honest"]
    print(f"\n  Honesty Benchmark  (n = {result['sample_size']} annotated items)")
    print("  " + result["disclaimer"])
    print("  " + "-" * 66)
    print(f"  {'metric':<34}{'baseline':>14}{'honest':>14}")
    print("  " + "-" * 66)
    rows = [
        ("unsupported claims / listing  (lower)", "unsupported_per_listing"),
        ("correct-abstention rate       (higher)", "correct_abstention_rate"),
        ("supported attrs retained      (higher)", "supported_retained_rate"),
        ("human corrections / listing   (lower)", "corrections_per_listing"),
    ]
    for label, key in rows:
        print(f"  {label:<34}{b[key]:>14}{h[key]:>14}")
    print("  " + "-" * 66)
    drop = b["unsupported_per_listing"] - h["unsupported_per_listing"]
    print(f"  Honest Lister removes {drop:.2f} unsupported claim(s) per listing "
          f"while retaining {h['supported_retained_rate']*100:.0f}% of real attributes.\n")


if __name__ == "__main__":
    res = run()
    if "--json" in sys.argv:
        print(json.dumps(res, indent=2))
    else:
        _print(res)

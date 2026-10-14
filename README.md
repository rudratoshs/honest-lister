<div align="center">

# 🧾 Honest Lister

**The evidence-backed listing copilot for resellers.**
Every product claim is supported, every uncertainty is visible, and nothing a photo can't prove gets written as fact.

*"Most AI listing tools optimize what they can **generate**. Honest Lister optimizes what a seller can safely **claim**."*

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Open weights](https://img.shields.io/badge/AI-open--weight%20VLM-0d9488)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## The problem

Resellers on eBay / Vinted / Etsy lose money two ways: slow listing, and **inaccurate listings** — a wrong brand, size, or "like new" that wasn't — which get items pulled, refunded, or the seller banned.

Every "AI listing writer" optimizes the first problem and *worsens* the second. Ask a vision model to "write a listing" and it will confidently invent a brand it can't see, a fabric it's guessing, a size that was never in frame. The copy reads great. The claim is a liability.

## The idea

Honest Lister treats a listing as a set of **claims that must be defensible**, not prose to be generated. An open-weight vision model looks at the photo and, for each attribute, returns a value, a confidence, and the **visual evidence** behind it — or abstains. Then an evidence ledger decides what may be stated:

| Status | Meaning | Goes in the listing? |
|---|---|---|
| 🟢 **Observed** | Directly visible, above the bar | ✅ as fact |
| 🔵 **Inferred** | Plausible, not certain | ✅ hedged ("appears to be…") |
| 🟠 **Unverified** | A guess: low confidence or no visual evidence | ❌ held back, surfaced to confirm |
| 🔴 **Contradicted** | Conflicts with the photo or seller input | ⛔ blocks publishing |

High-stakes attributes (brand, size, material, authenticity) are held to a stricter bar and are **never** merely inferred.

## It's measured, not claimed — the Honesty Benchmark

Same vision-model output, two systems: a normal "state everything" generator vs Honest Lister, scored against hand-annotated ground truth (is each attribute actually establishable from the photo?).

```
  Honesty Benchmark  (n = 7 annotated items)   ·   preliminary evidence, small set
  ------------------------------------------------------------------
  metric                                  baseline        honest
  ------------------------------------------------------------------
  unsupported claims / listing  (lower)       1.14           0.0
  correct-abstention rate       (higher)      0.47           1.0
  supported attrs retained      (higher)      1.0            0.96
  human corrections / listing   (lower)       1.14           0.0
  ------------------------------------------------------------------
  Removes 1.14 unsupported claims/listing while keeping 96% of real attributes.
```

The 96% (not 100%) is the point: Honest Lister sometimes over-abstains on a real attribute rather than risk a false claim — and you recover it in one click by confirming. The goal isn't maximum abstention (a system that says nothing is safe but useless); it's **fewer false claims while the listing stays usable.**

```bash
python -m lister.benchmark          # the table above
python -m lister.benchmark --json   # machine-readable
```

> These 7 items are a reproducible seed set. The annotated set expands toward 30–50 real photos; results are preliminary evidence, not proof of universal reliability. Sample size is printed with every run.

## Pricing you can defend

A photo can't tell you market value, so we don't pretend. Price comes **only from verified attributes** against real comparable sales, returned as a **range + number of comparables + uncertainty**. Too few comps → it abstains and tells you which photo would unlock more.

TabPFN (open-source tabular foundation model) is wired in as an **experiment** behind the median-of-comparables baseline — used only if it measurably beats the baseline. Its recommended weights are non-commercial, so it stays a benchmarked option, reported honestly, never silently shipped into a product you'd sell.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload          # open http://127.0.0.1:8000
```

The demo runs on a deterministic **mock** backend (no model download) so the page never hangs. Point it at a real open-weight model with one env var:

```bash
HONEST_LISTER_VLM=transformers   # moondream2 / Qwen2-VL locally
HONEST_LISTER_VLM=hf_space       # an open-weight VLM on a HF Space (for the hosted demo)
```

## Architecture

```
  photo ─► open-weight VLM ─► CLAIMS (value, confidence, evidence) + defects
                                   │
                                   ▼
                        evidence ledger (grounding.py)
                        observed / inferred / unverified / contradicted
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
          listing copy     publishability gate   pricing (verified attrs
        (stateable only)  (blocks / warns / asks)  → comps range, or abstain)
```

No closed API is ever in the path — that's the "why open innovation matters" argument made concrete: a reseller doing hundreds of listings pays **~0 per listing** on local open-weight inference. That margin is the product.

## What this is / isn't

- **Is:** an honesty layer for commercial listing workflows, with a measurable benchmark.
- **Isn't:** a guarantee of truth. A vision model's confidence is not proof — evidence checks are validated against annotated ground truth, and an observed-looking logo does **not** establish authenticity.
- **Privacy:** on the local backend, images never leave the machine. On the hosted demo, images are sent to the configured open-weight inference Space — stated plainly, not described as "fully local."

## Roadmap

- Expand the benchmark to 30–50 real annotated secondhand-clothing photos.
- Region-level evidence (point to *where* in the image a claim came from).
- Marketplace export; direct posting only if the API/auth fits the trust model.

## License

[MIT](LICENSE).

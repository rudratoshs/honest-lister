---
title: "I built an AI that writes resale listings — then refuses to publish the ones it can't prove"
published: false
tags: devchallenge, hf26challenge, ai, opensource
cover_image: https://raw.githubusercontent.com/rudratoshs/devto-assets/main/honest-lister/cover.png
---

*My entry for the Hacktoberfest Open-Source AI Challenge, Week 2 (Side Hustle).*

A reseller I know got a pair of trainers pulled and refunded last month. Not because the shoes were bad — because her listing said "leather" and they were synthetic. She hadn't lied on purpose. She'd pasted whatever the AI listing tool wrote, and the AI had guessed.

That's the thing nobody says about "AI listing writers": they're optimized to make the item sound good, which is exactly the wrong objective. On a resale platform, a confident wrong word — the wrong brand, the wrong size, a "like new" that wasn't — doesn't just lose a sale. It gets your item removed, your money clawed back, and eventually your account banned.

So I built the opposite tool. It's called **Honest Lister**, and its whole job is to write the most *defensible* listing, not the most *sellable* one.

> **Most AI listing tools optimize what they can _generate_. Honest Lister optimizes what a seller can safely _claim_.**

> 🔴 **Try the live demo:** **[honest-lister.onrender.com](https://honest-lister.onrender.com)** · **Code:** [github.com/rudratoshs/honest-lister](https://github.com/rudratoshs/honest-lister)
> *(Free tier — if it's been idle it may take ~30s to wake, then click the "prove it or leave it out" example.)*

---

## What I built

Honest Lister looks at a photo of a second-hand clothing item and produces a listing where **every factual claim is backed by something actually visible in the photo** — and anything it can't verify is shown to you as "confirm manually," never silently invented.

It does four things:

1. **Reads the photo with an open-weight vision model** and turns it into *claims*, not prose — each with a confidence and the visual evidence behind it.
2. **Runs every claim through an evidence ledger** that decides what may be stated as fact, what gets hedged, what gets held back, and what *contradicts* other evidence.
3. **Prices the item from real comparable sales** — as a range with a sample size, or it abstains if there aren't enough comps.
4. **Gates publishing** so a listing with an unresolved high-risk claim (an unverified brand, a contradiction) can't go out by accident.

And — the part I care about most — **it ships with a benchmark that measures whether the honesty actually works**, instead of me just telling you it does.

---

## Demo

The scenario I'm proudest of is **"prove it or leave it out."** Upload a jacket with no visible brand or size label and watch what happens.

First, the listing it writes — and the price it *refuses* to guess:

![The generated listing with only verifiable attributes, and a price that abstains because there aren't enough comparable sales](https://raw.githubusercontent.com/rudratoshs/devto-assets/main/honest-lister/ui-listing.png)

- Category, colour, condition come back **observed** — they're in the photo.
- Brand, size, material come back **unverified** — the photo doesn't establish them, so they're kept *out* of the listing.
- The price **abstains** — only 3 comparable sales, not enough to be confident.

Then the publishability gate — which doesn't just flag the gaps, it tells you the exact photo that would close each one:

![The publishability gate marks the listing reviewable, lists the unverified attributes to confirm, and requests the specific close-up photos that would resolve them](https://raw.githubusercontent.com/rudratoshs/devto-assets/main/honest-lister/ui-gate.png)

A normal AI tool, handed the same photo, will cheerfully write "Navy cotton jacket, size M" — and two of those three words are guesses.

**Try the hosted demo:** **[honest-lister.onrender.com](https://honest-lister.onrender.com)** (give it ~30s to wake if it's been idle).

**Or run it yourself (no model download needed):**

```bash
git clone https://github.com/rudratoshs/honest-lister && cd honest-lister
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload        # open http://127.0.0.1:8000
```

The demo runs on a deterministic backend so the page never hangs — point it at a real open-weight model with one environment variable (more on that below).

---

## How I built it: the evidence ledger

The core idea is that a listing is **a set of claims that each have to earn their place**, not a paragraph to be generated. So the vision model isn't asked to "write a listing." It's asked, per attribute: what is it, how sure are you, and *what in the photo shows that?* Then a small, boring, auditable classifier assigns each claim one of four statuses:

| Status | Meaning | Goes in the listing? |
|---|---|---|
| 🟢 **Observed** | Directly visible, above the confidence bar | ✅ stated as fact |
| 🔵 **Inferred** | Plausible from the photo, not certain | ✅ stated, hedged ("appears to be…") |
| 🟠 **Unverified** | A guess: low confidence or no visual evidence | ❌ held back, surfaced to confirm |
| 🔴 **Contradicted** | Conflicts with the photo or with what you typed | ⛔ blocks publishing |

Two rules do most of the work:

- **A claim with no cited visual evidence is a guess, no matter how confident the model sounds.** If the model says "Nike, 0.62 confidence" but can't point to a logo, that's unverified. Full stop.
- **High-stakes attributes (brand, size, material, authenticity) are never merely "inferred."** They clear a stricter bar or they don't get stated. These are the ones that get you banned.

Here's the actual classifier — it's deliberately small enough to read:

```python
# high-stakes attributes are never merely 'inferred'
if claim.is_high_stakes:
    if claim.confidence >= HIGH_STAKES_CONF:        # 0.90
        claim.status = Status.OBSERVED
    else:
        claim.status = Status.UNVERIFIED
    return claim

# ordinary attributes can be observed or inferred
if claim.confidence >= OBSERVE_CONF:                 # 0.80
    claim.status = Status.OBSERVED
elif claim.confidence >= INFER_CONF:                 # 0.55
    claim.status = Status.INFERRED
else:
    claim.status = Status.UNVERIFIED
```

Here's the ledger on the jacket — each attribute carrying its status, confidence, and the exact visual evidence (or the reason it couldn't be established):

![The evidence ledger: category, colour and condition are observed (teal); brand, size and material are unverified (amber), each with the reason it was held back](https://raw.githubusercontent.com/rudratoshs/devto-assets/main/honest-lister/ui-ledger.png)

The listing copy is then assembled **only from observed and inferred claims**. An unverified brand physically cannot reach the title. A contradiction (you typed "Mango," the photo clearly shows a Zara tag) blocks the publish and tells you why.

### It really runs on an open-weight model

This isn't a prompt-wrapper around a closed API. The vision backend runs **moondream2** (open weights, ~2B params) locally. Here's a genuine run on a real photo of a pair of jeans:

```
RAW MOONDREAM ANSWER (real photo):  'Blue Jeans'
PARSED CLAIM: {value: 'Blue Jeans', confidence: 0.92, evidence: 'Blue Jeans'}
```

And an honest number, because this whole project is about honest numbers: on my laptop's **CPU**, that single query took **285 seconds**. A 2B model in float32 with no GPU is slow. That's not a flaw in the idea — it's the reason the architecture separates the fast, reliable demo backend from the real-inference backend (which belongs on a GPU, e.g. a Hugging Face Space), and it's the reason open-weight *local* inference is the whole economic argument (next section). I'd rather tell you it took 285 seconds than quietly hide it behind a cached result.

---

## The part I'm proudest of: measuring the honesty

Anyone can *say* their AI hallucinates less. So I built a benchmark into the repo that proves it — or would have exposed it if it weren't true.

It takes the **same vision-model output** and runs it through two systems: a normal "state everything" generator, and Honest Lister. Both are scored against hand-annotated ground truth — for each attribute, *is it actually establishable from the photo?*

```
  Honesty Benchmark   (n = 7 annotated items)   ·   preliminary, small set
  ------------------------------------------------------------------
  metric                                  baseline        honest
  ------------------------------------------------------------------
  unsupported claims / listing  (lower)       1.14           0.0
  correct-abstention rate       (higher)      0.47           1.0
  supported attrs retained      (higher)      1.0            0.96
  human corrections / listing   (lower)       1.14           0.0
  ------------------------------------------------------------------
```

![The Honesty Benchmark running in the app: baseline vs Honest Lister across the four metrics, with the better cell highlighted in each row](https://raw.githubusercontent.com/rudratoshs/devto-assets/main/honest-lister/ui-benchmark.png)

Read it carefully, because the interesting number is **0.96, not 1.0**.

- Honest Lister cuts unsupported claims from **1.14 per listing to 0** and takes correct abstention from 47% to 100%.
- But it retains **96%** of the genuinely knowable attributes — **not 100%.** One of my test items is a North Face coat where the logo *is* readable to a human, but the model's confidence landed just under the high-stakes bar, so Honest Lister over-abstained and dropped a real attribute.

That 4% is the honest cost, and I left it in on purpose. The goal was never *maximum abstention* — a tool that refuses to describe anything is perfectly safe and perfectly useless. The goal is **fewer false claims while the listing stays usable**, and you recover the dropped attribute in one click by confirming it (which turns it into a trusted, seller-sourced claim). A benchmark that showed a perfect 100%/0 would be the thing I'd distrust most.

```bash
python -m lister.benchmark          # the table above
python -m lister.benchmark --json   # machine-readable
```

Seven items is a seed set, clearly labelled as preliminary — the sample size prints with every run. The honest framing matters more than a big round number.

---

## Pricing you can actually defend

A photo can't tell you market value, so Honest Lister doesn't pretend. Price comes **only from verified attributes** matched against real comparable sales, and it's returned as a **range + number of comparables**, never a single false-precise number. Too few comps? It abstains and tells you which photo would unlock more:

> *"Only 3 comparable sales found (need 5); not enough evidence for a confident price. Confirm brand/size to find more comps."*

That's a price recommendation behaving like a **decision tool** instead of an arbitrary AI number.

---

## Why open innovation matters here

This is the part where "open" isn't a buzzword — it's the business model.

**1. The unit economics only work open.** A serious reseller lists *hundreds* of items. On a closed vision API, every listing is a metered call, and that per-listing cost eats the thin margin on a £15 t-shirt. On a local **open-weight** model, the marginal cost of a listing is **≈ £0**. The open model isn't a nice-to-have — it's the only reason the side hustle is profitable at volume. (The 285-second CPU number is exactly why "local and free" has to be weighed against "fast on a GPU Space" — but both paths are open-weight, and neither sends a cent per listing to a closed vendor.)

**2. Privacy.** On the local backend, photos of someone's inventory — and their home in the background — never leave the machine. I'm careful *not* to overclaim here: if you point it at a hosted inference Space, the image does go to that Space, and the README says so plainly. Honesty about the product includes honesty about where the pixels go.

**3. Inspectability.** The honesty mechanism is the product's whole identity, and it's open source — the ledger, the thresholds, the benchmark, the fixtures. Anyone can check that the abstention is real and not theatre. You can't say "trust my closed hallucination filter" and be taken seriously; you can show the code and the measured trade-off.

---

## My first user

I'll be straight, because this article would be hypocritical otherwise: at submission time the evidence is my seed benchmark, **not** a finished user study. I'd rather say that than fabricate a glowing quote in a piece about not fabricating things.

Here's exactly what I'm putting in front of resellers next, and the three questions that decide whether this is a product or a toy:

1. Does it save you time versus writing the listing yourself?
2. Do the "unverified — confirm" prompts feel like friction, or like a safety net?
3. Would you trust it enough to publish — and would you pay for it?

My hypothesis — the one I want to be proven wrong on — is that the abstention prompts are the *selling point*, not the friction: they flag the exact claims that get listings pulled. The honest next step is 30–50 real annotated photos and a handful of real sellers, which is the top of the roadmap.

---

## Prize categories

- **Best Use of TabPFN** — TabPFN (an open-source tabular foundation model) is wired in as the pricing predictor, but *behind a median-of-comparables baseline*, and used only if it measurably beats it. I'm flagging openly that TabPFN's recommended weights are **non-commercial**, so in a tool you'd actually sell it stays a benchmarked experiment, not a silent dependency. Honesty about licensing is part of honesty about the product.
- **Best Use of Render** — the demo front end deploys on Render, with the open-weight model hosted separately so the web service stays small and the demo stays responsive.

---

## Honest limitations (because that's the whole point)

- A vision model's confidence is **not proof**. An observed-looking logo doesn't establish authenticity; an apparent texture doesn't establish fabric. The evidence checks are validated against annotated ground truth, and the benchmark is 7 items — preliminary, not a universal guarantee.
- Local CPU inference is slow (the 285s number). Real-time use wants a GPU-backed open-weight Space.
- The benchmark needs to grow to 30–50 real photos before any number here should be quoted as settled.

---

## Try it / the code

- **Live demo:** https://honest-lister.onrender.com *(free tier, ~30s cold start)*
- **Repo:** https://github.com/rudratoshs/honest-lister (MIT)
- **Run locally:** `uvicorn app.main:app --reload` → http://127.0.0.1:8000
- **Run the benchmark:** `python -m lister.benchmark`

If the idea that AI should make selling **more truthful**, not just faster, resonates with you — or if you think the evidence-ledger approach breaks somewhere — tell me. I'd rather hear where it fails than where it flatters.

**One question to leave you with:** the last time an AI tool wrote something for you and you shipped it — did you actually check what it *couldn't* know, or just what it *said*? 👇

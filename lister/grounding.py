"""
The honest core of Honest Lister: the evidence ledger.

A reseller gets banned and refunded when a listing claims something the item
isn't. So instead of writing the most *sellable* description, this writes the
most *defensible* one. Every attribute becomes a CLAIM with a status, the
evidence behind it, and a confidence -- and only claims we can stand behind are
written as fact. The rest are surfaced for the seller to confirm, never invented
and never silently dropped.

Four statuses, borrowed from how a careful appraiser actually thinks:

    OBSERVED     directly visible in the photo, above the bar  -> stated as fact
    INFERRED     plausible from the photo, not certain         -> stated, hedged
    UNVERIFIED   a guess: low confidence or no visual evidence -> held back
    CONTRADICTED conflicts with other evidence / seller input  -> blocks publish

"Most AI listing tools optimize what they can generate.
 Honest Lister optimizes what a seller can safely claim."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# Attributes that get a listing pulled or a seller banned if wrong.
# Held to a stricter bar, and never allowed to be merely "inferred".
HIGH_STAKES_ATTRS = {"brand", "size", "material", "model", "authenticity"}

# Attributes a buyer must see before a listing is complete.
MANDATORY_ATTRS = {"category", "condition"}

OBSERVE_CONF = 0.80          # at/above -> directly observed
INFER_CONF = 0.55           # [INFER, OBSERVE) with evidence -> inferred
HIGH_STAKES_CONF = 0.90     # high-stakes must clear this to be stated at all

_ABSTAIN_TOKENS = {"", "unknown", "unclear", "n/a", "not visible",
                   "cannot tell", "can't tell", "none"}


class Status(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


class Source(str, Enum):
    IMAGE = "image"            # derived from the photo by the model
    SELLER = "seller"          # supplied/confirmed by the human


@dataclass
class Claim:
    attr: str
    value: str
    confidence: float
    evidence: str                       # what in the image supports this
    source: Source = Source.IMAGE
    status: Status = Status.UNVERIFIED
    reason: str = ""                    # why it got this status (shown in the ledger)

    @property
    def is_high_stakes(self) -> bool:
        return self.attr in HIGH_STAKES_ATTRS

    @property
    def stateable(self) -> bool:
        """May this claim appear in the listing copy at all?"""
        return self.status in (Status.OBSERVED, Status.INFERRED)


@dataclass
class Defect:
    note: str                          # "light creasing on the toe box"
    confidence: float
    evidence: str


@dataclass
class EvidenceRequest:
    """A concrete photo/action that would resolve an unverified claim."""
    attr: str
    ask: str                           # "add a close-up of the size label"


@dataclass
class Listing:
    title: str
    description: str
    tags: list[str]
    claims: list[Claim] = field(default_factory=list)
    defects: list[Defect] = field(default_factory=list)
    requests: list[EvidenceRequest] = field(default_factory=list)

    def by_status(self, status: Status) -> list[Claim]:
        return [c for c in self.claims if c.status is status]

    @property
    def honesty_score(self) -> float:
        """Share of claims the system was willing to stand behind as fact."""
        total = len(self.claims)
        stated = len([c for c in self.claims if c.stateable])
        return round(stated / total, 2) if total else 1.0


@dataclass
class Publishability:
    ok: bool
    blockers: list[str]                 # must fix before listing
    warnings: list[str]                 # should review
    requests: list[EvidenceRequest]     # photos/confirmations that would help


def classify(claim: Claim, seller_values: dict[str, str] | None = None) -> Claim:
    """Assign one claim its status, considering any seller-provided value."""
    seller_values = seller_values or {}
    value = (claim.value or "").strip()
    low = value.lower()
    evidence = (claim.evidence or "").strip()

    # Seller confirmation is authoritative and needs no image evidence.
    if claim.source is Source.SELLER and low not in _ABSTAIN_TOKENS:
        claim.status = Status.OBSERVED
        claim.reason = "confirmed by seller"
        return claim

    # Contradiction: the photo says one thing with confidence, the seller another.
    seller_val = (seller_values.get(claim.attr) or "").strip().lower()
    if (seller_val and low not in _ABSTAIN_TOKENS and seller_val != low
            and claim.confidence >= OBSERVE_CONF and evidence):
        claim.status = Status.CONTRADICTED
        claim.reason = f"photo shows '{value}' but seller entered '{seller_values[claim.attr]}'"
        return claim

    # The model declined -- the honest path.
    if low in _ABSTAIN_TOKENS:
        claim.status = Status.UNVERIFIED
        claim.reason = "not establishable from the photo"
        return claim

    # A value with no stated visual evidence is a guess, however confident.
    if len(evidence) < 4:
        claim.status = Status.UNVERIFIED
        claim.reason = "no visual evidence cited"
        return claim

    # High-stakes attributes are never merely 'inferred'.
    if claim.is_high_stakes:
        if claim.confidence >= HIGH_STAKES_CONF:
            claim.status = Status.OBSERVED
            claim.reason = "clearly visible and high-confidence"
        else:
            claim.status = Status.UNVERIFIED
            claim.reason = f"high-stakes attribute below {HIGH_STAKES_CONF} confidence"
        return claim

    # Ordinary attributes can be observed or inferred.
    if claim.confidence >= OBSERVE_CONF:
        claim.status, claim.reason = Status.OBSERVED, "clearly visible"
    elif claim.confidence >= INFER_CONF:
        claim.status, claim.reason = Status.INFERRED, "plausible but not certain"
    else:
        claim.status, claim.reason = Status.UNVERIFIED, "confidence too low"
    return claim


def _title_from(claims: list[Claim]) -> str:
    """Title from observed claims only, never an unverified high-stakes value."""
    order = ["brand", "category", "color", "material", "size", "condition"]
    observed = {c.attr: c.value for c in claims if c.status is Status.OBSERVED}
    parts = [observed[a] for a in order if a in observed]
    return " ".join(parts).strip() or "Item for sale (add details)"


def _description_from(claims: list[Claim]) -> str:
    lines = []
    for c in claims:
        if c.status is Status.OBSERVED:
            lines.append(f"{c.attr.capitalize()}: {c.value}.")
        elif c.status is Status.INFERRED:
            lines.append(f"{c.attr.capitalize()}: appears to be {c.value} (not verified).")
    if not lines:
        return ("We could not confidently read any attribute from this photo. "
                "Add clearer, well-lit images before listing.")
    return " ".join(lines)


def build(claims: list[Claim], tags: list[str],
          defects: list[Defect] | None = None,
          seller_values: dict[str, str] | None = None) -> Listing:
    """Classify every claim and assemble a listing from the stateable ones."""
    classified = [classify(c, seller_values) for c in claims]
    defects = defects or []

    requests: list[EvidenceRequest] = []
    for c in classified:
        if c.status is Status.UNVERIFIED and c.is_high_stakes:
            requests.append(EvidenceRequest(
                attr=c.attr,
                ask=f"add a close-up showing the {c.attr} (e.g. the label/tag)",
            ))

    return Listing(
        title=_title_from(classified),
        description=_description_from(classified),
        tags=[t for t in tags if t],
        claims=classified,
        defects=defects,
        requests=requests,
    )


def publishability(listing: Listing, seller_values: dict[str, str] | None = None) -> Publishability:
    """
    Gate: can this listing be published as-is?

    Blocks on contradictions and on missing mandatory attributes. Warns on
    unresolved high-stakes guesses and on detected defects not yet described.
    The seller can clear blockers by confirming facts (which become SELLER claims).
    """
    blockers, warnings = [], []
    present = {c.attr for c in listing.claims if c.stateable}

    for attr in MANDATORY_ATTRS:
        if attr not in present:
            blockers.append(f"missing required attribute: {attr}")

    for c in listing.claims:
        if c.status is Status.CONTRADICTED:
            blockers.append(f"contradiction on {c.attr}: {c.reason}")
        if c.status is Status.UNVERIFIED and c.is_high_stakes:
            warnings.append(f"{c.attr} is unverified -- confirm manually or omit it")

    for d in listing.defects:
        warnings.append(f"possible defect not yet described: {d.note}")

    return Publishability(
        ok=not blockers,
        blockers=blockers,
        warnings=warnings,
        requests=listing.requests,
    )

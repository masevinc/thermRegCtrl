"""
ISO 14505-2:2006(E) Annex D comfort-zone boundaries for equivalent
temperature (teq), per body region.

Values are visually estimated from the standard's Figure D.1 (summer,
cooling mode, 0.6 clo) and Figure D.2 (winter, heating mode, 1.0 clo) --
the standard publishes these boundaries as a graph (17 rows), not a
numeric table, and no precise digitization tool was available while
reading it. The 17 rows visually collapse into 3 clearly distinguishable
tiers (lower body colder-shifted/narrower, hands+arms in between, upper
body+head warmer-tolerant/wider), so only those 3 tiers are encoded here
rather than false per-region precision across all 17 rows.

CONFIDENCE: moderate at best -- this was read off a static rendered image
without pixel-level measurement tools, by eye, over two passes (the first
pass had the lower/upper trend backwards). Treat these numbers as a rough
starting point, not a citable digitization of the standard. If exact
values matter for the thesis, re-read Figure D.1/D.2 directly (with zoom)
from the standard PDF.

Each region has 4 boundaries separating 5 zones:
    1 too cold | 2 cold but comfortable | 3 neutral | 4 warm but comfortable | 5 too hot
"""

from __future__ import annotations

ZONE_LABELS = {
    1: "too cold",
    2: "cold but comfortable",
    3: "neutral",
    4: "warm but comfortable",
    5: "too hot",
}

# The 17 chart rows collapse into 3 visually distinguishable tiers (see
# module docstring): lower body is narrower/warmer-shifted, hands+arms are
# in between, upper body+head is wider/more cold-tolerant.
_SUMMER_TIERS = {
    "lower": (20, 23, 27, 32),   # foot, calf, thigh
    "mid":   (19, 22, 26, 30),   # hand, lower arm, upper arm
    "upper": (17, 21, 25, 29),   # upper back, chest, face, scalp, whole body
}
_WINTER_TIERS = {
    "lower": (18, 21, 24, 30),
    "mid":   (17, 20, 24, 28),
    "upper": (14, 19, 23, 27),
}

# region -> (1|2, 2|3, 3|4, 4|5) boundary teq [degC]
SUMMER_BANDS: dict[str, tuple[float, float, float, float]] = {
    "foot":      _SUMMER_TIERS["lower"],
    "calf":      _SUMMER_TIERS["lower"],
    "thigh":     _SUMMER_TIERS["lower"],
    "hand":      _SUMMER_TIERS["mid"],
    "lowerArm":  _SUMMER_TIERS["mid"],
    "upperArm":  _SUMMER_TIERS["mid"],
    "upperBack": _SUMMER_TIERS["upper"],
    "chest":     _SUMMER_TIERS["upper"],
    "face":      _SUMMER_TIERS["upper"],
    "scalp":     _SUMMER_TIERS["upper"],
    "wholeBody": _SUMMER_TIERS["upper"],
}

WINTER_BANDS: dict[str, tuple[float, float, float, float]] = {
    "foot":      _WINTER_TIERS["lower"],
    "calf":      _WINTER_TIERS["lower"],
    "thigh":     _WINTER_TIERS["lower"],
    "hand":      _WINTER_TIERS["mid"],
    "lowerArm":  _WINTER_TIERS["mid"],
    "upperArm":  _WINTER_TIERS["mid"],
    "upperBack": _WINTER_TIERS["upper"],
    "chest":     _WINTER_TIERS["upper"],
    "face":      _WINTER_TIERS["upper"],
    "scalp":     _WINTER_TIERS["upper"],
    "wholeBody": _WINTER_TIERS["upper"],
}

BANDS_BY_SEASON = {"summer": SUMMER_BANDS, "winter": WINTER_BANDS}


def classify(teq: float, region: str, season: str) -> int:
    """Returns the ISO 14505-2 zone number (1..5) for a measured teq."""
    bands = BANDS_BY_SEASON[season]
    b12, b23, b34, b45 = bands[region]
    if teq < b12:
        return 1
    if teq < b23:
        return 2
    if teq < b34:
        return 3
    if teq < b45:
        return 4
    return 5


def zone_to_sensation_scale(zone: float) -> float:
    """Maps the 5-point ISO zone (1..5) onto a -4..+4 scale, for visual
    comparison against continuous sensation models like BerkeleyModel's
    (-4 very cold .. +4 very hot). zone=3 (neutral) -> 0."""
    return (zone - 3) * 2

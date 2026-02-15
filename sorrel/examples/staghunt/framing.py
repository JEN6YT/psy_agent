from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import re


DEFAULT_NEUTRAL_HARE_LABEL = "ijjhu"
DEFAULT_NEUTRAL_STAG_LABEL = "guydguug"


@dataclass(frozen=True)
class StagHuntFraming:
    mode: str = "natural"
    hare: str = "hare"
    stag: str = "stag"

    @property
    def hare_plural(self) -> str:
        return _pluralize(self.hare)

    @property
    def stag_plural(self) -> str:
        return _pluralize(self.stag)


def _pluralize(term: str) -> str:
    t = str(term).strip()
    if not t:
        return "items"
    if t.endswith("s"):
        return t
    return f"{t}s"


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def resolve_staghunt_framing(config: Any) -> StagHuntFraming:
    world_cfg = _get_attr_or_key(config, "world", {})
    mode = str(_get_attr_or_key(world_cfg, "framing_mode", "natural")).strip().lower()
    if mode not in {"natural", "neutral"}:
        mode = "natural"

    neutral_hare = str(
        _get_attr_or_key(
            world_cfg,
            "neutral_hare_label",
            DEFAULT_NEUTRAL_HARE_LABEL,
        )
    ).strip() or DEFAULT_NEUTRAL_HARE_LABEL
    neutral_stag = str(
        _get_attr_or_key(
            world_cfg,
            "neutral_stag_label",
            DEFAULT_NEUTRAL_STAG_LABEL,
        )
    ).strip() or DEFAULT_NEUTRAL_STAG_LABEL

    if mode == "neutral":
        return StagHuntFraming(mode=mode, hare=neutral_hare, stag=neutral_stag)
    return StagHuntFraming(mode=mode, hare="hare", stag="stag")


def replace_resource_terms(text: str, framing: StagHuntFraming) -> str:
    if framing.mode != "neutral" or not text:
        return text

    def _case_match(source: str, replacement: str) -> str:
        if source.isupper():
            return replacement.upper()
        if source.istitle():
            return replacement.capitalize()
        return replacement

    replacements = {
        "hares": framing.hare_plural,
        "stags": framing.stag_plural,
        "hare": framing.hare,
        "stag": framing.stag,
    }

    out = text
    for src, repl in replacements.items():
        out = re.sub(
            rf"\b{src}\b",
            lambda m: _case_match(m.group(0), repl),
            out,
            flags=re.IGNORECASE,
        )
    return out


def commitment_terms(framing: StagHuntFraming) -> Dict[str, set[str]]:
    # Keep natural terms accepted even in neutral mode for robustness.
    return {
        "stag": {framing.stag.lower(), framing.stag_plural.lower(), "stag", "stags"},
        "hare": {framing.hare.lower(), framing.hare_plural.lower(), "hare", "hares"},
    }

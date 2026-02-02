from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import json
import os
from pathlib import Path

# Optional: package-data loader if you later turn prompts into a Python package
try:
    from importlib.resources import files as pkg_files  # py>=3.9
except Exception:  # pragma: no cover
    try:
        from importlib_resources import files as pkg_files  # type: ignore
    except Exception:
        pkg_files = None  # not strictly needed when using plain files

# ---------- locate prompts root ----------
def _prompts_root() -> Path:
    """
    Resolve where prompt templates live.

    Priority:
      1) Env var SORREL_PROMPTS_DIR (absolute path)
      2) Local './prompts' directory next to this Python file
      3) (Optional) packaged data: 'sorrel.llm_configs.prompts' if it exists
    """
    # 1) env var
    env = os.getenv("SORREL_PROMPTS_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p

    # 2) local ./prompts next to this module
    here = Path(__file__).resolve().parent
    local = here / "prompts"
    if local.is_dir():
        return local

    # 3) optional: packaged data (only if you later create that package)
    if pkg_files is not None:
        try:
            import sorrel.llm_configs.prompts as _prompts_pkg  # type: ignore
            # pkg_files returns a Traversable; convert to Path-like when reading
            return Path(str(pkg_files(_prompts_pkg)))
        except Exception:
            pass

    raise FileNotFoundError(
        "Could not locate prompts directory. "
        "Set SORREL_PROMPTS_DIR or create a ./prompts folder next to this file."
    )

# ---------- low-level loader ----------
def _load_text(rel_name: str) -> str:
    root = _prompts_root()
    path = root / rel_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")

# ---------- registry ----------
@dataclass(frozen=True)
class PromptSpec:
    name: str
    filename: str

class PromptRegistry:
    REFLECTION = PromptSpec("reflection", "reflection_template.txt")
    TURN       = PromptSpec("turn",       "turn_template.txt")
    STAG       = PromptSpec("system_staghunt",     "system_staghunt.md")
    TREASURE   = PromptSpec("system_treasurehunt", "system_treasurehunt.md")

    _cache: dict[str, str] = {}

    @classmethod
    def get(cls, spec: PromptSpec) -> str:
        if spec.filename not in cls._cache:
            cls._cache[spec.filename] = _load_text(spec.filename)
        return cls._cache[spec.filename]

# ---------- safe formatter ----------
class SafeDict(dict):
    def __missing__(self, key):
        # Leave the placeholder verbatim if not provided
        return "{" + key + "}"

def render(template: str, **kwargs) -> str:
    """
    Render using str.format_map with SafeDict so missing keys
    don't crash and remain visible in the output.
    """
    return template.format_map(SafeDict(**kwargs))

# ---------- convenience helpers ----------
def system_staghunt(
    role: str,
    reward_rule: dict | str | None = None,
    vision_radius: int | None = None,
    beam_length: int | None = None,
) -> str:
    tpl = PromptRegistry.get(PromptRegistry.STAG)
    # stringify reward rule in a compact, action-oriented format
    def _format_reward_rule(rule: dict | list | str | None) -> str:
        if isinstance(rule, dict):
            lines: list[str] = []
            params = rule.get("params", {})
            hare = params.get("hare_reward", None)
            stag = params.get("stag_reward", None)
            if hare is not None or stag is not None:
                lines.append(f"Rewards: hare={hare}, stag={stag}.")
            for key in ("rules", "policy", "tips"):
                items = rule.get(key, None)
                if isinstance(items, (list, tuple)) and items:
                    lines.append(f"{key.capitalize()}:")
                    lines.extend(f"- {item}" for item in items)
            if lines:
                return "\n".join(lines)
            return json.dumps(rule, ensure_ascii=False)
        if isinstance(rule, (list, tuple)):
            return "\n".join(f"- {item}" for item in rule)
        return str(rule) if rule else "Not specified."

    reward_rule_str = _format_reward_rule(reward_rule)

    return render(
        tpl,
        role=role,
        reward_rule=reward_rule_str,
        vision_radius=vision_radius,
        beam_length=beam_length,
    )


def system_treasurehunt(role: str, action_table: Mapping[int, str]) -> str:
    tpl = PromptRegistry.get(PromptRegistry.TREASURE)
    table_str = json.dumps({int(k): v for k, v in action_table.items()}, ensure_ascii=False)
    return render(tpl, role=role, action_table=table_str)

def turn_prompt(obs_text: str, memory_text: str, action_table: str) -> str:
    tpl = PromptRegistry.get(PromptRegistry.TURN)
    return render(
        tpl,
        obs_text=obs_text,
        memory_text=memory_text,
        action_table=action_table,
    )

def reflection_prompt(summary: str) -> str:
    tpl = PromptRegistry.get(PromptRegistry.REFLECTION)
    return render(tpl, summary=summary)

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
    action_table: Mapping[int, str],
    reward_rule: dict | str | None = None,
    vision_radius: int | None = None,
    beam_length: int | None = None,
) -> str:
    tpl = PromptRegistry.get(PromptRegistry.STAG)
    table_str = json.dumps({int(k): v for k, v in action_table.items()}, ensure_ascii=False)

    # stringify reward rule (pretty JSON for readability)
    if isinstance(reward_rule, (dict, list)):
        reward_rule_str = json.dumps(reward_rule, ensure_ascii=False, indent=2)
    else:
        reward_rule_str = str(reward_rule) if reward_rule else "Not specified."

    return render(
        tpl,
        role=role,
        action_table=table_str,
        reward_rule=reward_rule_str,
        vision_radius=vision_radius,
        beam_length=beam_length,
    )


def system_treasurehunt(role: str, action_table: Mapping[int, str]) -> str:
    tpl = PromptRegistry.get(PromptRegistry.TREASURE)
    table_str = json.dumps({int(k): v for k, v in action_table.items()}, ensure_ascii=False)
    return render(tpl, role=role, action_table=table_str)

def turn_prompt(obs_text: str, memory_text: str, action_space_json: str) -> str:
    tpl = PromptRegistry.get(PromptRegistry.TURN)
    return render(tpl,
                  obs_text=obs_text,
                  memory_text=memory_text,
                  action_space_json=action_space_json)

def reflection_prompt(summary: str) -> str:
    tpl = PromptRegistry.get(PromptRegistry.REFLECTION)
    return render(tpl, summary=summary)

# sorrel/models/agents.py

from __future__ import annotations
from typing import Any, Sequence, Optional, List, Dict, Mapping
import re, json
import numpy as np

# Keep your original import paths if these live elsewhere
from sorrel.models.base_model import BaseModel, APIClient
from sorrel.llm_configs.templates import (
    system_staghunt, system_treasurehunt, turn_prompt, reflection_prompt
)


def _extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.I | re.S)
    if m:
        return m.group(1)
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            _, end = decoder.raw_decode(text[start:])
            return text[start:start + end]
        except Exception:
            start = text.find("{", start + 1)
    return None


def parse_llm_fields(
    text: str,
    *,
    action_space: Optional[int] = None,
    default_action: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "REASONING": None,
        "ACTION": None,
        "MESSAGE": None,
        "CONFIDENCE": None,
    }
    if not text:
        out["ACTION"] = default_action
        return out

    # JSON first
    try:
        json_text = _extract_json_from_text(text) or text
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            norm = {str(k).strip().lower(): v for k, v in obj.items()}
            out["ACTION"] = norm.get("action", norm.get("action_id"))
            out["MESSAGE"] = norm.get("message")
            out["CONFIDENCE"] = norm.get("confidence", norm.get("conf"))
            out["REASONING"] = norm.get("reasoning", norm.get("reason"))
    except Exception:
        pass

    # Plain-text fallback
    if out["ACTION"] is None:
        m_act = re.search(r"\bACTION\s*[:=]\s*(\d+)", text, re.I)
        if m_act:
            out["ACTION"] = int(m_act.group(1))
    if out["MESSAGE"] is None:
        m_msg = re.search(r"\bMESSAGE\s*[:=]\s*(.+?)(?:\n|$)", text, re.I | re.S)
        if m_msg:
            out["MESSAGE"] = m_msg.group(1).strip()
    if out["CONFIDENCE"] is None:
        m_conf = re.search(r"\bCONFIDENCE\s*[:=]\s*(\d+)", text, re.I)
        if m_conf:
            out["CONFIDENCE"] = int(m_conf.group(1))
    if out["REASONING"] is None:
        m_reas = re.search(
            r"\bREASONING\s*[:=]\s*(.*?)(?=\n\s*(ACTION|MESSAGE|CONFIDENCE)\b|$)",
            text,
            re.I | re.S,
        )
        if m_reas:
            out["REASONING"] = m_reas.group(1).strip()

    # Normalize types/values
    if out["MESSAGE"] is not None:
        msg = str(out["MESSAGE"]).strip()
        out["MESSAGE"] = msg if msg else None
    if out["CONFIDENCE"] is not None:
        try:
            out["CONFIDENCE"] = max(0, min(100, int(out["CONFIDENCE"])))
        except Exception:
            out["CONFIDENCE"] = None
    if out["REASONING"] is not None and not isinstance(out["REASONING"], str):
        out["REASONING"] = str(out["REASONING"]).strip()

    if out["ACTION"] is None:
        nums = re.findall(r"\b(\d+)\b", text)
        if nums:
            out["ACTION"] = int(nums[-1])

    if out["ACTION"] is not None:
        try:
            a = int(out["ACTION"])
            if action_space is not None:
                a = max(0, min(action_space - 1, a))
            else:
                a = max(0, a)
            out["ACTION"] = a
        except Exception:
            out["ACTION"] = None

    if out["ACTION"] is None:
        out["ACTION"] = default_action

    return out


class LLMPlayer(BaseModel):
    """Transformers-backed LLM player with comms + reputation integration."""

    def __init__(
        self,
        *,
        agent_id: int,
        input_size: int | Sequence[int],
        action_space: int,
        memory_size: int,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 256,
        game_type: str = "custom",
        role: str = "strategic agent",
        action_descriptions: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        verbose: bool = False,
        episodic_capacity: int = 512,
        fine_tuning_enabled: bool = False,
        reward_rule: dict | str | None = None,
        vision_radius: int | None = None,
        beam_length: int | None = None,
        **hf_kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            input_size=input_size,
            action_space=action_space,
            memory_size=memory_size,
            episodic_capacity=episodic_capacity,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=custom_system_prompt,
            fine_tuning_enabled=fine_tuning_enabled,
        )
        # ---- FIX: don't write to BaseModel.model_name (read-only); use our own field
        self._hf_model_name = model_name

        self.verbose = verbose
        self.game_type = game_type
        self.role = role
        self.hf_kwargs = hf_kwargs or {}
        self.last_message: Optional[str] = None  # exposed for MessageBus

        self.action_descriptions = (
            [f"Action {i}" for i in range(action_space)]
            if action_descriptions is None else action_descriptions
        )
        if len(self.action_descriptions) != action_space:
            raise ValueError("action_descriptions length must equal action_space")
        self.action_table: Mapping[int, str] = {i: desc for i, desc in enumerate(self.action_descriptions)}

        # System prompt selection
        self.system_prompt = (
            custom_system_prompt or
            (system_staghunt(
                self.role,
                reward_rule,
                vision_radius=vision_radius,
                beam_length=beam_length,
            ) if game_type == "staghunt" else
             system_treasurehunt(self.role, self.action_table) if game_type == "treasurehunt" else
             self._default_system_prompt())
        )

        self._init_transformers()
        self.conversation_history: List[Dict[str, Any]] = []
        self.turn_count = 0

    # ---------------- init ----------------
    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tok_kwargs = (self.hf_kwargs.get("tokenizer_kwargs") or {})
        mdl_kwargs = (self.hf_kwargs.get("model_kwargs") or {})

        # ---- FIX: use self._hf_model_name everywhere
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._hf_model_name, trust_remote_code=True, **tok_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self._hf_model_name,
            trust_remote_code=True,
            torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
            **mdl_kwargs
        ).to(self.device)
        self.torch = torch  # convenience alias

    # ------------- prompt building -------------
    def _default_system_prompt(self) -> str:
        lines = [f"You are a {self.role}. You have {self.action_space} actions:"]
        lines += [f"{i}. {d}" for i, d in self.action_table.items()]
        lines += [
            "",
            "Respond *only* with a short reasoning (1-2 sentences) and an ACTION number.",
            "Format:",
            "REASONING: ...",
            "ACTION: <int>",
            "MESSAGE: <optional short message>"
        ]
        return "\n".join(lines)

    def _build_turn_prompt(self, state_text: str, context: Optional[str]) -> str:
        mem_parts: List[str] = []
        if self.conversation_history:
            recent = self.conversation_history[-3:]
            mem_parts.append("Recent History:\n" + "\n".join(
                f"Turn {h['turn']}: chose {h['action']} ({self.action_descriptions[h['action']]})"
                for h in recent
            ))
        rep_snapshot = self.reputation.snapshot_str(self.agent_id, top=3)
        if rep_snapshot and (not context or "REPUTATION:" not in context):
            mem_parts.append(f"REPUTATION: {rep_snapshot}")
        if context:
            mem_parts.append(context)
        memory_text = "\n\n".join(mem_parts) if mem_parts else "No previous context."

        prompt = turn_prompt(
            obs_text=state_text,
            memory_text=memory_text,
            action_table=json.dumps(self.action_table, ensure_ascii=False, indent=2)
        )
        return f"{self.system_prompt.strip()}\n\n{prompt.strip()}"

    # ------------- single generate() used everywhere -------------
    def generate(self, prompt: str, *, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        temp = float(self.temperature if temperature is None else temperature)
        max_new = int(self.max_tokens if max_tokens is None else max_tokens)

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = enc.input_ids.shape[-1]

        with self.torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                do_sample=(temp > 0),
                temperature=(temp if temp > 0 else 1.0),
                max_new_tokens=max_new,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out_ids[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    # ------------- public API -------------
    def take_action(self, state: Any, context: Optional[str] = None) -> int:
        """Return an action int; also records any MESSAGE: ... for the bus."""
        state_text = state if isinstance(state, str) else self._array_to_text(np.asarray(state))
        prompt = self._build_turn_prompt(state_text, context)
        llm_response = self.generate(prompt)

        action, message, confidence, reasoning = self._parse_json_response(llm_response)


        if self.verbose:
            print("\n" + "="*60)
            print(f"Turn {self.turn_count}")
            print("="*60)
            print(prompt)
            print("\nLLM RESPONSE:\n" + llm_response + "\n")

        # action, confidence = self._parse_action(llm_response)

        # capture MESSAGE for bus
        self.last_message = message

        self.last_parsed = {  # <-- add this field
            "ACTION": action,
            "MESSAGE": message,
            "CONFIDENCE": confidence,
            "REASONING": reasoning,
            "RAW": llm_response,
            "STATE": state_text,
        }

        self.conversation_history.append({
            "turn": self.turn_count,
            "state": state_text,
            "memory": context,
            "response": llm_response,
            "action": action,
            "confidence": confidence,
            "message": self.last_message,
        })
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
        self.turn_count += 1
        return action

    def _extract_json_block(self, src: str) -> Optional[str]:
        return _extract_json_from_text(src)
    
    def _parse_json_response(self, text: str):
        # returns (action:int, message:str|None, confidence:int|None, reasoning:str|None)
        fields = parse_llm_fields(
            text,
            action_space=self.action_space,
            default_action=0,
        )
        msg = fields["MESSAGE"]
        if isinstance(msg, str):
            msg = msg.strip()[:80] if msg else None
        return fields["ACTION"], msg, fields["CONFIDENCE"], fields["REASONING"]

    def generate_text(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        return self.generate(f"{self.system_prompt.strip()}\n\n{prompt.strip()}",
                             temperature=temperature, max_tokens=max_tokens)

    # ------------- parsing & utils -------------
    def _parse_action(self, text: str) -> tuple[int, Optional[int]]:
        # JSON first
        try:
            j = json.loads(text)
            if isinstance(j, dict) and "action_id" in j:
                a = int(j["action_id"])
                return self._sanitize_action(a), None
        except Exception:
            pass
        # "ACTION: N"
        m = re.search(r"action\s*[:=]\s*(\d+)", text, re.I)
        if m:
            return self._sanitize_action(int(m.group(1))), None
        # last number fallback
        nums = re.findall(r"\b(\d+)\b", text)
        return (self._sanitize_action(int(nums[-1])) if nums else 0), None

    def _parse_message(self, text: str) -> Optional[str]:
        # JSON first
        try:
            j = json.loads(text)
            msg = j.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        except Exception:
            pass
        # Plain "MESSAGE: ..."
        m = re.search(r"message\s*[:=]\s*(.+?)(?:\n|$)", text, re.I)
        return m.group(1).strip() if m else None

    def _sanitize_action(self, a: int) -> int:
        return 0 if a is None else max(0, min(self.action_space - 1, int(a)))

    def _array_to_text(self, arr: np.ndarray) -> str:
        return f"State shape: {arr.shape}, mean: {arr.mean():.3f}, std: {arr.std():.3f}"

    def generate_reflection(self, episode_summary: str) -> str:
        prompt = reflection_prompt(episode_summary)
        try:
            return self.generate_text(prompt, temperature=0.4, max_tokens=120).strip()
        except Exception:
            return "No reflection generated."

    def reset(self):
        self.conversation_history.clear()
        self.turn_count = 0
        self.last_message = None

    @property
    def model_name_str(self) -> str:
        # ---- FIX: reflect our internal HF model name
        return f"LLMPlayer-transformers-{self._hf_model_name}"


class APILLMPlayer(LLMPlayer):
    """GPT-4o-backed LLM player."""

    def __init__(
        self,
        *args,
        api_model: Optional[str] = "gpt-4o",
        api_key: Optional[str] = None,
        api_timeout_s: int = 60,
        **kwargs,
    ):
        self._api_model = api_model
        self._api_key = api_key
        self._api_timeout_s = api_timeout_s
        super().__init__(*args, **kwargs)

    def _init_transformers(self) -> None:
        if not self._api_model:
            raise ValueError("Must specify an api_model (default gpt-4o).")

        self.api_client = APIClient(
            model=self._api_model,
            api_key=self._api_key,
            timeout_s=self._api_timeout_s,
        )

        self.device = "api"
        self.tokenizer = None
        self.model = None

    def generate(self, prompt: str, *, temperature=None, max_tokens=None) -> str:
        temp = float(self.temperature if temperature is None else temperature)
        max_new = int(self.max_tokens if max_tokens is None else max_tokens)
        return self.api_client.generate(
            prompt,
            temperature=temp,
            max_tokens=max_new,
            system_prompt=self.system_prompt,
        )

    def generate_text(self, prompt: str, temperature=None, max_tokens=None) -> str:
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)



def resolve_model_class(model_name: str, **model_kwargs):
    api_provider = model_kwargs.get("api_provider")
    if api_provider in ("openai", "gemini"):
        return APILLMPlayer
    if isinstance(model_name, str) and model_name.startswith(("openai:", "gemini:")):
        return APILLMPlayer
    return LLMPlayer

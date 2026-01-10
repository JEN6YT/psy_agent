# sorrel/models/agents.py

from __future__ import annotations
from typing import Any, Sequence, Optional, List, Dict, Mapping
import re, json
import numpy as np

# Keep your original import paths if these live elsewhere
from sorrel.models.base_model import BaseModel
from sorrel.llm_configs.templates import (
    system_staghunt, system_treasurehunt, turn_prompt, reflection_prompt
)


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
                self.action_table,
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
        if context:
            mem_parts.append(context)
        memory_text = "\n\n".join(mem_parts) if mem_parts else "No previous context."

        action_space_json = json.dumps(self.action_table, ensure_ascii=False, indent=2)

        prompt = turn_prompt(
            obs_text=state_text,
            memory_text=memory_text,
            action_space_json=action_space_json
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
            if not src:
                return None
            m = re.search(r"```(?:json)?\s*({.*?})\s*```", src, re.I | re.S)
            if m:
                return m.group(1)
            start = src.find("{")
            end = src.rfind("}")
            if start != -1 and end != -1 and end > start:
                return src[start:end + 1]
            return None
    
    def _parse_json_response(self, text: str):
        # returns (action:int, message:str|None, confidence:int|None, reasoning:str|None)

        block = self._extract_json_block(text) or text

        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                norm = {str(k).strip().lower(): v for k, v in obj.items()}
                action_val = norm.get("action", norm.get("action_id"))
                msg_val = norm.get("message")
                conf_val = norm.get("confidence", norm.get("conf"))
                reas_val = norm.get("reasoning", norm.get("reason"))
                a = self._sanitize_action(int(action_val)) if action_val is not None else None
                m = (str(msg_val).strip()[:80]) if isinstance(msg_val, str) and msg_val.strip() else None
                c = None
                if conf_val is not None:
                    try:
                        c = max(0, min(100, int(conf_val)))
                    except Exception:
                        c = None
                r = str(reas_val).strip() if isinstance(reas_val, str) else None
                if a is not None:
                    return a, m, c, r
        except Exception:
            pass

        import re
        m_act = re.search(r"\bACTION\s*[:=]\s*(\d+)", text, re.I)
        if m_act:
            a = self._sanitize_action(int(m_act.group(1)))
            m_msg = re.search(r"\bMESSAGE\s*[:=]\s*(.+?)(?:\n|$)", text, re.I)
            m_conf = re.search(r"\bCONFIDENCE\s*[:=]\s*(\d+)", text, re.I)
            m_reas = re.search(r"\bREASONING\s*[:=]\s*(.+?)(?:\n|$)", text, re.I)
            msg = m_msg.group(1).strip()[:80] if m_msg else None
            conf = max(0, min(100, int(m_conf.group(1)))) if m_conf else None
            reas = m_reas.group(1).strip() if m_reas else None
            return a, msg, conf, reas

        nums = re.findall(r"\b(\d+)\b", text)
        a = self._sanitize_action(int(nums[-1])) if nums else 0
        return a, None, None, None

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

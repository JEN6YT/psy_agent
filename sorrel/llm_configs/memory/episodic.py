# ============================================================================
# Memory Components
# ============================================================================

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class StepNote:
    """Record of a single interaction step."""
    turn: int
    action_id: int
    message: str
    summary: str
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Episodic memory to store compact step summaries + reflections.
    
    We keep short notes so the agent can recall key past encounters without 
    stuffing the entire history each turn.
    """
    
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._steps: List[StepNote] = []
        self._reflections: List[str] = []

    def add_step(
        self, 
        turn: int, 
        action_id: int, 
        message: str, 
        notes: List[str],
        reward: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Append a compact step summary. Oldest entries roll off."""
        s = f"t{turn}: a={action_id}; notes=" + "; ".join(notes or [])
        step_note = StepNote(
            turn=turn,
            action_id=action_id,
            message=message,
            summary=s,
            reward=reward,
            metadata=metadata or {}
        )
        self._steps.append(step_note)
        if len(self._steps) > self.capacity:
            self._steps.pop(0)

    def add_reflection(self, text: str):
        """Store the last episode's lesson (2 lines max recommended)."""
        self._reflections.append(text)
        if len(self._reflections) > 64:
            self._reflections.pop(0)

    def recent_text(self, k: int = 6) -> str:
        """Return the last k step summaries + (optionally) the latest reflection.
        
        Included in the prompt each turn to aid recall.
        """
        lines = [s.summary for s in self._steps[-k:]]
        if self._reflections:
            lines.append("REFLECT: " + self._reflections[-1].replace("\n", " "))
        return "\n".join(lines)
    
    def get_steps(self, last_n: Optional[int] = None) -> List[StepNote]:
        """Get recent steps for analysis or fine-tuning."""
        if last_n is None:
            return self._steps.copy()
        return self._steps[-last_n:]
    
    def clear(self):
        """Clear all stored steps."""
        self._steps.clear()
    
    def __len__(self):
        return len(self._steps)
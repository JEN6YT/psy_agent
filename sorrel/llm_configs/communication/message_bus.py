
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

class MessageBus:
    """
    Proximity-limited message bus.
      - queue(sender_id, message)
      - deliver(positions, radius): fills inboxes for next turn
      - inbox_for(agent_id): list[str]
    """
    def __init__(self, max_per_agent: int = 16):
        self._pending: List[Tuple[int, str]] = []
        self._inboxes: Dict[int, List[str]] = {}
        self._max = max_per_agent
        self.last_queued = []

    def reset(self, agent_ids: List[int]):
        self._pending.clear()
        self._inboxes = {aid: [] for aid in agent_ids}
        self.last_queued = []

    def queue(self, sender_id: int, message: Optional[str]) -> None:
        if not message:
            return
        self._pending.append((sender_id, str(message)))
        self.last_queued.append((sender_id, str(message)))

    @staticmethod
    def _manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def deliver(self, positions: Dict[int, Tuple[int,int]], radius: int) -> None:
        for sender, msg in self._pending:
            sp = positions.get(sender)
            if sp is None: continue
            for rid, rp in positions.items():
                if rid == sender: continue
                if self._manhattan(sp, rp) <= radius:
                    inbox = self._inboxes.setdefault(rid, [])
                    if len(inbox) < self._max:
                        inbox.append(f"A{sender}: {msg}")
        self._pending.clear()

    def inbox_for(self, agent_id: int) -> List[str]:
        return list(self._inboxes.get(agent_id, []))

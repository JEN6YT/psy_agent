from collections import defaultdict
from typing import Dict

class Reputation:
    """Reputation: per-agent trust (me -> other).
    
    Usage:
    - Update after joint-action outcomes (e.g., +1 for partner stag cooperation,
      -1 when they defect).
    - We show a compact snapshot in the prompt as a bias toward trust/distrust.
    """
    
    def __init__(self):
        self._trust = defaultdict(float)  # key: (me, other) -> score

    def update_pair(self, me: int, other: int, delta: float):
        """Update trust score between two agents."""
        self._trust[(me, other)] += delta

    def get_trust(self, me: int, other: int) -> float:
        """Get trust score for a specific pair."""
        return self._trust[(me, other)]

    def snapshot_str(self, me: int, top: int = 3) -> str:
        """Get formatted string of top trusted/distrusted agents."""
        pairs = [(o, s) for (m, o), s in self._trust.items() if m == me]
        pairs.sort(key=lambda x: -x[1])
        return "; ".join([f"A{o}:{s:+.2f}" for o, s in pairs[:top]])
    
    def get_all_trust_scores(self, me: int) -> Dict[int, float]:
        """Get all trust scores for a specific agent."""
        return {o: s for (m, o), s in self._trust.items() if m == me}
    
    def clear(self):
        """Clear all reputation scores."""
        self._trust.clear()

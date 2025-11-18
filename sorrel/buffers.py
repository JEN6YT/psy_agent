from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from typing import Any, Dict, List, Optional

class LLMBuffer:
    """Buffer class for storing LLM agent interactions.
    
    This stores structured interaction data suitable for LLM agents.
    
    Attributes:
        capacity: Maximum number of experiences to store.
        experiences: List of interaction dictionaries.
        idx: Current position in the buffer.
        size: Current number of stored experiences.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.experiences: List[Dict[str, Any]] = []
        self.idx = 0
        self.size = 0
    
    def add(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add an experience to the buffer.
        
        Args:
            state: The current state (can be text, dict, or numeric).
            action: The action taken.
            reward: The reward received.
            next_state: The resulting next state.
            done: Whether the episode is complete.
            prompt: The prompt used for generation.
            response: The LLM's response.
            metadata: Additional metadata.
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
        else:
            self.experiences[self.idx] = experience
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences randomly.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            List of experience dictionaries.
        """
        if self.size == 0:
            return []
        
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return [self.experiences[i] for i in indices]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all stored experiences."""
        return self.experiences[:self.size]
    
    def clear(self):
        """Clear all experiences."""
        self.experiences.clear()
        self.idx = 0
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.experiences[idx]
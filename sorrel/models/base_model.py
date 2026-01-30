import os
import json
from abc import abstractmethod
from typing import Sequence, Dict, Any, Optional, List, Tuple

import numpy as np
import urllib.request
import urllib.error
from openai import OpenAI

from sorrel.buffers import LLMBuffer
from sorrel.llm_configs.communication.reputation import Reputation
from sorrel.llm_configs.memory.episodic import EpisodicMemory

# ============================================================================
# Base Model
# ============================================================================

class APIClient:
    """Minimal HTTP client for GPT-4o text generation."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout_s = timeout_s

        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY.")

        # Initialize the official OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_tokens: int,
    ) -> str:

        resp = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        return resp.output_text.strip()


class BaseModel:
    """Generic model class for Sorrel LLM agents. All models should wrap around this
    implementation.

    Attributes:
        agent_id: Unique identifier for this agent.
        input_size: The size or shape of the input.
        action_space: The number of possible actions/outputs available.
        memory: The replay buffer for storing experiences.
        episodic_memory: Memory for past interactions and reflections.
        reputation: Reputation tracking for other agents.
        temperature: The sampling temperature for text generation.
        max_tokens: Maximum number of tokens to generate.
        system_prompt: Optional system prompt for the LLM.
        fine_tuning_enabled: Whether fine-tuning is enabled for this model.
    """

    def __init__(
        self,
        agent_id: int,
        input_size: int | Sequence[int],
        action_space: int,
        memory_size: int,
        episodic_capacity: int = 512,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        fine_tuning_enabled: bool = False,
    ):
        """Initialize the LLM agent model.

        Args:
            agent_id: Unique identifier for this agent.
            input_size: The size of the input (e.g., context window size).
            action_space: The number of discrete actions available.
            memory_size: Size of the experience replay buffer.
            episodic_capacity: Capacity of episodic memory.
            temperature: Sampling temperature for generation (0.0-2.0).
            max_tokens: Maximum tokens to generate per action.
            system_prompt: Optional system prompt to guide agent behavior.
            fine_tuning_enabled: Whether to enable fine-tuning capabilities.
        """
        self.agent_id = agent_id
        self.input_size = input_size
        self.action_space = action_space
        
        # LLM-specific buffer instead of numeric buffer
        self.memory = LLMBuffer(capacity=memory_size)
        
        # Add episodic memory and reputation tracking
        self.episodic_memory = EpisodicMemory(capacity=episodic_capacity)
        self.reputation = Reputation()
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.fine_tuning_enabled = fine_tuning_enabled

        # Optional API client for non-local models (e.g., OpenAI/Gemini).
        self.api_client: Optional["APIClient"] = None
        
        # Track current episode
        self.current_turn = 0
        self.current_episode = 0

    @abstractmethod
    def take_action(self, state: Any, context: Optional[str] = None) -> int:
        """Generate an action based on the observed state and optional context.
        Must be implemented by all subclasses.

        Args:
            state: The current state/observation.
            context: Optional additional context for the LLM.

        Returns:
            The action chosen (as an integer index).
        """
        pass

    @abstractmethod
    def generate_text(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text based on a prompt. Must be implemented by subclasses.

        Args:
            prompt: The input prompt for text generation.
            temperature: Override default temperature if provided.
            max_tokens: Override default max_tokens if provided.

        Returns:
            Generated text string.
        """
        pass

    def add_experience(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        message: str = "",
        notes: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an experience to both replay buffer and episodic memory.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting next state.
            done: Whether the episode is complete.
            message: Short message about this step.
            notes: List of notes for episodic memory.
            prompt: The prompt used for generation.
            response: The LLM's response.
            metadata: Optional metadata.
        """
        # Add to replay buffer
        self.memory.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            prompt=prompt,
            response=response,
            metadata=metadata
        )
        
        # Add to episodic memory
        self.episodic_memory.add_step(
            turn=self.current_turn,
            action_id=action,
            message=message,
            notes=notes or [],
            reward=reward,
            metadata=metadata
        )
        
        self.current_turn += 1
        
        if done:
            self.current_episode += 1

    def update_reputation(self, other_agent_id: int, delta: float):
        """Update reputation score for another agent.
        
        Args:
            other_agent_id: ID of the other agent.
            delta: Change in trust score (positive = more trust).
        """
        self.reputation.update_pair(self.agent_id, other_agent_id, delta)

    def get_context_prompt(self, recent_steps: int = 6, top_agents: int = 3) -> str:
        """Build a context prompt with recent memory and reputation.
        
        Args:
            recent_steps: Number of recent steps to include.
            top_agents: Number of top reputation scores to show.
            
        Returns:
            Formatted context string for the LLM.
        """
        parts = []
        
        if self.system_prompt:
            parts.append(f"SYSTEM: {self.system_prompt}")
        
        # Add recent episodic memory
        recent = self.episodic_memory.recent_text(k=recent_steps)
        if recent:
            parts.append(f"RECENT MEMORY:\n{recent}")
        
        # Add reputation snapshot
        rep_snapshot = self.reputation.snapshot_str(self.agent_id, top=top_agents)
        if rep_snapshot:
            parts.append(f"REPUTATION: {rep_snapshot}")
        
        return "\n\n".join(parts)

    def train_step(
        self, 
        batch_size: int = 32,
        learning_rate: float = 1e-5
    ) -> Dict[str, float]:
        """Train the model on a batch of experiences from memory.

        Args:
            batch_size: Number of samples to use for training.
            learning_rate: Learning rate for fine-tuning.

        Returns:
            Dictionary containing training metrics (e.g., loss, perplexity).
        """
        if not self.fine_tuning_enabled:
            return {"loss": 0.0, "status": "fine_tuning_disabled"}
        
        if len(self.memory) < batch_size:
            return {"loss": 0.0, "status": "insufficient_data"}
        
        # Sample batch from memory
        batch = self.memory.sample(batch_size)
        
        # Subclasses should implement actual training logic
        return {"loss": 0.0, "perplexity": 1.0, "samples": len(batch)}

    def fine_tune(
        self,
        training_data: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        use_memory: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune the LLM on provided training data or stored experiences.

        Args:
            training_data: List of training examples. If None, uses memory buffer.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimization.
            use_memory: Whether to use stored experiences if training_data is None.
            **kwargs: Additional fine-tuning parameters.

        Returns:
            Dictionary containing fine-tuning results and metrics.
        """
        if not self.fine_tuning_enabled:
            return {"status": "error", "message": "Fine-tuning not enabled"}
        
        # Use memory buffer if no training data provided
        if training_data is None and use_memory:
            training_data = self.memory.get_all()
        
        if not training_data:
            return {"status": "error", "message": "No training data available"}
        
        # Placeholder - subclasses should implement actual fine-tuning logic
        return {
            "status": "success",
            "epochs": epochs,
            "final_loss": 0.0,
            "samples_trained": len(training_data)
        }

    def reset(self):
        """Reset episode-specific parameters at the beginning of a new epoch."""
        self.current_turn = 0

    def reset_all_memory(self):
        """Clear all memory stores (use with caution)."""
        self.memory.clear()
        self.episodic_memory.clear()
        self.reputation.clear()
        self.current_turn = 0
        self.current_episode = 0

    def set_temperature(self, new_temperature: float) -> None:
        """Update the sampling temperature for text generation.

        Args:
            new_temperature: New temperature value (typically 0.0-2.0).
        """
        self.temperature = max(0.0, min(new_temperature, 2.0))

    def adjust_temperature(self, adjustment: float) -> None:
        """Adjust temperature by a relative amount.

        Args:
            adjustment: Amount to adjust (positive increases, negative decreases).
        """
        self.temperature = max(0.0, min(self.temperature + adjustment, 2.0))

    def add_reflection(self, text: str):
        """Add a reflection about the last episode.
        
        Args:
            text: Reflection text (keep it concise, 1-2 sentences).
        """
        self.episodic_memory.add_reflection(text)

    def start_epoch_action(self, **kwargs):
        """Actions to perform before each epoch."""
        pass

    def end_epoch_action(self, **kwargs):
        """Actions to perform after each epoch.
        
        This is a good place to generate reflections or adjust strategies.
        """
        pass

    def save(self, file_path: str | os.PathLike) -> None:
        """Save the model weights and parameters in the specified location.

        If the model has been fine-tuned, saves the fine-tuned weights.

        .. note:: This is an abstract function. It must be implemented by a subclass 
                  in order to save a model.

        Parameters:
            file_path: The full path to the model, including file extension.
        """
        pass

    def load(self, file_path: str | os.PathLike) -> None:
        """Load model weights and parameters from the specified location.

        Parameters:
            file_path: The full path to the saved model.
        """
        pass

    @property
    def model_name(self) -> str:
        """Get the name of the model class."""
        return self.__class__.__name__

    @property
    def supports_fine_tuning(self) -> bool:
        """Check if the model supports fine-tuning."""
        return self.fine_tuning_enabled

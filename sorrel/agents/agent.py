from abc import abstractmethod
from typing import Optional, List, Dict, Any, Tuple

from sorrel.action.action_spec import ActionSpec
from sorrel.entities import Entity
from sorrel.models import BaseModel
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds import Gridworld


class LLMAgent[W: Gridworld](Entity[W]):
    """An LLM-based agent for multi-agent strategic environments.
    
    Unlike traditional RL agents that work with numeric observations, LLM agents:
    - Process text-based observations and generate natural language reasoning
    - Communicate with other agents via messages
    - Track reputation and past interactions
    - Use episodic memory for strategic decision-making
    
    This is a subclass of :py:class:`sorrel.entities.Entity`.

    Attributes:
        observation_spec: The observation specification to use for this agent.
        action_spec: The action specification defining available actions.
        model: The BaseModel (LLM-based) that this agent uses.
        agent_id: Unique identifier for this agent in multi-agent scenarios.
        communication_enabled: Whether this agent can send/receive messages.
        current_message: The message this agent wants to broadcast this turn.
        received_messages: Messages received from other agents this turn.
        
    Attributes that override parent (Entity)'s default values:
        - :attr:`has_transitions` - Defaults to True instead of False.
    """

    observation_spec: ObservationSpec
    action_spec: ActionSpec
    model: BaseModel
    agent_id: int
    communication_enabled: bool
    current_message: Optional[str]
    received_messages: List[Tuple[int, str]]  # (sender_id, message)

    def __init__(
        self,
        agent_id: int,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        model: BaseModel,
        location=None,
        communication_enabled: bool = True,
        communication_range: Optional[int] = None,
    ):
        """Initialize an LLM agent.
        
        Args:
            agent_id: Unique identifier for this agent.
            observation_spec: Specification for observations.
            action_spec: Specification for actions.
            model: The LLM-based model for decision-making.
            location: Initial location in the world.
            communication_enabled: Whether agent can communicate with others.
            communication_range: Maximum distance for communication. 
                If None, uses observation_spec.vision_radius.
                Set to -1 for unlimited range.
        """
        # initializations based on parameters
        self.agent_id = agent_id
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.model = model
        self._location = location
        self.communication_enabled = communication_enabled
        
        # Set communication range
        if communication_range is None:
            # Default: use vision radius from observation spec
            self.communication_range = getattr(observation_spec, 'vision_radius', None)
        elif communication_range == -1:
            # Unlimited range
            self.communication_range = None
        else:
            self.communication_range = communication_range
        
        # Communication state
        self.current_message = None
        self.received_messages = []
        
        # Interaction tracking
        self.last_action = None
        self.last_state_text = None
        self.turn_count = 0

        super().__init__()

        # overriding parent default attributes
        self.has_transitions = True

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent for a new episode.
        
        This should clear episode-specific state but preserve long-term memory
        like reputation scores and episodic memories across episodes.
        """
        self.current_message = None
        self.received_messages = []
        self.last_action = None
        self.last_state_text = None
        self.turn_count = 0
        self.inventory = {"hare": 0, "stag": 0}
        self.ready = False
        self.pending_reward = 0.0
        self.received_interaction_reward = False
        if hasattr(self.model, "reset"):
            self.model.reset()

    @abstractmethod
    def pov(self, world: W) -> str:
        """Defines the agent's observation function (text-based for LLM).

        For LLM agents, this returns a natural language description of what
        the agent observes, rather than a numeric array.

        Args:
            world: The environment that this agent is observing.

        Returns:
            str: A textual description of the observed state, e.g.:
                "You are at position (3, 5). You see Agent 2 nearby. 
                 Resources: 10 coins. Turn: 5/20."
        """
        pass

    def format_observation_with_context(self, world: W) -> str:
        """Create a complete observation including state, memory, and messages.
        
        This combines:
        1. Current state observation from pov()
        2. Recent episodic memory
        3. Reputation scores
        4. Messages from other agents
        
        Args:
            world: The environment being observed.
            
        Returns:
            str: Formatted observation string for the LLM.
        """
        parts = []
        
        # Current state
        current_state = self.pov(world)
        parts.append(f"CURRENT STATE:\n{current_state}")
        
        # Memory context (from model)
        memory_context = self.model.get_context_prompt(
            recent_steps=6, 
            top_agents=3
        )
        if memory_context:
            parts.append(f"\n{memory_context}")
        
        # Messages from other agents
        if self.received_messages:
            msg_text = "\n".join([
                f"Agent {sender}: {msg}" 
                for sender, msg in self.received_messages
            ])
            parts.append(f"\nMESSAGES:\n{msg_text}")
        
        # # Available actions
        # action_desc = self.action_spec.describe_actions()
        # parts.append(f"\nAVAILABLE ACTIONS:\n{action_desc}")
        
        return "\n".join(parts)

    @abstractmethod
    def get_action(self, state_text: str) -> int:
        """Gets the action to take based on the textual state description.

        For LLM agents, this involves:
        1. Formatting a prompt with the state and context
        2. Querying the LLM
        3. Parsing the response to extract an action
        4. Optionally extracting a message to broadcast

        Args:
            state_text: The textual description of the current state.

        Returns:
            int: The action chosen by the agent's model given the state.
        """
        pass

    @abstractmethod
    def parse_llm_response(self, llm_output: str) -> Tuple[int, Optional[str], List[str]]:
        """Parse the LLM's output to extract action, message, and notes.
        
        The LLM response might look like:
        "REASONING: I should cooperate since Agent 3 helped me before.
         ACTION: 2
         MESSAGE: Let's work together on this!
         NOTES: building_trust, reciprocity"
        
        Args:
            llm_output: Raw output from the LLM.
            
        Returns:
            Tuple of (action_id, message, notes):
                - action_id: Integer action to take
                - message: Optional message to broadcast
                - notes: List of short notes for episodic memory
        """
        pass

    @abstractmethod
    def act(self, world: W, action: int) -> float:
        """Act on the environment and return the reward.

        Args:
            world: The environment in which the agent is acting.
            action: An element from this agent's action space indicating the action to take.

        Returns:
            float: The reward associated with the action taken.
        """
        pass

    @abstractmethod
    def is_done(self, world: W) -> bool:
        """Determines if the agent is done acting given the environment.

        This might be based on:
        - Maximum number of turns reached
        - Goal achieved
        - Agent eliminated/failed

        Args:
            world: The environment that the agent is in.

        Returns:
            bool: Whether the agent is done acting.
        """
        pass

    def receive_message(self, sender_id: int, message: str) -> None:
        """Receive a message from another agent.
        
        Args:
            sender_id: ID of the agent sending the message.
            message: The message content.
        """
        if self.communication_enabled:
            self.received_messages.append((sender_id, message))

    def broadcast_message(self) -> Optional[str]:
        """Get the message this agent wants to broadcast to others.
        
        Returns:
            Optional[str]: The message to broadcast, or None if no message.
        """
        return self.current_message

    def clear_messages(self) -> None:
        """Clear received messages (call at start of new turn)."""
        self.received_messages = []
        self.current_message = None

    def update_reputation_for_interaction(
        self, 
        other_agent_id: int, 
        outcome: str,
        delta: Optional[float] = None
    ) -> None:
        """Update reputation based on an interaction outcome.
        
        Args:
            other_agent_id: ID of the other agent.
            outcome: Description of outcome (e.g., "cooperated", "defected").
            delta: Optional explicit trust change. If None, inferred from outcome.
        """
        if delta is None:
            # Infer delta from outcome
            if "cooperate" in outcome.lower() or "help" in outcome.lower():
                delta = 1.0
            elif "defect" in outcome.lower() or "betray" in outcome.lower():
                delta = -2.0
            else:
                delta = 0.0
        
        self.model.update_reputation(other_agent_id, delta)

    def add_memory(
        self, 
        state_text: str, 
        action: int, 
        reward: float, 
        done: bool,
        llm_response: Optional[str] = None,
        message_sent: Optional[str] = None,
        notes: Optional[List[str]] = None
    ) -> None:
        """Add an experience to the agent's memory.

        For LLM agents, this stores both structured experience data and
        episodic memory with natural language notes.

        Args:
            state_text: Textual description of the state.
            action: The action taken by the agent.
            reward: The reward received by the agent.
            done: Whether the episode terminated after this experience.
            llm_response: The full LLM response (for fine-tuning).
            message_sent: Message broadcast by the agent.
            notes: Short notes about this turn for episodic memory.
        """
        # Get next state for experience replay
        # Note: This is called AFTER acting, so current pov is "next state"
        # The state_text parameter is the "previous state"
        
        # Create prompt that was used (for fine-tuning)
        prompt = f"STATE: {state_text}\nWhat action should you take?"
        
        # Add to model's memory buffers
        self.model.add_experience(
            state=state_text,
            action=action,
            reward=reward,
            next_state=self.last_state_text,  # Will be updated next turn
            done=done,
            message=message_sent or "",
            notes=notes or [],
            prompt=prompt,
            response=llm_response,
            metadata={
                "turn": self.turn_count,
                "agent_id": self.agent_id,
                "received_messages": self.received_messages.copy()
            }
        )

    def generate_reflection(self, world: W) -> None:
        """Generate a reflection about the episode using the LLM.
        
        This creates a short summary/lesson that will be stored in episodic
        memory and influence future decisions.
        
        Args:
            world: The environment (for getting episode statistics).
        """
        # Build reflection prompt
        episode_steps = self.model.episodic_memory.get_steps()
        total_reward = sum(step.reward for step in episode_steps)
        
        reflection_prompt = f"""
You are Agent {self.agent_id}. The episode just ended.
Total reward: {total_reward}
Number of turns: {len(episode_steps)}

Recent interactions:
{self.model.episodic_memory.recent_text(k=10)}

Write a brief 1-2 sentence reflection about what you learned and how you should 
adjust your strategy in future episodes.
"""
        
        try:
            reflection = self.model.generate_text(
                reflection_prompt, 
                temperature=0.5,
                max_tokens=100
            )
            self.model.add_reflection(reflection.strip())
        except Exception as e:
            # If reflection fails, just skip it
            print(f"Warning: Reflection generation failed for Agent {self.agent_id}: {e}")

    def transition(self, world: W) -> None:
        """Processes a full transition step for the LLM agent.

        This function does the following:
        1. Clear old messages and get current state observation
        2. Format observation with memory/reputation context
        3. Query LLM for action (and extract message)
        4. Execute action and get reward
        5. Broadcast message to other agents
        6. Update memory and reputation
        7. Check if done, generate reflection if episode ended

        Args:
            world: The environment that this agent is acting in.
        """
        # Clear messages from previous turn
        self.clear_messages()
        
        # Get current state with full context
        state_text = self.format_observation_with_context(world)
        self.last_state_text = state_text
        
        # Get action from LLM
        action = self.get_action(state_text)
        self.last_action = action
        
        # Execute action and get reward
        reward = self.act(world, action)
        
        # Check if episode is done
        done = self.is_done(world)
        
        # Update world total reward
        world.total_reward += reward
        
        # Store the experience (subclasses should call this with additional info)
        # Note: Subclasses typically override add_memory in their get_action method
        # to include LLM response and parsed notes
        
        # Increment turn counter
        self.turn_count += 1
        
        # Generate reflection if episode ended
        if done:
            self.generate_reflection(world)

    def get_strategy_summary(self) -> str:
        """Get a summary of this agent's learned strategy.
        
        Returns:
            str: Summary including reputation scores and recent reflections.
        """
        parts = []
        
        parts.append(f"Agent {self.agent_id} Strategy Summary")
        parts.append(f"Episodes completed: {self.model.current_episode}")
        
        # Reputation summary
        trust_scores = self.model.reputation.get_all_trust_scores(self.agent_id)
        if trust_scores:
            parts.append("\nTrust Scores:")
            for other_id, score in sorted(trust_scores.items(), key=lambda x: -x[1]):
                parts.append(f"  Agent {other_id}: {score:+.2f}")
        
        # Recent reflections
        if self.model.episodic_memory._reflections:
            parts.append("\nRecent Reflections:")
            for reflection in self.model.episodic_memory._reflections[-3:]:
                parts.append(f"  - {reflection}")
        
        return "\n".join(parts)

    @property
    def model_name(self) -> str:
        """Get the name of the underlying model."""
        return self.model.model_name
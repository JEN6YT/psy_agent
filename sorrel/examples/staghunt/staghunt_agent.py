"""LLM Agent for Stag Hunt environment - Integrated with MessageBus.

This module provides a concrete implementation of LLMAgent specialized for
the Stag Hunt game, using the MessageBus for efficient message delivery.
"""

from sorrel.action.action_spec import ActionSpec
from sorrel.models.agents import LLMPlayer
from sorrel.llm_configs.observation.serializer import create_llm_observation_parser
from sorrel.examples.staghunt.config import ExperimentConfig, create_default_staghunt_config
from sorrel.agents.agent import LLMAgent
from sorrel.examples.staghunt.env import StagHuntEnv
from sorrel.examples.staghunt.entities import StagResource, HareResource, Empty
from sorrel.llm_configs.communication.message_bus import MessageBus
from sorrel.llm_configs.communication.reputation import Reputation

import json
import re
from typing import Dict, Any, List, Optional, Tuple

ACTION_DESCRIPTIONS = [
    "stay — remain still to wait or conserve energy.",      # 0
    "up — move one tile upward (north)",        # 1
    "right — move one tile to the right (east)",     # 2
    "down — move one tile downward (south)",      # 3
    "left — move one tile to the left (west)",      # 4
    "attack - attack resources with beam "   # 5
]


class StagHuntLLMAgent(LLMAgent[StagHuntEnv]):
    """LLM Agent specialized for Stag Hunt environment.
    
    This agent:
    - Builds concise text observations from the environment
    - Provides detailed information about nearby agents for intelligent communication
    - Uses MessageBus for efficient message delivery
    - Uses LLM to decide actions (0-5: stay/up/right/down/left/interact)
    - Tracks reputation and interactions with other agents
    """

    def __init__(
        self, 
        agent_id: int, 
        config: ExperimentConfig, 
        model: LLMPlayer,
        message_bus: Optional[MessageBus] = None,
        reputation: Optional[Reputation] = None
    ):
        """Initialize a Stag Hunt LLM agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Experiment configuration
            model: LLM model for decision-making (must be initialized with same agent_id)
            message_bus: Shared message bus for communication (optional, creates new if None)
            reputation: Shared reputation tracker (optional, uses model's if None)
        """
        # Verify model has correct agent_id
        if model.agent_id != agent_id:
            raise ValueError(f"Model agent_id ({model.agent_id}) must match agent_id ({agent_id})")
        
        # Create observation spec for text-based observations
        obs_spec = create_llm_observation_parser(
            entity_list=["Empty", "Agent", "Hare", "Stag", "Wall"],
            vision_radius=config.observation.vision_radius,
            style="concise"
        )
        
        # Create action spec (6 actions)
        action_spec = ActionSpec([
            "stay",      # 0
            "up",        # 1
            "right",     # 2
            "down",      # 3
            "left",      # 4
            "attack"   # 5
        ])
        
        super().__init__(
            agent_id=agent_id,
            observation_spec=obs_spec,
            action_spec=action_spec,
            model=model,
            location=None,
            communication_enabled=True,
            communication_range=config.observation.vision_radius,
        )
        
        self.max_turns = config.world.max_turns
        self.config = config
        
        # Use provided message bus or create new one
        self.message_bus = message_bus
        
        # Use provided reputation or model's reputation
        self.reputation = reputation if reputation is not None else model.reputation
        self._has_neighbor_recent: bool = False

        self.inventory: Dict[str, int] = {"hare": 0, "stag": 0}
        # optional: used if you later plug in matrix-style stag-hunt interactions
        self.ready: bool = False
        # rewards that will be added on the agent's *next* step (for shared kill reward etc.)
        self.pending_reward: float = 0.0
        # whether this agent received any pending reward last step
        self.received_interaction_reward: bool = False
        self.orientation = 0

    def reset(self) -> None:
        """Reset the agent for a new episode."""
        super().reset()
        # Note: message_bus.reset() should be called by the environment runner

    def _get_nearby_agents(self, world: StagHuntEnv) -> List[Dict[str, Any]]:
        """Get detailed information about nearby agents within vision radius.
        
        Args:
            world: The Stag Hunt environment
            
        Returns:
            List of dictionaries with agent information (id, distance, direction, position)
        """
        my_y, my_x, _ = world.agent_positions[self.agent_id]
        nearby = []
        
        for other_id in range(len(world.agent_positions)):
            if other_id == self.agent_id:
                continue
            
            other_y, other_x, _ = world.agent_positions[other_id]
            distance = abs(my_y - other_y) + abs(my_x - other_x)
            
            # Check if within vision radius
            vision_radius = self.observation_spec.vision_radius
            if distance <= vision_radius:
                # Determine direction
                dy = other_y - my_y
                dx = other_x - my_x
                
                if abs(dy) > abs(dx):
                    direction = "north" if dy < 0 else "south"
                elif abs(dx) > abs(dy):
                    direction = "west" if dx < 0 else "east"
                else:
                    # Diagonal - combine both
                    ns = "north" if dy < 0 else "south"
                    ew = "west" if dx < 0 else "east"
                    direction = f"{ns}-{ew}"
                
                # Check if adjacent (distance 1)
                is_adjacent = distance == 1
                
                # Get trust score if available
                trust_score = self.reputation.get_trust(self.agent_id, other_id)
                
                nearby.append({
                    'id': other_id,
                    'distance': distance,
                    'direction': direction,
                    'position': (other_y, other_x),
                    'is_adjacent': is_adjacent,
                    'trust': trust_score
                })
        
        # Sort by distance (closest first)
        nearby.sort(key=lambda x: x['distance'])
        return nearby

    def pov(self, world: StagHuntEnv) -> str:
        """Generate a text-based observation from the agent's perspective.
        
        Args:
            world: The Stag Hunt environment
            
        Returns:
            str: Natural language description of the current state
        """
        # Get agent's position
        y, x, _ = world.agent_positions[self.agent_id]
        
        # Get nearby information
        nearby = world._get_nearby_info(self.agent_id)
        
        # Build observation text
        lines = [
            f"Turn {world.turn}/{world.max_turns}",
            f"Your position: ({y}, {x})",
            f"Your cumulative reward: {world.cum_rewards[self.agent_id]:.1f}",
        ]
        
        # Add detailed nearby agent information with trust scores
        nearby_agents = self._get_nearby_agents(world)

        vr = self.config.observation.vision_radius
        self._has_neighbor_recent = any(a["distance"] is not None and a["distance"] <= vr for a in nearby_agents)
        if nearby_agents:
            lines.append("\nNearby Agents (within vision):")
            for agent_info in nearby_agents:
                proximity = "ADJACENT" if agent_info['is_adjacent'] else f"distance {agent_info['distance']}"
                trust_str = f", trust:{agent_info['trust']:+.1f}" if agent_info['trust'] != 0 else ""
                lines.append(
                    f"  - Agent {agent_info['id']}: {proximity}, "
                    f"{agent_info['direction']} of you{trust_str}"
                )
        
        # Add resource information
        if nearby["stag_adjacent"]:
            lines.append("\nA STAG is adjacent!")
        
        if nearby["hare_dir"]:
            lines.append(f"A HARE is {nearby['hare_dir']}")
        
        # Add action reminder
        lines.append("\nActions: 0=stay, 1=up, 2=right, 3=down, 4=left, 5=attack")
        
        # Add strategic context based on what we see
        if nearby["stag_adjacent"] and nearby_agents:
            adjacent_agents = [a for a in nearby_agents if a['is_adjacent']]
            if adjacent_agents:
                # Sort by trust (highest first)
                adjacent_agents.sort(key=lambda a: a['trust'], reverse=True)
                agent_ids = ", ".join(str(a['id']) for a in adjacent_agents)
                lines.append(f"TIP: You and Agent(s) {agent_ids} can cooperate on STAG")
            # else:
            #     lines.append("TIP: STAG needs 2+ agents. Wait for ally to come closer.")
        # elif nearby["stag_adjacent"] and not nearby_agents:
        #     lines.append("TIP: STAG nearby but no allies in sight. Find allies or hunt HARE.")
        
        return "\n".join(lines)

    def format_observation_with_context(self, world: StagHuntEnv) -> str:
        """Create a complete observation including state, memory, and messages.
        
        This OVERRIDES the base class method to use MessageBus for messages.
        
        Args:
            world: The environment being observed.
            
        Returns:
            str: Formatted observation string for the LLM.
        """
        parts = []
        
        # Current state observation (now includes detailed agent info with trust)
        current_state = self.pov(world)
        parts.append(f"CURRENT STATE:\n{current_state}")
        
        # Get messages from MessageBus (if available)
        if self.message_bus:
            messages = self.message_bus.inbox_for(self.agent_id)
            if messages:
                msg_text = "\n".join(messages)
                parts.append(f"\nMESSAGES FROM OTHER AGENTS:\n{msg_text}")
        elif self.received_messages:
            # Fallback to old method if no message bus
            msg_text = "\n".join([
                f"A{sender}: {msg}" 
                for sender, msg in self.received_messages
            ])
            parts.append(f"\nMESSAGES FROM OTHER AGENTS:\n{msg_text}")
        
        # Add communication tip
        nearby_agents = self._get_nearby_agents(world)
        if nearby_agents:
            parts.append(
                f"\nYou can send a MESSAGE to nearby agents. "
                f"Add 'MESSAGE: <your message>' to your response."
            )
        
        # Note: Memory context and action descriptions will be added by
        # the model's _build_turn_prompt() method to avoid duplication
        
        return "\n".join(parts)

    def get_action(self, state_text: str) -> int:
        # include memory + reputation in context (model assembles it too)
        memory_ctx = self.model.get_context_prompt(recent_steps=6, top_agents=3)

        # Query the model. This records action + self.model.last_message internally.
        action = self.model.take_action(state_text, context=memory_ctx)
        action = max(0, min(5, int(action)))

        # Surface an outbound message to the bus if present
        self.current_message = getattr(self.model, "last_message", None)
        if self.message_bus and self.current_message:
            self.message_bus.queue(self.agent_id, self.current_message)

        return action


    def parse_llm_response(
        self, 
        llm_output: str
    ) -> Tuple[int, Optional[str], List[str]]:
        """Parse LLM output to extract action, message, and notes.
        
        Expected formats:
        - JSON: {"action_id": 3, "message": "Let's hunt!", "reasoning": "..."}
        - Plain text: "ACTION: 3" and optionally "MESSAGE: ..."
        
        Args:
            llm_output: Raw text output from the LLM
            
        Returns:
            Tuple of (action_id, message, notes)
        """
        # Try JSON format first
        try:
            data = json.loads(llm_output)
            action = int(data.get("ACTION", data.get("ACTION", 0)))
            action = max(0, min(5, action))
            
            message = data.get("MESSAGE", data.get("message", None))
            if message and isinstance(message, str):
                message = message.strip()
            else:
                message = None
            
            # Extract notes from reasoning
            reasoning = data.get("REASONING", "")
            notes = self._extract_notes(reasoning)
            
            return action, message, notes
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Fallback: Parse plain text format
        # Look for "ACTION: 3" or "action_id: 3" etc.
        action_match = re.search(
            r"action(?:_id)?\s*[:=]\s*(\d+)", 
            llm_output, 
            re.IGNORECASE
        )
        action = int(action_match.group(1)) if action_match else 0
        action = max(0, min(5, action))
        
        # Look for "MESSAGE: ..."
        message_match = re.search(
            r"message\s*[:=]\s*(.+?)(?:\n|$)", 
            llm_output, 
            re.IGNORECASE
        )
        message = message_match.group(1).strip() if message_match else None
        
        # Extract notes from reasoning section
        notes = self._extract_notes(llm_output)
        
        return action, message, notes

    def _extract_notes(self, text: str) -> List[str]:
        """Extract short note keywords from reasoning text.
        
        Args:
            text: Reasoning or thinking text from LLM
            
        Returns:
            List of short note strings
        """
        notes = []
        
        # Look for explicit NOTES section
        notes_match = re.search(r"notes?\s*[:=]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if notes_match:
            notes_text = notes_match.group(1)
            # Split by commas or semicolons
            note_items = re.split(r'[,;]', notes_text)
            notes = [n.strip() for n in note_items if n.strip()]
        
        # Also look for common strategic keywords in the text
        keywords = [
            "cooperate", "defect", "trust", "betray", "stag", "hare",
            "wait", "ally", "solo", "hunt", "coordinate"
        ]
        
        for keyword in keywords:
            if keyword.lower() in text.lower():
                notes.append(keyword)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_notes = []
        for note in notes:
            if note.lower() not in seen:
                seen.add(note.lower())
                unique_notes.append(note)
        
        return unique_notes[:5]  # Limit to 5 notes

    def act(self, world: StagHuntEnv, action: int) -> float:
        """Execute the action in the environment.
        
        For this environment, movement and interactions are handled by
        the environment's step() function. This method just returns 0
        since rewards are calculated by the environment.
        
        Args:
            world: The Stag Hunt environment
            action: Action to execute (0-5)
            
        Returns:
            float: Reward (always 0 here, actual rewards from env.step())
        """
        # The actual action execution happens in world.step()
        # This method is called during the transition but doesn't modify world
        return 0.0

    def is_done(self, world: StagHuntEnv) -> bool:
        """Check if the episode is complete.
        
        Args:
            world: The Stag Hunt environment
            
        Returns:
            bool: True if max turns reached
        """
        return world.turn >= self.max_turns

    def transition(self, world: StagHuntEnv) -> None:
        """Execute a full turn for this agent.
        
        This is called by the environment runner to:
        1. Get observation with messages from MessageBus
        2. Query LLM for action (model adds memory context internally)
        3. Parse response for action and message
        4. Queue message to MessageBus if present
        5. Store experience in memory
        
        Args:
            world: The Stag Hunt environment
        """
        # Clear old-style messages (if not using message bus)
        if not self.message_bus:
            self.clear_messages()
        
        # Get observation with messages and agent details (memory context added by model)
        state_text = self.format_observation_with_context(world)
        self.last_state_text = state_text
        
        # Get action from LLM (also sets self.current_message and queues to message bus)
        action = self.get_action(state_text)
        self.last_action = action
        
        # The environment runner will:
        # 1. Collect actions from all agents
        # 2. Call message_bus.deliver() to distribute messages
        # 3. Call world.step() with all actions
        # 4. Distribute rewards via update_with_reward()
        
        self.current_message = getattr(self.model, "last_message", None)

        self.turn_count += 1

    def update_with_reward(
        self,
        new_obs: Dict[str, Any],
        reward: float,
        done: bool,
        llm_response: Optional[str] = None
    ) -> None:
        """
        Called by the runner AFTER env.step(). Records the reward and the
        LLM output (if provided). Prefer using model.last_parsed if present.
        """
        # Prefer already-parsed fields from the model
        lp = getattr(self.model, "last_parsed", {}) or {}
        raw = llm_response or lp.get("RAW") or ""
        message_sent = self.current_message  # set in transition from model.last_message

        # Build structured notes (don’t lose reasoning/confidence)
        notes = {
            "reasoning": lp.get("REASONING"),
            "confidence": lp.get("CONFIDENCE"),
            "turn": getattr(self, "turn_count", None),
        }

        # Fall back to lightweight parse if you didn’t use last_parsed
        if not notes["reasoning"] and raw:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    notes["reasoning"] = obj.get("REASONING") or obj.get("reasoning")
                    conf = obj.get("CONFIDENCE") or obj.get("confidence")
                    notes["confidence"] = None if conf is None else max(0, min(100, int(conf)))
            except Exception:
                pass
                
        self.last_observation = new_obs
        next_state_text = str(new_obs)

        if not hasattr(self, "obs_history"):
            self.obs_history = []

        self.obs_history.append(next_state_text)
        self.obs_history = self.obs_history[-3:]   # keep last 3 obs only

        self.last_state_text = "\n".join(str(o) for o in self.obs_history)

        # Store experience
        self.add_memory(
            state_text=self.last_state_text,
            action=self.last_action,
            reward=reward,
            done=done,
            llm_response=raw,
            message_sent=message_sent,
            notes=notes,
        )

        if done and hasattr(self, "generate_reflection"):
            self.generate_reflection(world=None)


    def update_reputation_after_interaction(
        self,
        other_agent_ids: List[int],
        outcome: str,
        reward: float
    ) -> None:
        """Update reputation scores after a multi-agent interaction.
        
        Args:
            other_agent_ids: IDs of other agents involved
            outcome: Description of outcome ("cooperated_stag", "solo_hare", etc.)
            reward: Reward received
        """
        for other_id in other_agent_ids:
            if other_id == self.agent_id:
                continue
            
            # Update trust based on outcome
            if "stag" in outcome and reward > 0:
                # Successfully cooperated on stag
                delta = 2.0
                self.reputation.update_pair(self.agent_id, other_id, delta)
            elif "stag" in outcome and reward == 0:
                # Failed stag hunt (other agent may have defected)
                delta = -1.5
                self.reputation.update_pair(self.agent_id, other_id, delta)
            elif "help" in outcome.lower():
                # General helpful behavior
                delta = 1.0
                self.reputation.update_pair(self.agent_id, other_id, delta)


# Utility functions for creating agents

def create_staghunt_agent(
    agent_id: int,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    config: Optional[ExperimentConfig] = None,
    message_bus: Optional[MessageBus] = None,
    reputation: Optional[Reputation] = None,
    **model_kwargs
) -> StagHuntLLMAgent:
    """Create a Stag Hunt LLM agent with default configuration.
    
    Args:
        agent_id: Unique identifier for the agent
        model_name: HuggingFace model name to use
        config: Optional configuration (uses default if None)
        message_bus: Shared message bus (optional)
        reputation: Shared reputation tracker (optional)
        **model_kwargs: Additional kwargs for LLMPlayer
        
    Returns:
        StagHuntLLMAgent: Configured agent ready to play
    """
    if config is None:
        config = create_default_staghunt_config()

    reward_rule = StagHuntEnv.reward_rules_from_config(config)
    
    # Create the LLM model with correct agent_id
    model = LLMPlayer(
        agent_id=agent_id,
        input_size=0,  # Not used for text-based
        action_space=6,
        memory_size=1000,
        model_name=model_name,
        game_type="staghunt",
        action_descriptions=ACTION_DESCRIPTIONS,
        reward_rule=reward_rule,
        **model_kwargs
    )
    
    # Override model's reputation with shared one if provided
    if reputation is not None:
        model.reputation = reputation
    
    return StagHuntLLMAgent(agent_id, config, model, message_bus, reputation)


def create_agent_team(
    num_agents: int,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    config: Optional[ExperimentConfig] = None,
    **model_kwargs
) -> Tuple[List[StagHuntLLMAgent], MessageBus, Reputation]:
    """Create a team of Stag Hunt agents with shared message bus and reputation.
    
    Args:
        num_agents: Number of agents to create
        model_name: HuggingFace model name to use
        config: Optional configuration (uses default if None)
        **model_kwargs: Additional kwargs for LLMPlayer
        
    Returns:
        Tuple of (agents, message_bus, reputation)
    """
    if config is None:
        config = create_default_staghunt_config()
    
    # Create shared message bus and reputation tracker
    message_bus = MessageBus(max_per_agent=16)
    reputation = Reputation()
    
    # Initialize message bus with agent IDs
    agent_ids = list(range(num_agents))
    message_bus.reset(agent_ids)
    
    # Create agents
    agents = []
    for i in range(num_agents):
        agent = create_staghunt_agent(
            agent_id=i,
            model_name=model_name,
            config=config,
            message_bus=message_bus,
            reputation=reputation,
            **model_kwargs
        )
        agents.append(agent)
    
    return agents, message_bus, reputation
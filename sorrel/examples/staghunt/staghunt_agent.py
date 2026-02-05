"""LLM Agent for Stag Hunt environment - Integrated with MessageBus.

This module provides a concrete implementation of LLMAgent specialized for
the Stag Hunt game, using the MessageBus for efficient message delivery.
"""

from sorrel.action.action_spec import ActionSpec
from sorrel.models.agents import LLMPlayer, resolve_model_class, parse_llm_fields
from sorrel.llm_configs.observation.serializer import create_llm_observation_parser
from sorrel.examples.staghunt.config import ExperimentConfig, create_default_staghunt_config
from sorrel.agents.agent import LLMAgent
from sorrel.examples.staghunt.env import StagHuntEnv
from sorrel.examples.staghunt.entities import StagResource, HareResource, Empty
from sorrel.llm_configs.communication.message_bus import MessageBus
from sorrel.llm_configs.communication.reputation import Reputation
from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.examples.staghunt.env import ORIENTATION_VECTORS
from sorrel.examples.staghunt.entities import AttackBeam

import json
import os
import random
import re
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

ACTION_DESCRIPTIONS = [
    "stay — remain still to wait or conserve energy.",      # 0
    "up — move one tile upward (north)",        # 1
    "right — move one tile to the right (east)",     # 2
    "down — move one tile downward (south)",      # 3
    "left — move one tile to the left (west)",      # 4
    "attack - attack resources with beam "   # 5
]

ATTACK = 5
MOVES = (1, 2, 3, 4)

class AntiStallPolicy:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._last_actions: Dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=2))
        self._last_pos: Dict[int, Tuple[int, int]] = {}
        self._last_visible_target_count: Dict[int, int] = {}

    def reset(self) -> None:
        self._last_actions.clear()
        self._last_pos.clear()
        self._last_visible_target_count.clear()
        
    def filter_action(
        self,
        agent_id: int,
        proposed_action: int,
        attack_valid: bool,
        pos_rc: Tuple[int, int],
        visible_target_count: int,
    ) -> int:
        final_action = proposed_action

        # 1) Enforce legality only: if ATTACK is invalid, do NOT attack.
        if final_action == ATTACK and not attack_valid:
            # Prefer repeating the last non-attack (keeps policy more consistent),
            # otherwise pick a random movement.
            last_non_attack = getattr(self, "_last_non_attack_action", {}).get(agent_id)
            if last_non_attack in MOVES:
                final_action = last_non_attack
            else:
                final_action = MOVES[self._rng.randrange(len(MOVES))]

        # 2) Anti-stuck: if repeating the same MOVE 3 times with no progress, jitter.
        history = self._last_actions[agent_id]
        if (
            final_action in MOVES
            and len(history) == 2
            and history[0] == history[1] == final_action
        ):
            last_pos = self._last_pos.get(agent_id)
            last_count = self._last_visible_target_count.get(agent_id)

            pos_unchanged = (last_pos == pos_rc) if last_pos is not None else False
            count_unchanged = (
                last_count == visible_target_count
                if last_count is not None
                else False
            )

            # You can make this stricter by using `and` instead of `or`.
            if pos_unchanged or count_unchanged:
                alternatives = [a for a in MOVES if a != final_action]
                if alternatives:
                    final_action = alternatives[self._rng.randrange(len(alternatives))]

        # 3) Bookkeeping
        self._last_actions[agent_id].append(final_action)
        self._last_pos[agent_id] = pos_rc
        self._last_visible_target_count[agent_id] = visible_target_count

        # Track last non-attack action for better fallback behavior.
        if not hasattr(self, "_last_non_attack_action"):
            self._last_non_attack_action = {}
        if final_action != ATTACK:
            self._last_non_attack_action[agent_id] = final_action

        return final_action


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
        self.attack_cooldown_timer = 0
        self.max_health = getattr(config.world, "agent_health", 5)
        self.health = self.max_health
        # Agents can hunt stags by default; env can override per-agent.
        self.can_hunt = True
        self._anti_stall = AntiStallPolicy(seed=1337 + agent_id)
        self._last_obs_meta: Dict[str, Any] = {}

    def reset(self) -> None:
        """Reset the agent for a new episode."""
        super().reset()
        self.health = self.max_health
        self._anti_stall.reset()
        self._last_obs_meta = {}
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
        
        # Build observation text
        lines = [
            f"Turn {world.turn}/{world.max_turns}",
            f"Your position: ({y}, {x})",
            f"Your cumulative reward: {world.cum_rewards[self.agent_id]:.1f}",
            f"Your health: {self.health}/{self.max_health}",
            f"Attack cooldown: {self.attack_cooldown_timer}",
        ]
        inv = getattr(self, "inventory", {"hare": 0, "stag": 0})
        lines.append(f"Inventory: hare={inv.get('hare', 0)}, stag={inv.get('stag', 0)}")

        vision_radius = int(getattr(world, "vision_radius", self.observation_spec.vision_radius))
        h, w = world.world.height, world.world.width

        def _in_bounds(ny: int, nx: int) -> bool:
            return 0 <= ny < h and 0 <= nx < w

        def _format_offset(ny: int, nx: int) -> str:
            dy = ny - y
            dx = nx - x
            parts = []
            if dy < 0:
                parts.append(f"up {abs(dy)}")
            elif dy > 0:
                parts.append(f"down {dy}")
            if dx < 0:
                parts.append(f"left {abs(dx)}")
            elif dx > 0:
                parts.append(f"right {dx}")
            return ", ".join(parts) if parts else "here"

        ody, odx = ORIENTATION_VECTORS[self.orientation]
        facing_dir = {
            (-1, 0): "NORTH",
            (0, 1): "EAST",
            (1, 0): "SOUTH",
            (0, -1): "WEST",
        }.get((ody, odx), "NORTH")
        lines.append(f"Facing direction (beam): {facing_dir}")

        beam_length = int(getattr(world, "beam_length", 3))
        beam_tiles = world.beam_tiles((y, x), facing_dir, beam_length)
        lines.append(f"POS_RC (Your current position): ({y}, {x})")
        lines.append(f"ORIENTATION: {facing_dir}")
        lines.append(f"BEAM_LEN: {beam_length}")
        lines.append(f"BEAM_TILES_RC: {beam_tiles}")

        # Nearby agents (square vision radius)
        nearby_agent_positions = []
        nearby_agent_details = []
        agent_positions = getattr(world, "agent_positions", {})
        if isinstance(agent_positions, dict):
            agent_items = agent_positions.items()
        else:
            agent_items = enumerate(agent_positions)
        ally_adjacent = False
        for other_id, pos in agent_items:
            if other_id == self.agent_id:
                continue
            oy, ox = pos[0], pos[1]
            if max(abs(oy - y), abs(ox - x)) <= vision_radius:
                oy_i, ox_i = int(oy), int(ox)
                nearby_agent_positions.append((oy_i, ox_i))
                trust_score = self.reputation.get_trust(self.agent_id, other_id)
                nearby_agent_details.append((oy_i, ox_i, trust_score))
                if abs(oy - y) + abs(ox - x) == 1:
                    ally_adjacent = True
        self._has_neighbor_recent = bool(nearby_agent_positions)
        if nearby_agent_positions:
            nearby_agent_details.sort()
            lines.append(
                "Nearby agents within vision: "
                + ", ".join(
                    f"({ay}, {ax}) [{_format_offset(ay, ax)}] trust={trust:.2f}"
                    for ay, ax, trust in nearby_agent_details
                )
            )
        else:
            lines.append("Nearby agents within vision: none")

        # Nearby walls or unpassable obstacles (square vision radius)
        blocked_positions = []
        for dy in range(-vision_radius, vision_radius + 1):
            for dx in range(-vision_radius, vision_radius + 1):
                ny, nx = y + dy, x + dx
                if not _in_bounds(ny, nx):
                    continue
                terrain = world.world.observe((ny, nx, world.world.terrain_layer))
                if not getattr(terrain, "passable", False):
                    blocked_positions.append((ny, nx))
        if blocked_positions:
            blocked_positions.sort()
            lines.append(
                "Nearby walls/unpassable tiles: "
                + ", ".join(
                    f"({by}, {bx}) [{_format_offset(by, bx)}]"
                    for by, bx in blocked_positions
                )
            )
        else:
            lines.append("Nearby walls/unpassable tiles: none")

        # Nearby resources (square vision radius)
        hare_positions = []
        stag_positions = []
        visible_targets: List[Dict[str, Any]] = []
        stag_adjacent = False
        for dy in range(-vision_radius, vision_radius + 1):
            for dx in range(-vision_radius, vision_radius + 1):
                ny, nx = y + dy, x + dx
                if not _in_bounds(ny, nx):
                    continue
                ent = world.world.observe((ny, nx, world.world.dynamic_layer))
                if isinstance(ent, HareResource):
                    hare_positions.append((ny, nx))
                    tgt = {"type": "hare", "pos_rc": (ny, nx)}
                    if hasattr(ent, "health"):
                        tgt["hp"] = int(ent.health)
                    visible_targets.append(tgt)
                elif isinstance(ent, StagResource):
                    stag_positions.append((ny, nx))
                    if abs(dy) + abs(dx) == 1:
                        stag_adjacent = True
                    tgt = {"type": "stag", "pos_rc": (ny, nx)}
                    if hasattr(ent, "health"):
                        tgt["hp"] = int(ent.health)
                    visible_targets.append(tgt)

        # Beam reach info to mark resources in range (unchanged logic).
        beam_hare_positions = set()
        beam_stag_positions = set()
        for i in range(1, beam_length + 1):
            ty, tx = y + ody * i, x + odx * i
            if not world.is_valid_location((ty, tx)):
                continue
            ent = world.world.observe((ty, tx, world.world.dynamic_layer))
            if isinstance(ent, HareResource):
                beam_hare_positions.add((ty, tx))
            elif isinstance(ent, StagResource):
                beam_stag_positions.add((ty, tx))

        attack_valid, hittable_targets = world.compute_attack_valid(
            (y, x),
            facing_dir,
            beam_length,
            visible_targets,
        )
        lines.append(f"HITTABLE_TARGETS: {json.dumps(hittable_targets, ensure_ascii=False)}")
        lines.append(f"ATTACK_VALID: {str(attack_valid).lower()}")

        if hare_positions:
            hare_positions.sort()
            lines.append(
                "Hares within vision: "
                + ", ".join(
                    f"({hy}, {hx}) [{_format_offset(hy, hx)}]"
                    for hy, hx in hare_positions
                )
            )
        else:
            lines.append("Hares within vision: none")
        if beam_hare_positions:
            lines.append(
                "Hares in beam: "
                + ", ".join(
                    f"({hy}, {hx}) [{_format_offset(hy, hx)}]"
                    for hy, hx in sorted(beam_hare_positions)
                )
            )
        else:
            lines.append("Hares in beam: none")

        if stag_positions:
            stag_positions.sort()
            lines.append(
                "Stags within vision: "
                + ", ".join(
                    f"({sy}, {sx}) [{_format_offset(sy, sx)}]"
                    for sy, sx in stag_positions
                )
            )
        else:
            lines.append("Stags within vision: none")
        if beam_stag_positions:
            lines.append(
                "Stags in beam: "
                + ", ".join(
                    f"({sy}, {sx}) [{_format_offset(sy, sx)}]"
                    for sy, sx in sorted(beam_stag_positions)
                )
            )
        else:
            lines.append("Stags in beam: none")
        
        # Add action reminder
        lines.append("\nActions: 0=stay, 1=up, 2=right, 3=down, 4=left, 5=attack")
        
        # Add strategic context based on what we see
        if stag_adjacent and ally_adjacent:
            lines.append("TIP: You may cooperate on STAG and share a message by adding 'MESSAGE: <your message>' to your response.")
            # else:
            #     lines.append("TIP: STAG needs 2+ agents. Wait for ally to come closer.")
        # elif nearby["stag_adjacent"] and not nearby_agents:
        #     lines.append("TIP: STAG nearby but no allies in sight. Find allies or hunt HARE.")
        
        self._last_obs_meta = {
            "pos_rc": (y, x),
            "orientation": facing_dir,
            "beam_len": beam_length,
            "beam_tiles_rc": beam_tiles,
            "visible_target_count": len(visible_targets),
            "attack_valid": attack_valid,
            "hittable_targets": hittable_targets,
        }

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
        
        # # Add communication tip
        # nearby_agents = self._get_nearby_agents(world)
        # if nearby_agents:
        #     parts.append(
        #         f"\nYou can send a MESSAGE to nearby agents. "
        #         f"Add 'MESSAGE: <your message>' to your response."
        #     )
        
        # Note: Memory context and action descriptions will be added by
        # the model's _build_turn_prompt() method to avoid duplication
        
        observation_text = "\n".join(parts)

        # Optional debug output for inspecting agent observations.
        if os.getenv("STAGHUNT_DEBUG_OBS") == "1":
            print(f"\n[DEBUG][Agent {self.agent_id}] Observation:\n{observation_text}\n")

        return observation_text

    def get_action(self, state_text: str, obs_meta: Optional[Dict[str, Any]] = None) -> int:
        # include memory + reputation in context (model assembles it too)
        memory_ctx = self.model.get_context_prompt(recent_steps=6, top_agents=3)

        # Query the model. This records action + self.model.last_message internally.
        action = self.model.take_action(state_text, context=memory_ctx)
        action = max(0, min(5, int(action)))

        meta = obs_meta or self._last_obs_meta or {}
        pos_rc = meta.get("pos_rc")
        if isinstance(pos_rc, (list, tuple)) and len(pos_rc) == 2:
            action = self._anti_stall.filter_action(
                agent_id=self.agent_id,
                proposed_action=action,
                attack_valid=bool(meta.get("attack_valid", False)),
                pos_rc=(int(pos_rc[0]), int(pos_rc[1])),
                visible_target_count=int(meta.get("visible_target_count", 0)),
            )

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
        fields = parse_llm_fields(
            llm_output,
            action_space=6,
            default_action=0,
        )
        action = fields["ACTION"]
        message = fields["MESSAGE"]
        reasoning = fields["REASONING"] or ""
        notes = self._extract_notes(reasoning or llm_output)
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
        action = self.get_action(state_text, obs_meta=self._last_obs_meta)
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

        # Build structured notes as list of strings for episodic memory.
        notes = []
        reasoning = lp.get("REASONING")
        confidence = lp.get("CONFIDENCE")
        if reasoning:
            notes.append(f"reasoning={reasoning}")
        if confidence is not None:
            notes.append(f"confidence={confidence}")
        notes.append(f"turn={getattr(self, 'turn_count', None)}")

        # Fall back to lightweight parse if you didn’t use last_parsed
        if not reasoning and raw:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    extracted_reasoning = obj.get("REASONING") or obj.get("reasoning")
                    if extracted_reasoning:
                        notes.append(f"reasoning={extracted_reasoning}")
                    conf = obj.get("CONFIDENCE") or obj.get("confidence")
                    if conf is not None:
                        notes.append(f"confidence={max(0, min(100, int(conf)))}")
            except Exception:
                pass
                
        # Preserve the pre-action state for the replay transition.
        prev_state_text = self.last_state_text

        self.last_observation = new_obs
        next_state_text = str(new_obs)

        if not hasattr(self, "obs_history"):
            self.obs_history = []

        self.obs_history.append(next_state_text)
        self.obs_history = self.obs_history[-3:]   # keep last 3 obs only

        self.last_state_text = "\n".join(str(o) for o in self.obs_history)

        # Store experience
        self.add_memory(
            state_text=prev_state_text or self.last_state_text,
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

    def spawn_attack_beam(self, world: StagHuntWorld) -> list[tuple[int, int, int]]:
        """Generate an attack beam extending in front of the agent.

        Args:
            world: The world to spawn the beam in.
            
        Returns:
            List of beam locations that were spawned.
        """
        # Get the tiles in front of the agent
        dy, dx = ORIENTATION_VECTORS[self.orientation]

        # Beam length controls how far forward the attack reaches.
        beam_radius = getattr(world, "beam_radius", 1)
        beam_length = getattr(world, "beam_length", beam_radius)
        
        # Check if single-tile beam mode or area attack mode is enabled
        single_tile_attack = getattr(world, "single_tile_attack", False)
        area_attack = getattr(world, "area_attack", False)

        # Calculate beam locations
        beam_locs = []
        y, x, z = self.location

        if area_attack:
            # 3x3 area attack: covers a 3x3 region in front of the agent
            # Calculate perpendicular vectors for left/right
            right_dy, right_dx = -dx, dy  # 90 degrees clockwise
            left_dy, left_dx = dx, -dy  # 90 degrees counter-clockwise
            
            # The 3x3 area is centered 1 tile forward from the agent
            # Generate all 9 tiles in the 3x3 grid
            for i in range(-1, 2):  # -1, 0, 1 (back, center, forward relative to center tile)
                for j in range(-1, 2):  # -1, 0, 1 (left, center, right relative to center tile)
                    # Center tile is 1 tile forward: (y + dy, x + dx)
                    # Offset by i tiles forward and j tiles to the side
                    target_y = y + dy + (i * dy) + (j * left_dy)
                    target_x = x + dx + (i * dx) + (j * left_dx)
                    target = (target_y, target_x, world.beam_layer)
                    if world.valid_location(target):
                        beam_locs.append(target)
        elif single_tile_attack:
            # Attack tiles directly in front of the agent (configurable range, default: 2)
            attack_range = getattr(world, "attack_range", 2)
            for i in range(1, attack_range + 1):
                target = (y + dy * i, x + dx * i, world.beam_layer)
                if world.valid_location(target):
                    beam_locs.append(target)
        else:
            # Forward-only beam behavior
            for i in range(1, beam_length + 1):
                target = (y + dy * i, x + dx * i, world.beam_layer)
                if world.valid_location(target):
                    beam_locs.append(target)

        # Place attack beams in valid locations
        valid_beam_locs = []
        for loc in beam_locs:
            terrain_loc = (loc[0], loc[1], world.terrain_layer)
            if world.valid_location(terrain_loc) and world.map[terrain_loc].passable:
                world.add(loc, AttackBeam())
                valid_beam_locs.append(loc)
        
        return valid_beam_locs


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
    ModelCls = resolve_model_class(model_name, **model_kwargs)
    model = ModelCls(
        agent_id=agent_id,
        input_size=0,  # Not used for text-based
        action_space=6,
        memory_size=1000,
        model_name=model_name,
        game_type="staghunt",
        action_descriptions=ACTION_DESCRIPTIONS,
        reward_rule=reward_rule,
        vision_radius=getattr(config.observation, "vision_radius", None),
        beam_length=getattr(config.world, "beam_length", None),
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

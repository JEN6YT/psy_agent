"""
Stag Hunt Environment using Sorrel framework.

Integrates:
- StagHuntWorld (world.py) for the gridworld structure
- Entities (entities.py) for walls, resources, agents
- MapBasedWorldGenerator (map_generator.py) for ASCII map layouts
- MessageBus + Reputation for comms + memory

Actions (default):
  0: stay, 1: up, 2: right, 3: down, 4: left, 5: interact
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict
import numpy as np
from collections import deque

from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.examples.staghunt.entities import (
    Wall, Sand, Empty, Spawn, StagResource, HareResource,
)
from sorrel.examples.staghunt.map_generator import MapBasedWorldGenerator

# Communication + reputation
from sorrel.llm_configs.communication.message_bus import MessageBus
from sorrel.llm_configs.communication.reputation import Reputation


class StagHuntEnv:
    """
    Gym-like API:
        obs = env.reset()
        obs, rewards, done, info = env.step(actions)

    Reward semantics (configurable in config["world"]):
      - Primary payoffs require INTERACT (if require_interact=True)
      - HARE: exclusive safe payoff to a single interactor (hare_exclusive=True)
      - STAG: quorum_k+ interactors on same tile get stag_reward each
      - Lone stag attempt → sucker_payoff
      - Taste shaping on entry to resource (taste_reward)
      - Movement shaping r_step (if moved) / r_idle (if not)
    """

    # -----------------------------
    # Construction & initialization
    # -----------------------------
    def __init__(self, config: dict, agents: List[Any]):
        self.config = config
        self.agents = agents
        self.num_agents = len(agents)

        # World
        self.world = StagHuntWorld(config, default_entity=Empty())

        # World config
        wcfg = config.get("world", {}) if isinstance(config, dict) else {}
        self.max_turns = int(wcfg.get("max_turns", 100))
        self.generation_mode = wcfg.get("generation_mode", "random")

        # Agent state
        self.agent_positions: Dict[int, Tuple[int, int, int]] = {}
        self.agent_frozen: Dict[int, int] = {}      # agent_id -> remaining frozen turns
        self.cum_rewards: Dict[int, float] = {}

        # Turn counters
        self.turn = 0
        self.total_reward = 0.0

        # Shared communication + reputation
        bus_cfg = self.config.get("message_bus", {}) if isinstance(self.config, dict) else {}
        self.message_bus: MessageBus = MessageBus(max_per_agent=int(bus_cfg.get("max_per_agent", 10)))

        rep_cfg = self.config.get("reputation", {}) if isinstance(self.config, dict) else {}
        self.reputation: Reputation = Reputation(**rep_cfg) if isinstance(rep_cfg, dict) else Reputation()

        # Message radius: prefer bus radius, else obs vision, else 3
        obs_cfg = self.config.get("observation", {}) if isinstance(self.config, dict) else {}
        self.bus_radius: int = int(bus_cfg.get("radius", obs_cfg.get("vision_radius", 3)))

        # Optional LOS callback (not required)
        if not hasattr(self, "line_of_sight"):
            self.line_of_sight = None

        # Build initial terrain/resources shape (but not dynamic placements)
        self._initialize_world()

    # -----------------------------
    # World generation
    # -----------------------------
    def _initialize_world(self) -> None:
        if self.generation_mode == "ascii_map":
            self._initialize_from_map()
        else:
            self._initialize_random()

    def _initialize_from_map(self) -> None:
        map_file = self.config.get("world", {}).get("ascii_map_file")
        if not map_file:
            raise ValueError("ascii_map_file required when generation_mode='ascii_map'")

        if not self.world.map_generator:
            self.world.map_generator = MapBasedWorldGenerator(map_file)

        map_data = self.world.map_generator.parse_map()
        self.world.map_generator.validate_map_for_agents(map_data, self.num_agents)

        # Terrain: walls
        for y, x in map_data.wall_locations:
            self.world.add((y, x, self.world.terrain_layer), Wall())

        # Terrain: spawn (visual)
        for y, x in map_data.spawn_points:
            self.world.add((y, x, self.world.terrain_layer), Spawn())

        # Terrain: resource-capable sand (with optional explicit type)
        for y, x, resource_type in map_data.resource_locations:
            sand = Sand(
                can_convert_to_resource=True,
                respawn_ready=True,
                resource_type=resource_type if resource_type != "random" else None,
            )
            self.world.add((y, x, self.world.terrain_layer), sand)

        # Terrain: plain sand elsewhere
        for y, x in map_data.empty_locations:
            self.world.add((y, x, self.world.terrain_layer), Sand(can_convert_to_resource=False))

        # Cached spawn points (dynamic layer)
        self.world.agent_spawn_points = [(y, x, self.world.dynamic_layer) for y, x in map_data.spawn_points]
        self.world.resource_spawn_points = [(y, x, self.world.dynamic_layer) for y, x, _ in map_data.resource_locations]

    def _initialize_random(self) -> None:
        # Perimeter walls; interior sand (resource-capable)
        for y in range(self.world.height):
            for x in range(self.world.width):
                loc = (y, x, self.world.terrain_layer)
                if y == 0 or y == self.world.height - 1 or x == 0 or x == self.world.width - 1:
                    self.world.add(loc, Wall())
                else:
                    self.world.add(loc, Sand(can_convert_to_resource=True, respawn_ready=True))

        # Candidate interior dynamic cells
        interior = [
            (y, x, self.world.dynamic_layer)
            for y in range(2, self.world.height - 2)
            for x in range(2, self.world.width - 2)
        ]
        np.random.shuffle(interior)

        self.world.agent_spawn_points = interior[: self.num_agents]
        num_resources = min(self.num_agents * 2, max(0, len(interior) - self.num_agents))
        self.world.resource_spawn_points = interior[self.num_agents : self.num_agents + num_resources]

    # -----------------------------
    # Lifecycle
    # -----------------------------
    def reset(self) -> Dict[int, Dict[str, Any]]:
        # Clear dynamic & beam layers
        for y in range(self.world.height):
            for x in range(self.world.width):
                self.world.add((y, x, self.world.dynamic_layer), Empty())
                self.world.add((y, x, self.world.beam_layer), Empty())

        # Reset state
        self.agent_positions.clear()
        self.agent_frozen.clear()
        self.cum_rewards = {i: 0.0 for i in range(self.num_agents)}
        self.turn = 0
        self.total_reward = 0.0

        # Reset bus/reputation
        self.message_bus.reset(list(range(self.num_agents)))
        self.reputation.clear()

        # Place agents at spawns; ensure agents share bus + reputation
        for i, agent in enumerate(self.agents):
            spawn = self.world.agent_spawn_points[i]
            self.agent_positions[i] = spawn
            agent.location = spawn
            # self.world.add(spawn, agent)
            agent.reset()

            if getattr(agent, "message_bus", None) is None:
                agent.message_bus = self.message_bus
            if getattr(agent, "reputation", None) is None:
                agent.reputation = self.reputation

        # Spawn starting resources
        self._spawn_initial_resources()

        # First obs (no messages yet)
        return self._get_observations()

    def _spawn_initial_resources(self) -> None:
        for (y, x, layer) in self.world.resource_spawn_points:
            terrain = self.world.observe((y, x, self.world.terrain_layer))

            def _spawn(rt: Optional[str]) -> Any:
                if rt == "stag":
                    return StagResource(self.world.taste_reward, self.world.destroyable_health)
                if rt == "hare":
                    return HareResource(self.world.taste_reward, self.world.destroyable_health)
                # random fallback
                cls = StagResource if np.random.random() < 0.3 else HareResource
                return cls(self.world.taste_reward, self.world.destroyable_health)

            resource_type = getattr(terrain, "resource_type", None)
            resource = _spawn(resource_type)
            self.world.add((y, x, layer), resource)

    # -----------------------------
    # Step
    # -----------------------------
    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], bool, Dict[str, Any]]:
        """
        actions: {agent_id: action_id}
            0: stay, 1: up, 2: right, 3: down, 4: left, 5: interact
        """
        # Movement (also sets movement/entry flags)
        self._handle_movement(actions)

        # Interactions & rewards
        step_rewards = self._handle_interactions(actions)

        # Timers/transitions/regeneration
        self._update_frozen_timers()
        self._regeneration_step()
        self._transition_step()

        # Accumulate
        for aid, r in step_rewards.items():
            self.cum_rewards[aid] += r
            self.total_reward += r

        # Turn advance & episode end
        self.turn += 1
        done = self.turn >= self.max_turns

        # Deliver messages for NEXT obs
        self.deliver_messages()

        # Observations
        obs = self._get_observations()
        info = {"turn": self.turn, "total_reward": self.total_reward}
        return obs, step_rewards, done, info

    # -----------------------------
    # Movement
    # -----------------------------
    def _is_passable(self, location: Tuple[int, int, int]) -> bool:
        y, x, layer = location
        if not (0 <= y < self.world.height and 0 <= x < self.world.width):
            return False
        terrain = self.world.observe((y, x, self.world.terrain_layer))
        return bool(getattr(terrain, "passable", False))

    def _nearest_passable(self, y0: int, x0: int) -> Tuple[int, int]:
        """BFS to the closest non-wall tile."""
        H, W = self.world.height, self.world.width
        q = deque([(y0, x0)])
        seen = {(y0, x0)}
        while q:
            y, x = q.popleft()
            if self._is_passable((y, x, self.world.terrain_layer)):
                return y, x
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in seen:
                    seen.add((ny, nx))
                    q.append((ny, nx))
        # If the map is entirely walls, something is wrong:
        raise RuntimeError("No passable tile found on the map.")

    def _ensure_agents_on_passable(self) -> None:
        """If any agent is on a wall, relocate to nearest passable tile."""
        for aid, (y, x, layer) in self.agent_positions.items():
            if not self._is_passable((y, x, layer)):
                ny, nx = self._nearest_passable(y, x)
                new_pos = (ny, nx, layer)
                self.agent_positions[aid] = new_pos
                # keep agent object in sync
                self.agents[aid].location = new_pos

    def _handle_movement(self, actions: Dict[int, int]) -> None:
        move_deltas = {0: (0, 0), 1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}

        self._ensure_agents_on_passable()

        # Save old positions and whether they were on a resource
        old_positions = {aid: self.agent_positions[aid] for aid in self.agent_positions}
        old_is_resource: Dict[int, bool] = {}
        for aid, (y, x, _) in old_positions.items():
            ent_old = self.world.observe((y, x, self.world.dynamic_layer))
            old_is_resource[aid] = isinstance(ent_old, (HareResource, StagResource))

        # Propose targets
        targets: Dict[int, Tuple[int, int, int]] = {}
        for aid, act in actions.items():
            if aid >= self.num_agents:
                continue
            if self.agent_frozen.get(aid, 0) > 0:
                targets[aid] = self.agent_positions[aid]
                continue
            if act in move_deltas:
                dy, dx = move_deltas[act]
                y, x, layer = self.agent_positions[aid]
                ny, nx = y + dy, x + dx
                tloc = (ny, nx, layer)
                targets[aid] = tloc if self._is_valid_move(tloc) else self.agent_positions[aid]
            else:
                targets[aid] = self.agent_positions[aid]

        # Resolve collisions (greedy by agent id)
        occupied = set()
        new_positions: Dict[int, Tuple[int, int, int]] = {}
        for aid in sorted(targets.keys()):
            t = targets[aid]
            t2d = (t[0], t[1])
            if t2d in occupied:
                new_positions[aid] = self.agent_positions[aid]
            else:
                new_positions[aid] = t
                occupied.add(t2d)

        # IMPORTANT: do NOT touch dynamic_layer here; only update bookkeeping
        for aid, pos in new_positions.items():
            agent = self.agents[aid]
            agent.location = pos
            self.agent_positions[aid] = pos

        # Movement & entry flags (for shaping/taste)
        self._moved_flags = {}
        self._entered_resource_flags = {}
        for aid in range(self.num_agents):
            oy, ox, _ = old_positions[aid]
            ny, nx, _ = self.agent_positions[aid]
            moved = (oy, ox) != (ny, nx)
            self._moved_flags[aid] = moved

            ent_new = self.world.observe((ny, nx, self.world.dynamic_layer))
            new_is_resource = isinstance(ent_new, (HareResource, StagResource))
            self._entered_resource_flags[aid] = (not old_is_resource[aid]) and new_is_resource

        # (Optional) purely for visualization: draw agents on a non-resource layer
        # for aid, (y, x, _) in self.agent_positions.items():
        #     self.world.add((y, x, self.world.beam_layer), AgentMarker(aid))  # if you have one
        

    def _is_valid_move(self, location: Tuple[int, int, int]) -> bool:
        return self._is_passable(location)

    # -----------------------------
    # Interactions & rewards
    # -----------------------------
    def _handle_interactions(self, actions: Dict[int, int]) -> Dict[int, float]:
        rewards = {i: 0.0 for i in range(self.num_agents)}

        # --- params (support both dict config and object config) ---
        def _wp(key, default):
            if isinstance(self.config, dict):
                return self.config.get("world", {}).get(key, default)
            wobj = getattr(self.config, "world", None)
            return getattr(wobj, key, default) if wobj is not None else default

        hare_reward       = float(_wp("hare_reward", 1.0))
        stag_reward       = float(_wp("stag_reward", 5.0))
        sucker_payoff     = float(_wp("sucker_payoff", 0.0))
        taste_reward      = float(_wp("taste_reward", 0.0))
        r_step            = float(_wp("r_step", 0.0))
        r_idle            = float(_wp("r_idle", 0.0))
        quorum_k          = int(_wp("stag_quorum_k", 2))
        hare_exclusive    = bool(_wp("hare_exclusive", True))
        share_stag_reward = bool(_wp("share_stag_reward", False))

        # --- layer / resource read ---
        res_layer = getattr(self.world, "resource_layer",
                    getattr(self.world, "dynamic_layer", None))
        assert res_layer is not None, "No resource/dynamic layer on world"

        # who is standing on what
        from collections import defaultdict
        bucket = defaultdict(list)  # (y,x,type) -> [aid]
        for aid in range(self.num_agents):
            y, x, _ = self.agent_positions[aid]
            ent = self.world.observe((y, x, res_layer))
            rtype = getattr(ent, "resource_type", None)  # "HARE"/"STAG"/None
            if rtype in ("hare", "stag"):
                bucket[(y, x, rtype)].append(aid)

        consumed = []

        # --- HARE: standing gives reward; exclusivity optional ---
        for (y, x, t), group in bucket.items():
            if t != "hare":
                continue
            loc = (y, x, res_layer)
            ent = self.world.observe(loc)
            if getattr(ent, "resource_type", None) != "hare":
                continue

            if hare_exclusive:
                # deterministic single winner
                winner = min(group)
                rewards[winner] += hare_reward
            else:
                for aid in group:
                    rewards[aid] += hare_reward
            consumed.append(loc)

        # --- STAG: standing with quorum gives reward (no INTERACT gating) ---
        for (y, x, t), group in bucket.items():
            if t != "stag":
                continue
            loc = (y, x, res_layer)
            ent = self.world.observe(loc)
            if getattr(ent, "resource_type", None) != "stag":
                continue

            if len(group) >= quorum_k:
                if share_stag_reward and len(group) > 0:
                    per = stag_reward / len(group)
                    for aid in group:
                        rewards[aid] += per
                else:
                    for aid in group:
                        rewards[aid] += stag_reward
                consumed.append(loc)
            else:
                # optional: sucker payoff for failed solo stag attempts
                for aid in group:
                    rewards[aid] += sucker_payoff

        # --- optional shaping ---
        if taste_reward != 0.0:
            flags = getattr(self, "_entered_resource_flags", None)
            if flags:
                for aid, entered in flags.items():
                    if entered:
                        rewards[aid] += taste_reward

        if r_step != 0.0 or r_idle != 0.0:
            moved = getattr(self, "_moved_flags", {}) or {}
            for aid in range(self.num_agents):
                rewards[aid] += (r_step if moved.get(aid, False) else r_idle)

        # --- consume resources ---
        for loc in consumed:
            self.world.add(loc, Empty())

        # --- NOTE: INTERACT (action==5) should be handled in the *message* system, not here ---
        return rewards

    # -----------------------------
    # Timers, regeneration, transitions
    # -----------------------------
    def _update_frozen_timers(self) -> None:
        for aid in list(self.agent_frozen.keys()):
            self.agent_frozen[aid] -= 1
            if self.agent_frozen[aid] <= 0:
                del self.agent_frozen[aid]

    def _regeneration_step(self) -> None:
        # Dynamic layer: Empty cells may spawn per Sand rules (via world.transition)
        for y in range(self.world.height):
            for x in range(self.world.width):
                if isinstance(self.world.observe((y, x, self.world.dynamic_layer)), Empty):
                    # Regeneration triggered through Sand.transition in _transition_step
                    pass

    def _transition_step(self) -> None:
        # Terrain transitions (e.g., Sand respawn timers)
        for y in range(self.world.height):
            for x in range(self.world.width):
                terr = self.world.observe((y, x, self.world.terrain_layer))
                if getattr(terr, "has_transitions", False):
                    terr.transition(self.world)

                beam = self.world.observe((y, x, self.world.beam_layer))
                if getattr(beam, "has_transitions", False):
                    beam.transition(self.world)

    # -----------------------------
    # Messaging
    # -----------------------------
    def positions_dict(self) -> Dict[int, Tuple[int, int]]:
        return {aid: (pos[0], pos[1]) for aid, pos in self.agent_positions.items()}

    def deliver_messages(self) -> None:
        self.message_bus.deliver(positions=self.positions_dict(), radius=self.bus_radius)

    # -----------------------------
    # Observations
    # -----------------------------
    def _get_observations(self) -> Dict[int, Dict[str, Any]]:
        obs: Dict[int, Dict[str, Any]] = {}
        for aid in range(self.num_agents):
            y, x, _ = self.agent_positions[aid]
            inbox = self.message_bus.inbox_for(aid) if self.message_bus else []
            obs[aid] = {
                "turn": self.turn,
                "pos": (y, x),
                "nearby": self._get_nearby_info(aid),
                "frozen": self.agent_frozen.get(aid, 0),
                "inbox": inbox,
            }
        return obs

    def _get_nearby_info(self, agent_id: int) -> Dict[str, Any]:
        y, x, _ = self.agent_positions[agent_id]
        neighbors = [(y - 1, x, "up"), (y + 1, x, "down"), (y, x - 1, "left"), (y, x + 1, "right")]

        info = {
            "ally_adjacent": False,
            "stag_adjacent": False,
            "hare_dir": None,
            "on_hare": isinstance(self.world.observe((y, x, self.world.dynamic_layer)), HareResource),
            "on_stag": isinstance(self.world.observe((y, x, self.world.dynamic_layer)), StagResource),
        }

        agent_cells = {(py, px) for (py, px, _) in self.agent_positions.values()}

        for ny, nx, d in neighbors:
            if not (0 <= ny < self.world.height and 0 <= nx < self.world.width):
                continue

            if (ny, nx) in agent_cells:
                info["ally_adjacent"] = True

            ent = self.world.observe((ny, nx, self.world.dynamic_layer))
            if isinstance(ent, StagResource):
                info["stag_adjacent"] = True
            elif isinstance(ent, HareResource) and info["hare_dir"] is None:
                info["hare_dir"] = d

        return info

    # -----------------------------
    # Utilities
    # -----------------------------
    def is_valid_location(self, loc: Tuple[int, int]) -> bool:
        y, x = loc
        if not (0 <= y < self.world.height and 0 <= x < self.world.width):
            return False
        terrain = self.world.observe((y, x, self.world.terrain_layer))
        return bool(getattr(terrain, "passable", False))

    # For visualization / debugging
    def hare_positions(self) -> List[Tuple[int, int]]:
        out = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                if isinstance(self.world.observe((y, x, self.world.dynamic_layer)), HareResource):
                    out.append((y, x))
        return out

    def stag_positions(self) -> List[Tuple[int, int]]:
        out = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                if isinstance(self.world.observe((y, x, self.world.dynamic_layer)), StagResource):
                    out.append((y, x))
        return out

    def reward_rules_json(self) -> dict:
        return StagHuntEnv.reward_rules_from_config(self.config)

    # NEW: static helper usable without an instance
    @staticmethod
    def reward_rules_from_config(config: dict) -> dict:
        w = config.get("world", {}) if isinstance(config, dict) else {}
        return {
            "name": "staghunt_v2",
            "params": {
                "stag_quorum_k": int(w.get("stag_quorum_k", 2)),
                "hare_exclusive": bool(w.get("hare_exclusive", True)),
                "share_stag_reward": bool(w.get("share_stag_reward", False)),
                "hare_reward": float(w.get("hare_reward", 1.0)),
                "stag_reward": float(w.get("stag_reward", 5.0)),
                "sucker_payoff": float(w.get("sucker_payoff", 0.0)),
                "taste_reward": float(w.get("taste_reward", 0.0)),
                "r_step": float(w.get("r_step", 0.0)),
                "r_idle": float(w.get("r_idle", 0.0)),
            },
            "rules": [
                "HARE: Any agent *standing* on a hare tile gains hare_reward "
                "(exclusive winner if hare_exclusive=True; else all on tile).",
                "STAG: Any set of agents *standing* on the same stag tile gains stag_reward "
                f"if and only if number of agents in the group is larger than or equal to stag_quorum_k; "
                "otherwise each may get sucker_payoff.",
                "INTERACT (action 5) only affects **chat/message delivery**, not rewards.",
                "Taste shaping: award taste_reward when entering a resource tile this turn (optional).",
                "Movement shaping: add r_step if moved this turn, else r_idle (optional).",
            ],
        }

    
    def has_neighbor_within_radius(self, agent_id: int, radius: int | None = None) -> bool:
        """Check if this agent has another agent within the given (vision) radius."""
        if agent_id not in self.agent_positions:
            return False
        ay, ax, _ = self.agent_positions[agent_id]
        r = int(radius or getattr(self, "vision_radius", 3))
        for oid, (oy, ox, _) in self.agent_positions.items():
            if oid == agent_id:
                continue
            if abs(ay - oy) <= r and abs(ax - ox) <= r:
                return True
        return False


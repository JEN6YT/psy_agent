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

# TODO: Add Attack_Valid boolean variable to agent observations to indicate if an attack would hit a resource.
# This would help agents decide when to attack, instead of choosing if it is valid to attack.
# Less geometric. 

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
from sorrel.examples.staghunt.entities import AttackBeam
from sorrel.examples.staghunt.entities import InteractionBeam, PunishBeam

ORIENTATION_VECTORS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # north (up)
    1: (0, 1),  # east (right)
    2: (1, 0),  # south (down)
    3: (0, -1),  # west (left)
}

VECTOR_TO_ORIENTATION: Dict[Tuple[int, int], int] = {
    (-1, 0): 0,  # north
    (0, 1): 1,   # east
    (1, 0): 2,   # south
    (0, -1): 3,  # west
}

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
      - No movement/taste shaping; rewards are from successful kills only
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
        self.world.environment = self  # back-reference
        self.world.config = config

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
        self.vision_radius: int = int(obs_cfg.get("vision_radius", 3))

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
                    return StagResource(
                        self.world.stag_reward,
                        self.world.stag_health,
                        regeneration_cooldown=self.world.stag_regeneration_cooldown,
                    )
                if rt == "hare":
                    return HareResource(
                        self.world.hare_reward,
                        self.world.hare_health,
                        regeneration_cooldown=self.world.hare_regeneration_cooldown,
                    )
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
            0: stay, 1: up, 2: right, 3: down, 4: left, 5: attack
        """
        self.world.current_turn = self.turn
        # initialize per-agent rewards for this step
        step_rewards: Dict[int, float] = {aid: 0.0 for aid in range(self.num_agents)}
        
        for agent in self.agents:
            pr = getattr(agent, "pending_reward", 0.0)
            if pr != 0.0:
                step_rewards[agent.agent_id] += pr
                agent.pending_reward = 0.0  

        # Movement (also sets movement/entry flags)
        self._handle_movement(actions)

        # Interactions & rewards from ATTACK actions
        self._handle_attacks(actions, step_rewards)

        # Timers/transitions/regeneration
        self._update_frozen_timers()
        self._regeneration_step()
        self._transition_step() # resource respawns, beam decay, etc.

        # Accumulate episode totals
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
        dynamic = self.world.observe((y, x, self.world.dynamic_layer))
        return bool(getattr(terrain, "passable", False)) and bool(
            getattr(dynamic, "passable", False)
        )

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

        for aid, pos in new_positions.items():
            agent = self.agents[aid]
            agent.location = pos
            self.agent_positions[aid] = pos


        # TODO: Implement attack_valid here!

        # Movement & entry flags (for shaping/taste)
        self._moved_flags = {}
        self._entered_resource_flags = {}
        for aid in range(self.num_agents):
            oy, ox, _ = old_positions[aid]
            ny, nx, _ = self.agent_positions[aid]
            moved = (oy, ox) != (ny, nx)
            self._moved_flags[aid] = moved

            agent = self.agents[aid]
            if moved:
                dy = ny - oy
                dx = nx - ox
                # Only update if agent supports orientation mapping
                orient_map = VECTOR_TO_ORIENTATION
                if (dy, dx) in orient_map:
                    agent.orientation = orient_map[(dy, dx)]
                    
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

    def _handle_attacks(
        self,
        actions: Dict[int, int],
        step_rewards: Dict[int, float],
    ) -> None:
        """
        Process ATTACK actions for all agents.

        For any aid with action == 5:
        - pay attack_cost
        - spawn beam according to agent.orientation
        - apply on_attack to resources
        - call handle_resource_defeat to share rewards
        - add attacking agent's immediate reward into step_rewards[aid]
        """
        world = self.world
        env = getattr(world, "environment", None)
        metrics = getattr(env, "metrics_collector", None)
        dynamic_layer = getattr(world, "dynamic_layer", None)

        attackers_by_resource: Dict[int, set[int]] = {}

        for aid, act in actions.items():
            if act != 5:
                continue  # not ATTACK

            agent = self.agents[aid]
            def _apply_attack(entity) -> None:
                if not isinstance(entity, (StagResource, HareResource)):
                    return
                is_stag = isinstance(entity, StagResource)
                # Default to allowing stag damage unless agent explicitly opts out.
                should_harm = (not is_stag) or getattr(agent, "can_hunt", True)

                if metrics is not None:
                    rtype = "stag" if is_stag else "hare"
                    metrics.collect_attack_metrics(agent, rtype, entity)

                if not should_harm:
                    return

                attackers_by_resource.setdefault(id(entity), set()).add(agent.agent_id)
                defeated = entity.on_attack(world, world.current_turn)
                if defeated:
                    attackers = attackers_by_resource.get(id(entity), {agent.agent_id})
                    reward_map = self.handle_resource_defeat(attackers, entity, world)
                    for rid, amount in reward_map.items():
                        step_rewards[rid] += amount

                    if metrics is not None:
                        rtype = "stag" if is_stag else "hare"
                        if hasattr(metrics, "collect_resource_defeat_event"):
                            metrics.collect_resource_defeat_event(
                                attackers=attackers,
                                reward_map=reward_map,
                                resource_type=rtype,
                                turn=getattr(world, "current_turn", None),
                            )
                        else:
                            metrics.collect_resource_defeat_metrics(
                                agent, reward_map.get(agent.agent_id, 0.0), rtype
                            )

            # cooldown check
            if getattr(agent, "attack_cooldown_timer", 0) > 0:
                agent.attack_cooldown_timer -= 1
                continue

            if getattr(agent, "health", None) is None:
                agent.health = getattr(world, "agent_health", 5)
            # If agent is out of HP, it cannot attack.
            if int(agent.health) <= 0:
                agent.health = max(0, int(agent.health))
                continue
            agent.health = max(0, int(agent.health) - 1)

            if metrics is not None:
                metrics.collect_agent_cost_metrics(agent, attack_cost=1)

            # spawn beam in front of this agent
            beam_locs = agent.spawn_attack_beam(world)

            if dynamic_layer is None:
                continue

            has_attack_target = False
            for (by, bx, _) in beam_locs:
                target = (by, bx, dynamic_layer)
                if not world.valid_location(target):
                    continue

                ent = world.observe(target)
                if isinstance(ent, (StagResource, HareResource)):
                    has_attack_target = True
                _apply_attack(ent)

            if not has_attack_target:
                attack_cost = float(getattr(world, "attack_cost", 0.05))
                step_rewards[aid] -= attack_cost

            agent.attack_cooldown_timer = getattr(world, "attack_cooldown", 3)


    def handle_resource_defeat(
        self,
        attackers: set[int],
        resource,
        world: StagHuntWorld,
    ) -> Dict[int, float]:
        """
        Handle reward sharing when a resource is defeated by this agent's attack beam.

        Returns
        -------
        Dict[int, float]
            Mapping of attacker_id -> reward share for this kill.
        """
        # ---------- identify resource type ----------
        if isinstance(resource, HareResource):
            rtype = "hare"
        elif isinstance(resource, StagResource):
            rtype = "stag"
        else:
            return {}

        # ---------- total reward (from config, like _handle_interactions) ----------
        cfg = getattr(world, "config", None)
        total = float(getattr(resource, "value", 0.0))

        if cfg is not None:
            if isinstance(cfg, dict):
                wcfg = cfg.get("world", {})
                hare_total = wcfg.get("hare_reward", None)
                stag_total = wcfg.get("stag_reward", None)
            else:
                wcfg = getattr(cfg, "world", cfg)
                hare_total = getattr(wcfg, "hare_reward", None)
                stag_total = getattr(wcfg, "stag_reward", None)

            if rtype == "hare" and hare_total is not None:
                total = float(hare_total)
            elif rtype == "stag" and stag_total is not None:
                total = float(stag_total)

        if total == 0.0:
            return {}

        if not attackers:
            return {}

        per_share = total / float(len(attackers))

        # ---------- reputation outcome (same as old _handle_interactions) ----------
        if rtype == "stag":
            outcome = "cooperated_stag" if len(attackers) >= 2 else "solo_stag"
        else:  # hare
            outcome = "solo_hare" if len(attackers) == 1 else "shared_hare"

        env = getattr(world, "environment", None)
        metrics = getattr(env, "metrics_collector", None)
        reward_map: Dict[int, float] = {}
        
        for agent in getattr(env, "agents", []):
            # Inventory
            inv = getattr(agent, "inventory", None)
            if isinstance(inv, dict):
                if agent.agent_id in attackers:
                    inv[rtype] = inv.get(rtype, 0) + 1

            # Reputation
            if hasattr(agent, "update_reputation_after_interaction"):
                if agent.agent_id not in attackers:
                    continue
                others = [aid for aid in attackers if aid != agent.agent_id]
                try:
                    agent.update_reputation_after_interaction(
                        other_agent_ids=others,
                        outcome=outcome,
                        reward=per_share,
                    )
                except TypeError:
                    pass

            # Reward assignment
            if agent.agent_id in attackers:
                reward_map[agent.agent_id] = per_share
                if metrics is not None and agent.agent_id != next(iter(attackers)):
                    metrics.collect_shared_reward_metrics(agent, per_share)

        # Do NOT add `total` to world.total_reward here.
        return reward_map
    
    # def apply_attack_if_any(self, world, reward: float) -> float:
    #     """
    #     Beam-based attack:
    #     - pays attack_cost
    #     - spawns beam
    #     - applies on_attack to hit resources
    #     - if defeated, calls handle_resource_defeat to share rewards
    #     Returns the updated reward for this step.
    #     """
    #     # Only attack if cooldown is over
    #     if getattr(self, "attack_cooldown_timer", 0) > 0:
    #         # just decrement cooldown and return
    #         self.attack_cooldown_timer -= 1
    #         return reward

    #     # You decide when to actually fire (e.g., if action == attack)
    #     # Here we assume you have already decided to attack when calling this.
    #     attack_cost = getattr(world, "attack_cost", 0.05)
    #     reward -= attack_cost

    #     # Optional: log cost metrics
    #     env = getattr(world, "environment", None)
    #     metrics = getattr(env, "metrics_collector", None)
    #     if metrics is not None:
    #         metrics.collect_agent_cost_metrics(self, attack_cost=attack_cost)

    #     # Spawn visual beam and get beam locations (list of (y, x))
    #     beam_locs = self.spawn_attack_beam(world)

    #     dynamic_layer = getattr(world, "dynamic_layer", None)

    #     for (by, bx) in beam_locs:
    #         if dynamic_layer is None:
    #             continue
    #         target = (by, bx, dynamic_layer)
    #         if not world.valid_location(target):
    #             continue

    #         entity = world.observe(target)
    #         if isinstance(entity, (StagResource, HareResource)):
    #             # Hares always vulnerable; stags only if can_hunt
    #             is_stag = isinstance(entity, StagResource)
    #             should_harm = (not is_stag) or getattr(self, "can_hunt", False)

    #             # Always log attacks, even if they don't harm
    #             if metrics is not None:
    #                 rtype = "stag" if is_stag else "hare"
    #                 metrics.collect_attack_metrics(self, rtype, entity)

    #             if not should_harm:
    #                 continue

    #             # Apply damage
    #             defeated = entity.on_attack(world, world.current_turn)
    #             if defeated:
    #                 # Share reward via agent-side helper
    #                 shared_reward = self.handle_resource_defeat(entity, world)
    #                 reward += shared_reward

    #                 # Log resource defeat
    #                 if metrics is not None:
    #                     rtype = "stag" if is_stag else "hare"
    #                     metrics.collect_resource_defeat_metrics(self, shared_reward, rtype)

    #     # Reset cooldown after attacking
    #     self.attack_cooldown_timer = getattr(world, "attack_cooldown", 3)
    #     return reward
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

                dyn = self.world.observe((y, x, self.world.dynamic_layer))
                if getattr(dyn, "has_transitions", False):
                    dyn.transition(self.world)

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
    def beam_tiles(
        self,
        pos_rc: Tuple[int, int],
        facing: int | str,
        beam_len: int,
    ) -> List[Tuple[int, int]]:
        r, c = pos_rc
        if isinstance(facing, str):
            f = facing.strip().upper()
            dir_map = {
                "NORTH": (-1, 0),
                "EAST": (0, 1),
                "SOUTH": (1, 0),
                "WEST": (0, -1),
            }
            dr, dc = dir_map.get(f, ORIENTATION_VECTORS.get(0))
        else:
            dr, dc = ORIENTATION_VECTORS.get(int(facing), (-1, 0))
        return [(r + dr * i, c + dc * i) for i in range(1, int(beam_len) + 1)]

    def compute_attack_valid(
        self,
        pos_rc: Tuple[int, int],
        facing: int | str,
        beam_len: int,
        targets: List[Dict[str, Any]],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        tiles = set(self.beam_tiles(pos_rc, facing, beam_len))
        hittable: List[Dict[str, Any]] = []
        for tgt in targets:
            pos = tgt.get("pos_rc")
            if pos is None:
                continue
            try:
                pos_tuple = (int(pos[0]), int(pos[1]))
            except Exception:
                continue
            if pos_tuple in tiles:
                hittable.append(tgt)
        return bool(hittable), hittable

    def _get_observations(self) -> Dict[int, Dict[str, Any]]:
        obs: Dict[int, Dict[str, Any]] = {}
        for aid in range(self.num_agents):
            y, x, _ = self.agent_positions[aid]
            agent = self.agents[aid]
            inbox = self.message_bus.inbox_for(aid) if self.message_bus else []
            inventory = getattr(agent, "inventory", {"hare": 0, "stag": 0})
            obs[aid] = {
                "turn": self.turn,
                "pos": (y, x),
                "nearby": self._get_nearby_info(aid),
                "frozen": self.agent_frozen.get(aid, 0),
                "inventory": inventory,
                "inbox": inbox,
            }
        return obs

    def _get_nearby_info(self, agent_id: int) -> Dict[str, Any]:
        y, x, _ = self.agent_positions[agent_id]
        h, w = self.world.height, self.world.width
        r = int(getattr(self, "vision_radius", 3))
        agent_cells = {(py, px) for (py, px, _) in self.agent_positions.values()}

        def _in_bounds(ny: int, nx: int) -> bool:
            return 0 <= ny < h and 0 <= nx < w

        def _dir_from_delta(dy: int, dx: int) -> str:
            if dy == 0 and dx == 0:
                return "here"
            if abs(dy) > abs(dx):
                return "north" if dy < 0 else "south"
            if abs(dx) > abs(dy):
                return "west" if dx < 0 else "east"
            ns = "north" if dy < 0 else "south"
            ew = "west" if dx < 0 else "east"
            return f"{ns}-{ew}"

        info = {
            "ally_adjacent": False,
            "stag_adjacent": False,
            "hare_dir": None,
            "hare_nearby_dir": None,
            "hare_nearby_dist": None,
            "stag_nearby_dir": None,
            "stag_nearby_dist": None,
            "agent_nearby": [],
            "nearest_agent_dist": None,
            "nearest_agent_dir": None,
            "agent_count": 0,
        }

        best_hare: Optional[Tuple[int, str]] = None
        best_stag: Optional[Tuple[int, str]] = None
        best_agent: Optional[Tuple[int, str]] = None

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = abs(dy) + abs(dx)
                if dist > r:
                    continue
                ny, nx = y + dy, x + dx
                if not _in_bounds(ny, nx):
                    continue
                if dist == 1 and (ny, nx) in agent_cells:
                    info["ally_adjacent"] = True
                if dist != 0 and (ny, nx) in agent_cells:
                    direction = _dir_from_delta(dy, dx)
                    info["agent_nearby"].append(
                        {
                            "pos": (ny, nx),
                            "distance": dist,
                            "direction": direction,
                            "is_adjacent": dist == 1,
                        }
                    )
                    if best_agent is None or dist < best_agent[0]:
                        best_agent = (dist, direction)

                ent = self.world.observe((ny, nx, self.world.dynamic_layer))
                if dist == 1:
                    if isinstance(ent, StagResource):
                        info["stag_adjacent"] = True
                    elif isinstance(ent, HareResource) and info["hare_dir"] is None:
                        info["hare_dir"] = "up" if dy == -1 else "down" if dy == 1 else "left" if dx == -1 else "right"

                if isinstance(ent, HareResource):
                    if best_hare is None or dist < best_hare[0]:
                        best_hare = (dist, _dir_from_delta(dy, dx))
                elif isinstance(ent, StagResource):
                    if best_stag is None or dist < best_stag[0]:
                        best_stag = (dist, _dir_from_delta(dy, dx))

        if best_hare is not None:
            info["hare_nearby_dist"], info["hare_nearby_dir"] = best_hare
        if best_stag is not None:
            info["stag_nearby_dist"], info["stag_nearby_dir"] = best_stag
        if info["agent_nearby"]:
            info["agent_nearby"].sort(key=lambda a: a["distance"])
            info["agent_count"] = len(info["agent_nearby"])
        if best_agent is not None:
            info["nearest_agent_dist"], info["nearest_agent_dir"] = best_agent

        return info

    # -----------------------------
    # Utilities
    # -----------------------------
    def is_valid_location(self, loc: Tuple[int, ...]) -> bool:
        if len(loc) == 3:
            y, x, _ = loc
        else:
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

    def hare_states(self) -> List[Dict[str, int]]:
        out: List[Dict[str, int]] = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                ent = self.world.observe((y, x, self.world.dynamic_layer))
                if isinstance(ent, HareResource):
                    hp = int(getattr(ent, "health", 0))
                    out.append({"y": int(y), "x": int(x), "hp": hp})
        return out

    def stag_positions(self) -> List[Tuple[int, int]]:
        out = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                if isinstance(self.world.observe((y, x, self.world.dynamic_layer)), StagResource):
                    out.append((y, x))
        return out

    def stag_states(self) -> List[Dict[str, int]]:
        out: List[Dict[str, int]] = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                ent = self.world.observe((y, x, self.world.dynamic_layer))
                if isinstance(ent, StagResource):
                    hp = int(getattr(ent, "health", 0))
                    out.append({"y": int(y), "x": int(x), "hp": hp})
        return out

    def beam_positions(self) -> List[Tuple[int, int, str]]:
        out = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                ent = self.world.observe((y, x, self.world.beam_layer))
                if isinstance(ent, AttackBeam):
                    out.append((y, x, "attack"))
                elif isinstance(ent, PunishBeam):
                    out.append((y, x, "punish"))
                elif isinstance(ent, InteractionBeam):
                    out.append((y, x, "interaction"))
        return out

    def reward_rules_json(self) -> dict:
        return StagHuntEnv.reward_rules_from_config(self.config)

    # NEW: static helper usable without an instance
    @staticmethod
    def reward_rules_from_config(config: dict) -> dict:
        w = config.get("world", {}) if isinstance(config, dict) else {}
        hare_reward = float(w.get("hare_reward", 2.0))
        stag_reward = float(w.get("stag_reward", 5.0))

        agent_health = float(w.get("agent_health", 12))
        hare_health = float(w.get("hare_health", 1))
        stag_health = float(w.get("stag_health", 6))
        attack_cost = float(w.get("attack_cost", 0.05))

        regeneration_rate = float(w.get("regeneration_rate", 0.05))
        stag_regeneration_cooldown = int(w.get("stag_regeneration_cooldown", 5))
        hare_regeneration_cooldown = int(w.get("hare_regeneration_cooldown", 3))

        return {
            "name": "staghunt_beam_kill_v2",
            "params": {
                "hare_reward": hare_reward,
                "stag_reward": stag_reward,
                "agent_health": agent_health,
                "hare_health": hare_health,
                "stag_health": stag_health,
                "attack_cost": attack_cost,
            },
            "rules": [
                "When you choose to attack, the beam only fires forward and only hits within beam length.",
                "Attacking with no hare/stag in beam is penalized.",
                "Resource reward is hare_reward or stag_reward; if multiple attackers, split reward evenly.",
                "Each resource (hare/stag) has HP. Each agent also has HP.",
                f"Hare has {hare_health} HP; stag has {stag_health} HP.",
                f"Agent starts with {agent_health} HP.",
                "On a valid hit, hare/stag HP decreases by 1.",
                "If resource HP <= 0, resource is defeated and removed; attackers share reward.",
                f"If a hare is defeated, each attacker gains {hare_reward} / num_attackers reward.",
                f"If a stag is defeated, each attacker gains {stag_reward} / num_attackers reward.",
                f"When you attack (even if you hit), your HP decreases by 1, and your reward decreases by {attack_cost}.",   
                "If hare/stag has HP below max, it can regenerate over time, ",
                f"since last attck, starting after {hare_regeneration_cooldown} turns for hare and {stag_regeneration_cooldown} turns for stag, ",
                f"with a regeneration rate of {regeneration_rate} per turn.",
            ],
            "tips": [
                "Stag is higher value than hare, and coordination increases expected reward.",
                "Regeneration means partial progress can be lost, so consider focusing on finishing off targets.",
                "Hare is safe individually; stag requires cooperation to avoid wasted attacks.",
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
            if (abs(ay - oy) + abs(ax - ox)) <= r:
                return True
        return False

    def get_agent_positions(self):
        """
        Returns a dict mapping agent_id -> (y, x, orientation).
        """
        return dict(self.agent_positions)

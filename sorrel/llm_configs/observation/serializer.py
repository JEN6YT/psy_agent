# sorrel/llm_configs/observation/serializer.py
from __future__ import annotations

from typing import Sequence, Optional, Dict, Any, List, Tuple, Mapping
from dataclasses import dataclass

import numpy as np

from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds import Gridworld


class LLMObservationParser(ObservationSpec[str]):
    """
    Observation parser that converts gridworld observations to natural language.

    It produces short text suitable for LLM agents:
      - your position (relative or absolute)
      - nearby entities w/ distance & direction (configurable)
      - optional detailed world info in full-view mode

    NOTE: ObservationSpec.__init__ calls self.generate_map(...), so any fields
    used by generate_map() must be set BEFORE super().__init__().
    """

    def __init__(
        self,
        entity_list: List[str],
        full_view: bool,
        vision_radius: Optional[int] = None,
        env_dims: Optional[Sequence[int]] = None,
        include_distances: bool = True,
        include_directions: bool = True,
        include_coordinates: bool = False,
        use_relative_positions: bool = True,
        verbose_level: int = 1,
        custom_descriptions: Optional[Dict[str, str]] = None,
    ):
        # --- set fields USED by generate_map() BEFORE calling parent ---
        self.custom_descriptions: Dict[str, str] = dict(custom_descriptions or {})

        # --- set other subclass fields (safe before parent too) ---
        self.include_distances = include_distances
        self.include_directions = include_directions
        self.include_coordinates = include_coordinates
        self.use_relative_positions = use_relative_positions
        self.verbose_level = verbose_level

        # Parent init will call our generate_map()
        super().__init__(entity_list, full_view, vision_radius, env_dims)

        # LLMs consume text; keep a conventional input_size tuple
        self.input_size = (1,)

    # -------------------- hooks / core utils --------------------

    def generate_map(self, entity_list: List[str]) -> Dict[str, str]:
        """
        Build textual labels for entity kinds.
        Priority: custom_descriptions -> heuristics -> fallback name.
        """
        mp: Dict[str, str] = {}
        for entity in entity_list:
            if entity in self.custom_descriptions:
                mp[entity] = self.custom_descriptions[entity]
            elif entity == "EmptyEntity" or entity == "Empty":
                mp[entity] = "empty space"
            elif "Wall" in entity:
                mp[entity] = "wall"
            elif "Agent" in entity:
                mp[entity] = "agent"
            elif "Resource" in entity or "Item" in entity or entity in {"Hare", "Stag"}:
                mp[entity] = entity.replace("Entity", "").lower()
            else:
                mp[entity] = entity.replace("Entity", "").lower()
        return mp

    # -------------------- public API --------------------

    def observe(
        self,
        world: Gridworld,
        location: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Produce a natural-language observation.

        Args:
            world: Gridworld object with .map and (optionally) .agents/.resources
            location: (row, col) if not full_view
        """
        if not self.full_view and location is None:
            raise TypeError("location must be provided when full_view is False.")

        if self.full_view:
            return self._observe_full_world(world)
        return self._observe_local_view(world, location)  # type: ignore[arg-type]

    # -------------------- full-view --------------------

    def _observe_full_world(self, world: Gridworld) -> str:
        parts: List[str] = []

        h, w = world.map.shape[:2]
        parts.append(f"World: {w}×{h} grid")

        if getattr(world, "agents", None):
            parts.append(f"\nAgents ({len(world.agents)} total):")
            for i, agent in enumerate(world.agents):
                desc = self._describe_agent(agent, i, None)
                if desc:
                    parts.append(f"  {desc}")

        if getattr(world, "resources", None):
            parts.append(f"\nResources ({len(world.resources)} total):")
            for r in world.resources:
                rd = self._describe_entity(r, None)
                if rd:
                    parts.append(f"  {rd}")

        if self.verbose_level >= 2:
            terr = self._describe_terrain(world)
            if terr:
                parts.append(f"\n{terr}")

        return "\n".join(parts)

    # -------------------- local-view --------------------

    def _observe_local_view(
        self,
        world: Gridworld,
        location: Tuple[int, int],
    ) -> str:
        parts: List[str] = []
        r, c = location  # row, col

        if self.include_coordinates:
            parts.append(f"Position: ({r}, {c})")
        else:
            parts.append("You are at your current location.")

        nearby = self._get_nearby_entities(world, location)

        if not nearby:
            parts.append("\nYou see nothing of interest nearby.")
        else:
            by_dist = self._organize_by_distance(nearby)
            if self.verbose_level == 0:
                parts.append(f"\n{len(nearby)} entities visible.")
            elif self.verbose_level == 1:
                parts.append(self._format_normal_view(by_dist, location))
            else:
                parts.append(self._format_detailed_view(by_dist, location))

        if hasattr(world, "turn") and self.verbose_level >= 1:
            parts.append(f"\nTurn: {world.turn}")

        return "\n".join(parts)

    # -------------------- helpers --------------------

    def _get_nearby_entities(
        self,
        world: Gridworld,
        location: Tuple[int, int],
    ) -> List[Tuple[Any, Tuple[int, int], int, str]]:
        """Scan within vision_radius (Manhattan) for non-empty cells/entities."""
        results: List[Tuple[Any, Tuple[int, int], int, str]] = []
        r0, c0 = location
        h, w = world.map.shape[:2]
        vr = int(self.vision_radius)

        for r in range(max(0, r0 - vr), min(h, r0 + vr + 1)):
            for c in range(max(0, c0 - vr), min(w, c0 + vr + 1)):
                if (r, c) == (r0, c0):
                    continue
                dist = abs(r - r0) + abs(c - c0)
                if dist > vr:
                    continue
                ent = self._get_entity_at(world, (r, c))
                if ent is None or self._is_empty(ent):
                    continue
                direction = self._get_direction(r0, c0, r, c)
                results.append((ent, (r, c), dist, direction))
        return results

    def _get_entity_at(self, world: Gridworld, pos: Tuple[int, int]) -> Optional[Any]:
        """Return agent/object/cell at pos, if any."""
        # agents (with .location as (row, col))
        if getattr(world, "agents", None):
            for ag in world.agents:
                if getattr(ag, "location", None) == pos:
                    return ag

        r, c = pos
        if 0 <= r < world.map.shape[0] and 0 <= c < world.map.shape[1]:
            return world.map[r, c]
        return None

    def _is_empty(self, entity: Any) -> bool:
        if entity is None:
            return True
        # If map stores glyphs like ".", treat them as empty
        if isinstance(entity, str):
            return entity in {".", " ", "", "Empty", "EmptyEntity"}
        # Object case
        t = type(entity).__name__
        return t in {"EmptyEntity", "NoneType"}

    def _describe_entity(
        self,
        entity: Any,
        observer_location: Optional[Tuple[int, int]] = None,
    ) -> str:
        etype = type(entity).__name__ if not isinstance(entity, str) else entity
        base = self.entity_map.get(etype, etype)

        if hasattr(entity, "location") and getattr(entity, "location") is not None:
            loc = getattr(entity, "location")
            loc_desc = self._format_location(loc, observer_location)
            return f"{base} {loc_desc}".strip()
        return base

    def _describe_agent(
        self,
        agent: Any,
        agent_id: int,
        observer_location: Optional[Tuple[int, int]] = None,
    ) -> str:
        parts = [f"Agent {agent_id}"]
        if hasattr(agent, "location") and agent.location:
            parts.append(self._format_location(agent.location, observer_location))
        if self.verbose_level >= 2:
            if hasattr(agent, "resources"):
                parts.append(f"(resources: {agent.resources})")
            if hasattr(agent, "health"):
                parts.append(f"(health: {agent.health})")
        return " ".join(p for p in parts if p)

    def _format_location(
        self,
        location: Tuple[int, int],
        observer_location: Optional[Tuple[int, int]] = None,
    ) -> str:
        parts: List[str] = []
        if observer_location and self.use_relative_positions:
            r1, c1 = observer_location
            r2, c2 = location
            dist = abs(r2 - r1) + abs(c2 - c1)
            direction = self._get_direction(r1, c1, r2, c2)
            if self.include_distances and self.include_directions:
                parts.append(f"{dist} steps {direction}")
            elif self.include_distances:
                parts.append(f"{dist} steps away")
            elif self.include_directions:
                parts.append(f"to the {direction}")

        if self.include_coordinates:
            parts.append(f"at ({location[0]}, {location[1]})")

        return " ".join(parts)

    def _get_direction(self, r1: int, c1: int, r2: int, c2: int) -> str:
        dr, dc = r2 - r1, c2 - c1
        if dr == 0 and dc == 0:
            return "here"
        if abs(dr) > abs(dc):
            return "south" if dr > 0 else "north"
        if abs(dc) > abs(dr):
            return "east" if dc > 0 else "west"
        ns = "south" if dr > 0 else "north"
        ew = "east" if dc > 0 else "west"
        return f"{ns}-{ew}"

    def _organize_by_distance(
        self,
        nearby: List[Tuple[Any, Tuple[int, int], int, str]],
    ) -> Dict[int, List[Tuple[Any, Tuple[int, int], str]]]:
        by: Dict[int, List[Tuple[Any, Tuple[int, int], str]]] = {}
        for ent, pos, dist, direction in nearby:
            by.setdefault(int(dist), []).append((ent, pos, direction))
        return by

    def _format_normal_view(
        self,
        by_distance: Dict[int, List[Tuple[Any, Tuple[int, int], str]]],
        observer_location: Tuple[int, int],
    ) -> str:
        parts = ["\nNearby:"]
        for dist in sorted(by_distance):
            parts.append("  Adjacent:" if dist == 1 else f"  {dist} steps away:")
            for ent, pos, direction in by_distance[dist]:
                parts.append(f"    - {self._describe_entity(ent, observer_location)}")
        return "\n".join(parts)

    def _format_detailed_view(
        self,
        by_distance: Dict[int, List[Tuple[Any, Tuple[int, int], str]]],
        observer_location: Tuple[int, int],
    ) -> str:
        parts = ["\nDetailed view:"]
        for dist in sorted(by_distance):
            for ent, pos, direction in by_distance[dist]:
                etype = type(ent).__name__ if not isinstance(ent, str) else ent
                base = self.entity_map.get(etype, etype)
                details = [
                    f"  - {base}",
                    f"    Direction: {direction}",
                    f"    Distance: {dist}",
                    f"    Coordinates: {pos}",
                ]
                if hasattr(ent, "__dict__"):
                    for k, v in ent.__dict__.items():
                        if not k.startswith("_") and k != "location":
                            details.append(f"    {k}: {v}")
                parts.extend(details)
        return "\n".join(parts)

    def _describe_terrain(self, world: Gridworld) -> str:
        counts: Dict[str, int] = {}
        h, w = world.map.shape[:2]
        for r in range(h):
            for c in range(w):
                ent = world.map[r, c]
                etype = type(ent).__name__ if not isinstance(ent, str) else ent
                counts[etype] = counts.get(etype, 0) + 1
        parts = ["Terrain:"]
        for etype, cnt in counts.items():
            if etype not in {"EmptyEntity", "Empty", ".", " ", ""}:
                desc = self.entity_map.get(etype, etype)
                parts.append(f"  {desc}: {cnt}")
        return "\n".join(parts)


# ---------------------------- factory (matches your call site) ----------------------------

def create_llm_observation_parser(
    entity_list: List[str],
    vision_radius: int = 5,
    style: str = "concise",
) -> LLMObservationParser:
    """
    Convenience factory. Styles:
      - "minimal":    terse counts only
      - "concise":    default; names + relative positions
      - "detailed":   adds coords + per-entity attributes
    """
    styles: Dict[str, Dict[str, Any]] = {
        "minimal": {
            "verbose_level": 0,
            "include_distances": False,
            "include_directions": False,
            "include_coordinates": False,
        },
        "concise": {
            "verbose_level": 1,
            "include_distances": True,
            "include_directions": True,
            "include_coordinates": False,
        },
        "detailed": {
            "verbose_level": 2,
            "include_distances": True,
            "include_directions": True,
            "include_coordinates": True,
        },
    }
    cfg = styles.get(style, styles["concise"])
    return LLMObservationParser(
        entity_list=entity_list,
        full_view=False,
        vision_radius=vision_radius,
        **cfg,
    )

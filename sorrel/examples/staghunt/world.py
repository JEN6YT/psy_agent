"""The world definition for the Stag Hunt game.

This module defines a custom :class:`Gridworld` subclass for the stag hunt
social dilemma environment.  The world contains two layers: a bottom
terrain layer consisting of walls, empty spaces and designated spawn
locations, and a top layer containing all dynamic entities such as
resources and agents.  The world is parametrised by a configuration
object specifying the board dimensions, resource density, taste reward,
destroyable health for resources and other hyperparameters relevant to
the stag hunt mechanics.  See the accompanying design spec for a full
description of the environment rules.

The ``StagHuntWorld`` simply stores these hyperparameters; the logic for
resource regeneration, agent actions and interactions lives in the
entity and agent classes and in the environment wrapper.  The default
entity for empty cells is provided when constructing the world.
"""

from __future__ import annotations

try:
    # Optional dependency used in the original sorrel examples.  If
    # OmegaConf is unavailable, we fall back to treating the config as a
    # standard dictionary.
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

from typing import Any

from sorrel.examples.staghunt.map_generator import MapBasedWorldGenerator
from sorrel.worlds import Gridworld


class StagHuntWorld(Gridworld):
    """Gridworld implementation for the stag hunt arena.

    Parameters
    ----------
    config : dict or DictConfig or attribute-style object
        A configuration specifying the dimensions of the world and
        hyperparameters controlling the stag hunt mechanics. Expected keys under
        ``config['world']`` or attributes under ``config.world`` include:

        - ``height`` (int): rows in the grid.
        - ``width`` (int): columns in the grid.
        - ``num_agents`` (int): number of agents to spawn.
        - ``resource_density`` (float): prob. an empty cell spawns a resource.
        - ``taste_reward`` (float): intrinsic reward for collecting a resource.
        - ``destroyable_health`` (int): zap hits required to destroy a resource.
        - ``beam_length`` (int), ``beam_radius`` (int), ``beam_cooldown`` (int)
        - ``respawn_lag`` (int), ``respawn_delay`` (int)
        - ``payoff_matrix`` (list[list[int]])
        - ``generation_mode`` (str): "random" or "ascii_map"
        - ``ascii_map_file`` (str): path to ASCII map when using "ascii_map"

    default_entity : Entity
        The entity used to fill empty spaces on world creation.
    """

    def __init__(self, config: dict | Any, default_entity) -> None:
        """Initialise the world with values from a configuration."""
        # -------- robust world_cfg getter (OmegaConf, dict, or attr-style) --------
        if OmegaConf is not None and isinstance(config, DictConfig):  # type: ignore[arg-type]
            world_cfg = config.world  # type: ignore[attr-defined]
        elif isinstance(config, dict):
            world_cfg = config.get("world", {}) or {}
        elif hasattr(config, "world"):
            world_cfg = getattr(config, "world")
        else:
            world_cfg = {}

        def _get_wc(cfg: Any, key: str, default: Any) -> Any:
            """Return world config value for key; supports OmegaConf, dict, attrs."""
            if OmegaConf is not None and isinstance(config, DictConfig):  # type: ignore[arg-type]
                return getattr(cfg, key, default)
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        # -------- choose mode and determine (height, width) before allocating grid --------
        generation_mode = _get_wc(world_cfg, "generation_mode", "random")

        if generation_mode == "ascii_map":
            map_file = _get_wc(world_cfg, "ascii_map_file", None)
            if not map_file:
                raise ValueError("ascii_map_file required when generation_mode is 'ascii_map'")
            self.map_generator = MapBasedWorldGenerator(map_file)
            map_data = self.map_generator.parse_map()
            height, width = map_data.dimensions  # e.g., (9, 12)
        else:
            # fall back to explicit dims for random mode
            height = int(_get_wc(world_cfg, "height", 11))
            width  = int(_get_wc(world_cfg, "width", 11))
            self.map_generator = None

        # number of layers: bottom terrain, middle dynamic, top beam
        layers = 3
        super().__init__(height, width, layers, default_entity)

        # Define layer indices for clarity
        self.terrain_layer = 0
        self.dynamic_layer = 1
        self.beam_layer = 2

        # -------- read hyperparameters with the same robust getter --------
        def get_world_param(key: str, default: Any) -> Any:
            return _get_wc(world_cfg, key, default)

        self.num_agents: int = int(get_world_param("num_agents", 2))
        self.resource_density: float = float(get_world_param("resource_density", 0.05))
        self.taste_reward: float = float(get_world_param("taste_reward", 0.1))
        self.destroyable_health: int = int(get_world_param("destroyable_health", 3))
        self.respawn_lag: int = int(get_world_param("respawn_lag", 10))
        self.beam_length: int = int(get_world_param("beam_length", 3))
        self.beam_radius: int = int(get_world_param("beam_radius", 1))
        self.beam_cooldown: int = int(get_world_param("beam_cooldown", 3))
        self.freeze_duration: int = int(get_world_param("freeze_duration", 5))
        self.respawn_delay: int = int(get_world_param("respawn_delay", 10))
        self.payoff_matrix: list[list[int]] = [
            list(row) for row in get_world_param("payoff_matrix", [[4, 0], [2, 2]])
        ]

        # record spawn points; to be populated/overridden by the environment
        self.agent_spawn_points: list[tuple[int, int, int]] = [
            (2, 2, self.dynamic_layer),
            (3, 3, self.dynamic_layer),
            (4, 4, self.dynamic_layer),
            (5, 5, self.dynamic_layer),
        ]
        self.resource_spawn_points: list[tuple[int, int, int]] = []

    def reset_spawn_points(self) -> None:
        """Clear the list of spawn points. Called during environment reset."""
        self.agent_spawn_points = [
            (2, 2, self.dynamic_layer),
            (3, 3, self.dynamic_layer),
            (4, 4, self.dynamic_layer),
            (5, 5, self.dynamic_layer),
        ]
        self.resource_spawn_points = []

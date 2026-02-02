from __future__ import annotations
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, List, Dict, Any, Sequence, Tuple, Union
import yaml
import json
from pathlib import Path

# --------------------------- Dict-like mixin ---------------------------

class ConfigMixin:
    """Adds dict-like conveniences to dataclass configs."""

    # Simple getattr-based .get()
    def get(self, key: str, default: Any = None) -> Any:
        """
        Support:
          cfg.get("world")
          cfg.get("world.vision_radius", 5)  # dotted path
          cfg.get("agents.0.role")           # list index in dotted path
        """
        if not isinstance(key, str):
            raise TypeError("key must be a string")

        # Dotted path resolution
        if "." in key:
            cur: Any = self
            for part in key.split("."):
                if isinstance(cur, list):
                    try:
                        idx = int(part)
                    except ValueError:
                        return default
                    if 0 <= idx < len(cur):
                        cur = cur[idx]
                    else:
                        return default
                else:
                    cur = getattr(cur, part, None)
                if cur is None:
                    return default
            return cur
        # Single attribute
        return getattr(self, key, default)

    # Optional: indexing like a dict (cfg["world"])
    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    # Convert to nested dict
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Pretty
    def __repr__(self) -> str:
        cname = self.__class__.__name__
        return f"{cname}({self.to_dict()})"


# --------------------------- Dataclasses ---------------------------

@dataclass
class LLMConfig(ConfigMixin):
    """Configuration for LLM model."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature: float = 0.7
    max_new_tokens: int = 256
    verbose: bool = False

    # HF-specific kwargs
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig(ConfigMixin):
    """Configuration for individual agent."""
    agent_id: int
    role: str = "strategic agent"
    initial_location: Optional[tuple[int, int]] = None  # None = use spawn point from map
    communication_enabled: bool = True
    communication_range: Optional[int] = None  # None = use vision_radius

    # Model override (if different from default)
    llm_config: Optional[LLMConfig] = None


@dataclass
class ObservationConfig(ConfigMixin):
    """Configuration for observation/vision."""
    vision_radius: int = 5
    full_view: bool = False
    include_distances: bool = True
    include_directions: bool = True
    include_coordinates: bool = False
    use_relative_positions: bool = True
    verbose_level: int = 1  # 0=minimal, 1=normal, 2=detailed


@dataclass
class MemoryConfig(ConfigMixin):
    """Configuration for agent memory systems."""
    memory_size: int = 1000
    episodic_capacity: int = 512
    recent_steps_in_prompt: int = 6
    top_agents_in_prompt: int = 3
    fine_tuning_enabled: bool = False


@dataclass
class WorldConfig(ConfigMixin):
    """Configuration for the game world (Sorrel framework compatible)."""
    # Generation mode
    generation_mode: str = "random"  # "random" or "ascii_map"
    ascii_map_file: Optional[str] = None  # Required if generation_mode="ascii_map"

    # Dimensions (used for random generation, overridden by map for ascii_map)
    width: int = 20
    height: int = 20
    max_turns: int = 100
    num_agents: int = 2  # Number of agents (validated against map spawn points)

    # Resource parameters (Sorrel framework)
    resource_density: float = 0.05  # Probability of resource spawning in empty cells
    taste_reward: float = 0.1       # Small reward when stepping on resource
    destroyable_health: int = 3     # Hits needed to destroy a resource
    respawn_lag: int = 10           # Turns before resource can respawn

    # Interaction parameters (beams etc.)
    beam_length: int = 3
    beam_radius: int = 1
    beam_cooldown: int = 3
    freeze_duration: int = 5
    respawn_delay: int = 10
    attack_cooldown: int = 1

    # NEW: explicit radii for attacks and agent-agent interaction beams
    attack_radius: int = 3            # max Chebyshev distance to attack stag/hare
    interaction_radius: int = 3       # radius for interaction beams / chat

    # NEW: attack cost (used in env._handle_interactions)
    attack_cost: float = 0.05

    # Payoff matrix for stag hunt [row player perspective]
    payoff_matrix: List[List[int]] = field(default_factory=lambda: [[4, 0], [2, 2]])

    # Game-specific parameters (for backwards compatibility)
    game_type: str = "staghunt"
    stag_reward: float = 5.0
    hare_reward: float = 1.0
    sucker_payoff: float = 0.0

    stag_quorum_k: int = 2
    require_interact: bool = False      # primary rewards require ACTION==INTERACT
    hare_exclusive: bool = True         # exactly one interactor gets hare
    share_stag_reward: bool = False     # if True, split r_stag by group size
    r_step: float = 0.0                 # per move
    r_idle: float = 0.0                 # per stay

    # NEW: health regeneration parameters for resources
    # base rate; stag/hare can use different multipliers in their transition
    # Integer-friendly regen; use cooldowns to slow.
    health_regeneration_rate: float = 1.0
    stag_health: int = 6
    hare_health: int = 1
    agent_health: int = 12
    stag_regeneration_cooldown: int = 5
    hare_regeneration_cooldown: int = 5


@dataclass
class ExperimentConfig(ConfigMixin):
    """Main experiment configuration (Sorrel-compatible)."""
    # Experiment metadata
    experiment_name: str = "staghunt_experiment"
    seed: int = 42
    num_episodes: int = 10
    save_results: bool = True
    output_dir: str = "results"

    # Component configs
    world: WorldConfig = field(default_factory=WorldConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Agents configuration
    agents: List[AgentConfig] = field(default_factory=list)

    # Logging
    log_level: str = "INFO"
    log_frequency: int = 10  # Log every N turns
    save_trajectories: bool = True

    def __post_init__(self):
        """Initialize default agents if none provided."""
        if not self.agents:
            num_agents = self.world.num_agents
            self.agents = [
                AgentConfig(
                    agent_id=i,
                    role=f"agent {i}",
                    initial_location=None,
                )
                for i in range(num_agents)
            ]
        self.world.num_agents = len(self.agents)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        world_data = data.pop("world", {})
        obs_data = data.pop("observation", {})
        mem_data = data.pop("memory", {})
        llm_data = data.pop("llm", {})
        agents_data = data.pop("agents", [])

        world_config = WorldConfig(**world_data)
        obs_config = ObservationConfig(**obs_data)
        mem_config = MemoryConfig(**mem_data)
        llm_config = LLMConfig(**llm_data)

        agent_configs: List[AgentConfig] = []
        for agent_data in agents_data:
            agent_llm_data = agent_data.pop("llm_config", None)
            agent_llm_config = LLMConfig(**agent_llm_data) if agent_llm_data else None
            agent_configs.append(AgentConfig(**agent_data, llm_config=agent_llm_config))

        return cls(
            **data,
            world=world_config,
            observation=obs_config,
            memory=mem_config,
            llm=llm_config,
            agents=agent_configs,
        )

    def to_yaml(self, path: str | Path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> List[str]:
        warnings: List[str] = []

        # ascii map checks
        if self.world.generation_mode == "ascii_map":
            if not self.world.ascii_map_file:
                warnings.append("ERROR: ascii_map_file required when generation_mode='ascii_map'")
            elif not Path(self.world.ascii_map_file).exists():
                docs_path = Path("docs") / self.world.ascii_map_file
                if not docs_path.exists():
                    warnings.append(f"WARNING: ASCII map file not found: {self.world.ascii_map_file}")

        # world size
        if self.world.generation_mode == "random":
            if self.world.width < 10 or self.world.height < 10:
                warnings.append("WARNING: World size is small, agents may have limited space")

        # vision radius
        if self.observation.vision_radius >= min(self.world.width, self.world.height) / 2:
            warnings.append("WARNING: Vision radius is large relative to world size")

        # agent bounds
        if self.world.generation_mode == "random":
            for agent in self.agents:
                if agent.initial_location:
                    x, y = agent.initial_location
                    if x >= self.world.width or y >= self.world.height:
                        warnings.append(f"ERROR: Agent {agent.agent_id} starts outside world bounds")

        # memory size
        if self.memory.memory_size < self.world.max_turns:
            warnings.append("WARNING: Memory size smaller than max turns, early experiences will be lost")

        # tokens
        if self.llm.max_new_tokens > 512:
            warnings.append("WARNING: Large max_new_tokens may slow down inference significantly")

        # resource density
        if self.world.resource_density > 0.5:
            warnings.append("WARNING: Very high resource_density may crowd the world")
        elif self.world.resource_density < 0.01:
            warnings.append("WARNING: Very low resource_density may result in scarce resources")

        # respawn lag
        if self.world.respawn_lag >= self.world.max_turns / 2:
            warnings.append("WARNING: Long respawn_lag relative to max_turns")

        # agent count
        if len(self.agents) < 2:
            warnings.append("WARNING: Stag Hunt typically requires at least 2 agents for cooperation")
        elif len(self.agents) > 8:
            warnings.append("WARNING: Large number of agents may complicate coordination")

        return warnings


# --------------------------- Convenience creators ---------------------------

def create_default_staghunt_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="staghunt_2agents",
        seed=42,
        num_episodes=10,
        world=WorldConfig(
            generation_mode="random",
            width=20,
            height=20,
            max_turns=50,
            num_agents=2,
            game_type="staghunt",
            stag_reward=5.0,
            hare_reward=1.0,
            sucker_payoff=0.0,
            resource_density=0.05,
            taste_reward=0.1,
            destroyable_health=3,
            respawn_lag=10,
            require_interact=False,
            attack_radius=3,
            interaction_radius=3,
            attack_cost=0.05,
            health_regeneration_rate=0.5,
            stag_health=6,
            hare_health=1,
            agent_health=12,
            stag_regeneration_cooldown=5,
            hare_regeneration_cooldown=5,
        ),
        observation=ObservationConfig(
            vision_radius=5,
            verbose_level=1,
            use_relative_positions=True,
            include_coordinates=False,
        ),
        memory=MemoryConfig(
            memory_size=1000,
            episodic_capacity=512,
            recent_steps_in_prompt=6,
            top_agents_in_prompt=3,
        ),
        llm=LLMConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            temperature=0.7,
            max_new_tokens=256,
        ),
        agents=[
            AgentConfig(agent_id=0, role="cooperative and trusting"),
            AgentConfig(agent_id=1, role="cautious and strategic"),
        ],
    )


def create_map_based_staghunt_config(map_file: str = "simple_hunt.txt") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="staghunt_map_based",
        seed=42,
        num_episodes=1,
        world=WorldConfig(
            generation_mode="ascii_map",
            ascii_map_file=map_file,
            max_turns=50,
            num_agents=6,
            game_type="staghunt",
            stag_reward=5.0,
            hare_reward=1.0,
            sucker_payoff=0.0,
            resource_density=0.1,
            taste_reward=0.1,
            destroyable_health=3,
            respawn_lag=10,
            require_interact=False,
            attack_radius=3,
            interaction_radius=3,
            attack_cost=0.05,
            health_regeneration_rate=0.5,
            stag_health=4,
            hare_health=2,
            agent_health=12,
            stag_regeneration_cooldown=5,
            hare_regeneration_cooldown=5,
        ),
        observation=ObservationConfig(
            vision_radius=5,
            verbose_level=1,
            use_relative_positions=True,
            include_coordinates=False,
        ),
        memory=MemoryConfig(
            memory_size=1000,
            episodic_capacity=512,
        ),
        llm=LLMConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            temperature=0.7,
            max_new_tokens=256,
        ),
        agents=[
            AgentConfig(agent_id=0, role="always cooperative"),
            AgentConfig(agent_id=1, role="tit-for-tat strategy"),
            # AgentConfig(agent_id=2, role="opportunistic"),
            # AgentConfig(agent_id=3, role="cautious explorer"),
        ],
    )


def create_competitive_staghunt_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="staghunt_4agents_competitive",
        seed=42,
        num_episodes=20,
        world=WorldConfig(
            generation_mode="random",
            width=30,
            height=30,
            max_turns=100,
            num_agents=4,
            game_type="staghunt",
            stag_reward=5.0,
            hare_reward=1.0,
            sucker_payoff=-0.5,
            resource_density=0.08,
            taste_reward=0.1,
            destroyable_health=3,
            respawn_lag=12,
            require_interact=False,
            attack_radius=3,
            interaction_radius=3,
            attack_cost=0.05,
            health_regeneration_rate=0.5,
            stag_health=6,
            hare_health=1,
            agent_health=12,
            stag_regeneration_cooldown=5,
            hare_regeneration_cooldown=5,
        ),
        observation=ObservationConfig(
            vision_radius=7,
            verbose_level=1,
            use_relative_positions=True,
        ),
        memory=MemoryConfig(
            memory_size=2000,
            episodic_capacity=1024,
            recent_steps_in_prompt=8,
            top_agents_in_prompt=4,
        ),
        llm=LLMConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            temperature=0.8,
            max_new_tokens=300,
        ),
        agents=[
            AgentConfig(agent_id=0, role="always cooperative"),
            AgentConfig(agent_id=1, role="always defects"),
            AgentConfig(agent_id=2, role="tit-for-tat strategy"),
            AgentConfig(agent_id=3, role="random strategy"),
        ],
    )


def create_large_world_config(num_agents: int = 8, map_file: Optional[str] = None) -> ExperimentConfig:
    if map_file:
        generation_mode = "ascii_map"
        width, height = 0, 0
    else:
        generation_mode = "random"
        width, height = 40, 40

    return ExperimentConfig(
        experiment_name=f"staghunt_{num_agents}agents_large",
        seed=42,
        num_episodes=15,
        world=WorldConfig(
            generation_mode=generation_mode,
            ascii_map_file=map_file,
            width=width,
            height=height,
            max_turns=150,
            num_agents=num_agents,
            game_type="staghunt",
            stag_reward=5.0,
            hare_reward=1.0,
            sucker_payoff=0.0,
            resource_density=0.06,
            taste_reward=0.1,
            destroyable_health=3,
            respawn_lag=15,
            require_interact=False,
            attack_radius=3,
            interaction_radius=3,
            attack_cost=0.05,
            health_regeneration_rate=0.5,
            stag_health=6,
            hare_health=1,
            agent_health=12,
            stag_regeneration_cooldown=5,
            hare_regeneration_cooldown=5,
        ),
        observation=ObservationConfig(
            vision_radius=6,
            verbose_level=1,
            use_relative_positions=True,
        ),
        memory=MemoryConfig(
            memory_size=3000,
            episodic_capacity=1536,
            recent_steps_in_prompt=10,
            top_agents_in_prompt=5,
        ),
        llm=LLMConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            temperature=0.75,
            max_new_tokens=300,
        ),
        agents=[AgentConfig(agent_id=i, role=f"strategic agent {i}") for i in range(num_agents)],
    )


# --------------------------- Adapter to StagHuntWorld dict ---------------------------

def get_config_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Adapter from ExperimentConfig → dict that StagHuntWorld / StagHuntEnv expect."""
    return {
        "world": {
            "generation_mode": config.world.generation_mode,
            "ascii_map_file": config.world.ascii_map_file,
            "width": config.world.width,
            "height": config.world.height,
            "num_agents": config.world.num_agents,
            "max_turns": config.world.max_turns,
            "resource_density": config.world.resource_density,
            "taste_reward": config.world.taste_reward,
            "destroyable_health": config.world.destroyable_health,
            "respawn_lag": config.world.respawn_lag,
            "beam_length": config.world.beam_length,
            "beam_radius": config.world.beam_radius,
            "beam_cooldown": config.world.beam_cooldown,
            "freeze_duration": config.world.freeze_duration,
            "respawn_delay": config.world.respawn_delay,
            "payoff_matrix": config.world.payoff_matrix,
            "stag_reward": config.world.stag_reward,
            "hare_reward": config.world.hare_reward,
            "sucker_payoff": config.world.sucker_payoff,
            "stag_quorum_k": config.world.stag_quorum_k,
            "require_interact": config.world.require_interact,
            "hare_exclusive": config.world.hare_exclusive,
            "share_stag_reward": config.world.share_stag_reward,
            "r_step": config.world.r_step,
            "r_idle": config.world.r_idle,
            # NEW: beam + health parameters used in env
            "attack_radius": config.world.attack_radius,
            "interaction_radius": config.world.interaction_radius,
            "attack_cost": config.world.attack_cost,
            "attack_cooldown": config.world.attack_cooldown,
            "health_regeneration_rate": config.world.health_regeneration_rate,
            "stag_health": config.world.stag_health,
            "hare_health": config.world.hare_health,
            "agent_health": config.world.agent_health,
            "stag_regeneration_cooldown": config.world.stag_regeneration_cooldown,
            "hare_regeneration_cooldown": config.world.hare_regeneration_cooldown,
        }
    }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from openai import OpenAI


@dataclass
class EpisodeMetrics:
    episode_index: int
    agent_ids: List[int]
    first_successful_cooperation_turn: Optional[int] = None
    first_successful_cooperation_turn_by_agent: Dict[int, Optional[int]] = field(default_factory=dict)
    first_successful_cooperation_agents: Set[int] = field(default_factory=set)
    stags_hunted_by_agent: Dict[int, int] = field(default_factory=dict)
    communication_score_by_agent: Dict[int, Optional[float]] = field(default_factory=dict)
    reasoning_score_by_agent: Dict[int, Optional[float]] = field(default_factory=dict)
    communication_messages_by_agent: Dict[int, List[str]] = field(default_factory=dict)
    reasoning_texts_by_agent: Dict[int, List[str]] = field(default_factory=dict)
    episode_context: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.stags_hunted_by_agent:
            self.stags_hunted_by_agent = {aid: 0 for aid in self.agent_ids}
        if not self.first_successful_cooperation_turn_by_agent:
            self.first_successful_cooperation_turn_by_agent = {aid: None for aid in self.agent_ids}
        if not self.communication_score_by_agent:
            self.communication_score_by_agent = {aid: None for aid in self.agent_ids}
        if not self.reasoning_score_by_agent:
            self.reasoning_score_by_agent = {aid: None for aid in self.agent_ids}
        if not self.communication_messages_by_agent:
            self.communication_messages_by_agent = {aid: [] for aid in self.agent_ids}
        if not self.reasoning_texts_by_agent:
            self.reasoning_texts_by_agent = {aid: [] for aid in self.agent_ids}


class OpenAIJsonScorer:
    def __init__(self, model: str, api_key: Optional[str]) -> None:
        if not api_key:
            raise ValueError("Missing OpenAI API key for evaluation.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def score_json(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0,
            max_output_tokens=256,
        )
        text = resp.output_text.strip()
        payload = _extract_json_object(text)
        return payload


class StagHuntMetricsCollector:
    """
    Collects per-episode metrics from the StagHunt env.

    Communication and reasoning analysis are placeholders for now and treated
    as hyperparameters to be wired later.
    """

    def __init__(
        self,
        communication_prompt: Optional[str] = None,
        reasoning_prompt: Optional[str] = None,
        analysis_model: str = "gpt-4.1",
        api_key: Optional[str] = None,
    ) -> None:
        self.communication_prompt = communication_prompt
        self.reasoning_prompt = reasoning_prompt
        self.analysis_model = analysis_model
        self.api_key = api_key or os.getenv("EVAL_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.scorer: Optional[OpenAIJsonScorer] = None
        if self.communication_prompt or self.reasoning_prompt:
            if self.api_key:
                self.scorer = OpenAIJsonScorer(model=self.analysis_model, api_key=self.api_key)
        self.episodes: List[EpisodeMetrics] = []
        self._current: Optional[EpisodeMetrics] = None

    def start_episode(self, episode_index: int, agent_ids: List[int]) -> None:
        self._current = EpisodeMetrics(episode_index=episode_index, agent_ids=agent_ids)

    def end_episode(self) -> None:
        if self._current is not None:
            self._score_episode()
            self.episodes.append(self._current)
        self._current = None

    def set_episode_context(self, context: str) -> None:
        if self._current is None:
            return
        self._current.episode_context = context

    # ---- hooks called by the environment ----
    def collect_resource_defeat_event(
        self,
        attackers: Set[int],
        reward_map: Dict[int, float],
        resource_type: str,
        turn: Optional[int],
    ) -> None:
        if self._current is None:
            return
        if resource_type != "stag":
            return

        for aid in attackers:
            self._current.stags_hunted_by_agent[aid] = self._current.stags_hunted_by_agent.get(aid, 0) + 1

        if len(attackers) >= 2 and self._current.first_successful_cooperation_turn is None:
            self._current.first_successful_cooperation_turn = int(turn) if turn is not None else None
            self._current.first_successful_cooperation_agents = set(int(a) for a in attackers)
            for aid in attackers:
                self._current.first_successful_cooperation_turn_by_agent[int(aid)] = (
                    int(turn) if turn is not None else None
                )

    # ---- LLM scoring ----
    def record_message(self, agent_id: int, message: str, context: str) -> None:
        # Deprecated per-agent API (kept for compatibility)
        return

    def record_reasoning(self, agent_id: int, reasoning: str, context: str) -> None:
        # Deprecated per-agent API (kept for compatibility)
        return

    def record_message_for_episode(self, agent_id: int, message: str) -> None:
        if self._current is None:
            return
        self._current.communication_messages_by_agent.setdefault(agent_id, []).append(message)

    def record_reasoning_for_episode(self, agent_id: int, reasoning: str) -> None:
        if self._current is None:
            return
        self._current.reasoning_texts_by_agent.setdefault(agent_id, []).append(reasoning)

    def _score_episode(self) -> None:
        if self._current is None:
            return
        if not self.scorer:
            return
        context = self._current.episode_context or ""

        # Communication: per-agent, per-episode
        if self.communication_prompt:
            for aid, msgs in self._current.communication_messages_by_agent.items():
                if not msgs:
                    self._current.communication_score_by_agent[aid] = None
                    continue
                joined = json.dumps(msgs)
                prompt = self.communication_prompt.format(message=joined, context=context)
                result = self.scorer.score_json(prompt)
                self._current.communication_score_by_agent[aid] = float(result.get("communication_score", 0))

        # Reasoning: per-agent, per-episode
        if self.reasoning_prompt:
            for aid, texts in self._current.reasoning_texts_by_agent.items():
                if not texts:
                    self._current.reasoning_score_by_agent[aid] = None
                    continue
                joined = json.dumps(texts)
                prompt = self.reasoning_prompt.format(reasoning=joined, context=context)
                result = self.scorer.score_json(prompt)
                self._current.reasoning_score_by_agent[aid] = float(result.get("reasoning_score", 0))

    # Backward-compatible no-ops for existing hooks (kept intentionally)
    def collect_attack_metrics(self, agent, rtype: str, entity) -> None:
        return

    def collect_resource_defeat_metrics(self, *args, **kwargs) -> None:
        return

    def collect_shared_reward_metrics(self, agent, reward: float) -> None:
        return

    def collect_agent_cost_metrics(self, agent, attack_cost: float = 1) -> None:
        return


@dataclass
class EvaluationReport:
    summary: Dict[str, Any]
    per_episode: List[Dict[str, Any]]
    plots: Dict[str, str]
    analysis_hyperparams: Dict[str, Any]


def _default_eval_out_dir(base: Optional[str] = None) -> str:
    if base is None:
        base = ""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base, ts)


def _plot_agent_series(
    per_episode_values: List[Dict[int, float]],
    out_path: str,
    title: str,
    y_label: str,
) -> None:
    if not per_episode_values:
        return

    episode_count = len(per_episode_values)
    agent_ids = sorted({aid for ep in per_episode_values for aid in ep.keys()})

    x = np.arange(1, episode_count + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for aid in agent_ids:
        y = [per_episode_values[ep].get(aid, 0.0) for ep in range(episode_count)]
        ax.plot(x, y, marker="o", linewidth=2, label=f"agent_{aid}")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stags_hunted(per_episode_counts: List[Dict[int, int]], out_path: str) -> None:
    _plot_agent_series(
        per_episode_values=[{aid: float(v) for aid, v in ep.items()} for ep in per_episode_counts],
        out_path=out_path,
        title="Stags Hunted per Agent per Episode",
        y_label="Stags Hunted",
    )


def evaluate_staghunt(
    stats: List[dict],
    metrics_collector: Optional[StagHuntMetricsCollector],
    out_dir: Optional[str] = None,
) -> EvaluationReport:
    out_dir = out_dir or _default_eval_out_dir()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    num_episodes = len(stats)
    if num_episodes == 0:
        report = EvaluationReport(
            summary={"num_episodes": 0},
            per_episode=[],
            plots={},
            analysis_hyperparams={},
        )
        _write_report(report, out_dir)
        return report

    agent_ids: List[int] = sorted({aid for s in stats for aid in s.get("episode_rewards", {}).keys()})
    num_agents = len(agent_ids)

    # Total reward metric (average cumulative payoff per agent per episode)
    total_sum = 0.0
    for s in stats:
        for _, r in s.get("episode_rewards", {}).items():
            total_sum += float(r)
    total_reward_metric = total_sum / float(max(1, num_episodes * max(1, num_agents)))

    # First successful cooperation per episode
    per_episode: List[Dict[str, Any]] = []
    coop_turns: List[int] = []
    stags_hunted_per_episode: List[Dict[int, int]] = []
    rewards_per_episode: List[Dict[int, float]] = []
    comm_count_per_episode: List[Dict[int, int]] = []
    comm_avg_by_agent_all: Dict[int, List[float]] = {aid: [] for aid in agent_ids}
    reasoning_avg_by_agent_all: Dict[int, List[float]] = {aid: [] for aid in agent_ids}

    episodes = metrics_collector.episodes if metrics_collector is not None else []
    episode_map = {e.episode_index: e for e in episodes}

    for ep_idx in range(num_episodes):
        episode_metrics = episode_map.get(ep_idx)
        first_coop = None
        first_coop_by_agent = {aid: None for aid in agent_ids}
        first_coop_agents = []
        stags = {aid: 0 for aid in agent_ids}
        comm_scores = {aid: None for aid in agent_ids}
        reasoning_scores = {aid: None for aid in agent_ids}
        comm_count = {aid: 0 for aid in agent_ids}
        reasoning_count = {aid: 0 for aid in agent_ids}

        if episode_metrics is not None:
            first_coop = episode_metrics.first_successful_cooperation_turn
            stags.update(episode_metrics.stags_hunted_by_agent)
            first_coop_by_agent.update(episode_metrics.first_successful_cooperation_turn_by_agent)
            first_coop_agents = sorted(list(episode_metrics.first_successful_cooperation_agents))

            for aid in agent_ids:
                comm_score = episode_metrics.communication_score_by_agent.get(aid)
                comm_scores[aid] = comm_score
                if comm_score is not None:
                    comm_avg_by_agent_all[aid].append(float(comm_score))
                comm_count[aid] = int(len(episode_metrics.communication_messages_by_agent.get(aid, [])))

                reasoning_score = episode_metrics.reasoning_score_by_agent.get(aid)
                reasoning_scores[aid] = reasoning_score
                if reasoning_score is not None:
                    reasoning_avg_by_agent_all[aid].append(float(reasoning_score))
                reasoning_count[aid] = int(len(episode_metrics.reasoning_texts_by_agent.get(aid, [])))

        if first_coop is not None:
            coop_turns.append(int(first_coop))

        total_reward_by_agent = {
            aid: float(stats[ep_idx].get("episode_rewards", {}).get(aid, 0.0)) for aid in agent_ids
        }

        per_episode.append({
            "episode_index": ep_idx,
            "first_successful_cooperation_turn": first_coop,
            "first_successful_cooperation_turn_by_agent": first_coop_by_agent,
            "first_successful_cooperation_agents": first_coop_agents,
            "stags_hunted_by_agent": stags,
            "total_reward_by_agent": total_reward_by_agent,
            "communication_score_by_agent": comm_scores,
            "communication_message_count_by_agent": comm_count,
            "reasoning_score_by_agent": reasoning_scores,
            "reasoning_turn_count_by_agent": reasoning_count,
        })
        stags_hunted_per_episode.append(stags)
        rewards_per_episode.append(total_reward_by_agent)
        comm_count_per_episode.append(comm_count)

    avg_first_coop = float(np.mean(coop_turns)) if coop_turns else None

    # Plots
    plots: Dict[str, str] = {}
    stags_plot = os.path.join(out_dir, "plots", "stags_hunted_per_episode.png")
    _plot_stags_hunted(stags_hunted_per_episode, stags_plot)
    if os.path.exists(stags_plot):
        plots["stags_hunted_per_episode"] = stags_plot

    rewards_plot = os.path.join(out_dir, "plots", "total_reward_per_episode.png")
    _plot_agent_series(
        per_episode_values=rewards_per_episode,
        out_path=rewards_plot,
        title="Total Reward per Agent per Episode",
        y_label="Total Reward",
    )
    if os.path.exists(rewards_plot):
        plots["total_reward_per_episode"] = rewards_plot

    comm_count_plot = os.path.join(out_dir, "plots", "communication_message_count_per_episode.png")
    _plot_agent_series(
        per_episode_values=[{aid: float(v) for aid, v in ep.items()} for ep in comm_count_per_episode],
        out_path=comm_count_plot,
        title="Communication Message Count per Agent per Episode",
        y_label="Message Count",
    )
    if os.path.exists(comm_count_plot):
        plots["communication_message_count_per_episode"] = comm_count_plot

    report = EvaluationReport(
        summary={
            "num_episodes": num_episodes,
            "num_agents": num_agents,
            "total_reward_metric": total_reward_metric,
            "avg_first_successful_cooperation_turn": avg_first_coop,
            "avg_communication_score_by_agent": {
                aid: float(np.mean(comm_avg_by_agent_all[aid])) if comm_avg_by_agent_all[aid] else None
                for aid in agent_ids
            },
            "avg_reasoning_score_by_agent": {
                aid: float(np.mean(reasoning_avg_by_agent_all[aid])) if reasoning_avg_by_agent_all[aid] else None
                for aid in agent_ids
            },
        },
        per_episode=per_episode,
        plots=plots,
        analysis_hyperparams={
            "communication_prompt": getattr(metrics_collector, "communication_prompt", None),
            "reasoning_prompt": getattr(metrics_collector, "reasoning_prompt", None),
            "analysis_model": getattr(metrics_collector, "analysis_model", None),
        },
    )

    _write_report(report, out_dir)
    return report


def _write_report(report: EvaluationReport, out_dir: str) -> None:
    payload = {
        "summary": report.summary,
        "per_episode": report.per_episode,
        "plots": report.plots,
        "analysis_hyperparams": report.analysis_hyperparams,
    }
    path = os.path.join(out_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}

"""
Example environment runner for StagHunt with proper MessageBus + Reputation + W&B logging.

Key orchestration:
1) Deliver messages from the previous turn BEFORE agents decide.
2) Each agent decides once and may queue a message to the shared bus.
3) Execute env.step() with all actions at once.
4) Update rewards + reputation. Message delivery happens next loop.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import os
import json
import time
import re
from pathlib import Path

from sorrel.examples.staghunt.env import StagHuntEnv
from sorrel.examples.staghunt.staghunt_agent import StagHuntLLMAgent
from sorrel.examples.staghunt.staghunt_agent import create_agent_team
from sorrel.examples.staghunt.config import create_default_staghunt_config, create_map_based_staghunt_config
from sorrel.llm_configs.communication import MessageBus, Reputation  # adjust path if different
from sorrel.utils.logging import TensorboardLogger
import wandb
from datetime import datetime
from collections import defaultdict, deque
from sorrel.examples.staghunt.entities import StagResource, HareResource
from typing import Tuple
from sorrel.agents.agent import InteractionEvidence
from sorrel.models.agents import parse_llm_fields

ORIENT_TO_FACING = {
    0: "back",   # north
    1: "right",  # east
    2: "front",  # south
    3: "left",   # west
}

class StagHuntRunner:
    """Runner for managing StagHunt with LLM agents + MessageBus + Reputation + W&B + TensorBoard."""

    def __init__(
        self,
        env: StagHuntEnv,
        agents: List[StagHuntLLMAgent],
        run_ctx: Dict[str, Any] | None = None,
        tb: TensorboardLogger | None = None,
    ):
        self.env = env
        self.agents = sorted(agents, key=lambda a: a.agent_id)
        self.agent_dict = {a.agent_id: a for a in self.agents}

        # Shared bus
        self.bus: MessageBus = getattr(self.env, "message_bus", None) or MessageBus(max_per_agent=10)
        setattr(self.env, "message_bus", self.bus)
        for a in self.agents:
            a.message_bus = self.bus

        # Manhattan comm radius
        self._radius = (
            getattr(self.env, "vision_radius", None)
            or getattr(getattr(self.env, "config", None), "vision_radius", None)
            or getattr(getattr(getattr(self.env, "config", None), "world", None), "vision_radius", None)
            or 3
        )

        # --------- W&B setup ----------
        run_ctx = run_ctx or {}
        cfg = dict(run_ctx.get("config", {}) or {})
        cfg.setdefault("num_agents", len(self.agents))
        try:
            cfg.setdefault("map", getattr(getattr(self.env, "config", None), "map_name", None))
        except Exception:
            pass
        cfg["agent_names"] = [getattr(a, "name", f"agent_{a.agent_id}") for a in self.agents]

        self.wb_run = wandb.init(
            project=run_ctx.get("project", "sorrel-staghunt"),
            name=run_ctx.get("name", f"staghunt-{int(time.time())}"),
            group=run_ctx.get("group"),
            tags=run_ctx.get("tags"),
            config=cfg,
        )

        # --------- TensorBoard (provided logger) ----------
        # You must construct TensorboardLogger with a known max_epochs (we pass it in main)
        if tb is None:
            tb_dir = os.getenv("TB_LOGDIR", f"runs/staghunt/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            tb = TensorboardLogger(
                1,               # max_epochs
                tb_dir,                     # log_dir
                "hares", "stag_success", "stag_participants", "messages_sent", "episode_return"
            )

        self.tb = tb

        self.global_step = 0
        self.episode_reward_ma = defaultdict(lambda: deque(maxlen=20))

    # ---------- helpers ----------
    def _positions_dict(self) -> Dict[int, tuple[int, int]]:
        pos: Dict[int, tuple[int, int]] = {}
        ap = getattr(self.env, "agent_positions", None)
        if ap is None and hasattr(self.env, "get_agent_positions"):
            ap = self.env.get_agent_positions()
        if ap is None:
            raise RuntimeError("Environment must expose agent positions")
        for a in self.agents:
            yx = ap[a.agent_id]
            pos[a.agent_id] = (int(yx[0]), int(yx[1]))
        return pos

    def _deliver_bus_now(self) -> None:
        positions = self._positions_dict()
        self.bus.deliver(positions=positions, radius=self._radius)

    def _parse_llm_fields(self, text: str) -> Dict[str, Any]:
        return parse_llm_fields(text, default_action=None)

    def _reasoning_fields_for_agent(self, agent: StagHuntLLMAgent, raw: str) -> Dict[str, Any]:
        lp = getattr(agent.model, "last_parsed", {}) or {}
        if lp:
            fields = {
                "REASONING": lp.get("REASONING"),
                "ACTION": lp.get("ACTION"),
                "MESSAGE": lp.get("MESSAGE"),
                "CONFIDENCE": lp.get("CONFIDENCE"),
            }
            if all(v is not None for v in fields.values()):
                return fields
        parsed = self._parse_llm_fields(raw)
        return {
            "REASONING": (lp.get("REASONING") if lp and lp.get("REASONING") is not None else parsed.get("REASONING")),
            "ACTION": (lp.get("ACTION") if lp and lp.get("ACTION") is not None else parsed.get("ACTION")),
            "MESSAGE": (lp.get("MESSAGE") if lp and lp.get("MESSAGE") is not None else parsed.get("MESSAGE")),
            "CONFIDENCE": (lp.get("CONFIDENCE") if lp and lp.get("CONFIDENCE") is not None else parsed.get("CONFIDENCE")),
        }

    def _parse_commitment(self, text: str | None) -> str | None:
        if not text:
            return None
        t = text.lower()
        if "stag" in t and ("attack" in t or "hunt" in t):
            return "attack_stag"
        if "hare" in t and ("attack" in t or "hunt" in t):
            return "attack_hare"
        return None

    # ---------- main loops ----------
    def run_episode(self, max_steps: int = 100, verbose: bool = False, episode_idx: int | None = None) -> dict:
        # --- episode init ---
        trace = []
        obs = self.env.reset()
        for a in self.agents:
            if hasattr(a, "reset"):
                a.reset()
        # Use the shared bus with the env (env.reset() already reset its own bus)
        self.bus.reset([a.agent_id for a in self.agents])

        episode_rewards = {a.agent_id: 0.0 for a in self.agents}
        prev_inventory = {agent.agent_id: {"hare": 0, "stag": 0} for agent in self.agents}
        step_count, done = 0, False

        # Deliver anything queued before the first step (typically none)
        self._deliver_bus_now()

        while not done and step_count < max_steps:
            if verbose:
                print(f"\n{'='*60}\nStep {step_count + 1}\n{'='*60}")

            # ---- phase 1: agents decide (choose action & optionally queue a message) ----
            actions: Dict[int, int] = {}
            llm_responses: Dict[int, str] = {}

            for agent in self.agents:
                # Agent builds a contextual observation and chooses an action.
                agent.transition(self.env)  # sets agent.last_action and may set agent.current_message
                action = getattr(agent, "last_action", None)
                if action is None:
                    action = 0  # safe default
                actions[agent.agent_id] = int(action)

                # Capture the last LLM response if available (for logging/memory)
                hist = getattr(agent.model, "conversation_history", None)
                if hist:
                    try:
                        llm_responses[agent.agent_id] = hist[-1].get("response", "")
                    except Exception:
                        pass

                # Only queue chat if a neighbor is within the (Manhattan) vision radius
                msg = getattr(agent, "current_message", None)
                if msg and self.env.has_neighbor_within_radius(agent.agent_id, getattr(self.env, "vision_radius", 3)):
                    payload = msg if isinstance(msg, dict) else {"text": str(msg)}
                    self.bus.queue(sender_id=agent.agent_id, message=payload)
                else:
                    agent.current_message = None

                if verbose:
                    print(f"Agent {agent.agent_id}: action={actions[agent.agent_id]}, msg={getattr(agent, 'current_message', None)}")

            # ---- phase 2: env step (simultaneous action resolution) ----
            ordered_ids = [a.agent_id for a in self.agents]
            obs, rewards, done, info = self.env.step(actions)

            positions = self._positions_dict()

            # Extract messages sent *this* frame (for trace/logging only)
            frame_messages = []
            if hasattr(self.bus, "last_queued"):
                for m in self.bus.last_queued:
                    if isinstance(m, dict):
                        sid = m.get("sender_id") or m.get("sender")
                        msg = m.get("message")
                        text = msg.get("text", "") if isinstance(msg, dict) else m.get("text", "")
                    elif isinstance(m, tuple) and len(m) >= 2:
                        sid, payload = m[0], m[1]
                        text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
                    else:
                        sid, text = None, str(m)
                    frame_messages.append({"sender": sid, "text": text})
                # clear after consuming
                self.bus.last_queued = []

            # Parse model reasoning (if provided by your LLM wrapper)
            reasoning = {}
            for agent in self.agents:
                lp = getattr(agent.model, "last_parsed", {}) or {}
                raw = lp.get("RAW") if isinstance(lp, dict) else ""
                if not raw:
                    raw = llm_responses.get(agent.agent_id, "")
                fields = self._reasoning_fields_for_agent(agent, raw)
                if fields.get("ACTION") is None:
                    fields["ACTION"] = actions.get(agent.agent_id)
                if any(v is not None and v != "" for v in fields.values()):
                    reasoning[str(agent.agent_id)] = fields
            frame_rewards = {str(aid): float(rewards.get(aid, 0.0)) for aid in ordered_ids}


            # Trace frame (positions/resources/messages/reasoning)
            trace.append({
                "t": step_count,
                "agents": [
                    {"id": a.agent_id,
                    "y": int(positions[a.agent_id][0]),
                    "x": int(positions[a.agent_id][1]),
                    "facing": ORIENT_TO_FACING.get(int(getattr(a, "orientation", 2)), "front"),
                    "health": int(getattr(a, "health", 0)),
                    }
                    for a in self.agents
                ],
                "hares": self.env.hare_states(),
                "stags": self.env.stag_states(),
                "beams": [
                    {"y": int(y), "x": int(x), "kind": kind}
                    for (y, x, kind) in self.env.beam_positions()
                ],
                "actions": {str(aid): int(actions[aid]) for aid in ordered_ids},
                "rewards": frame_rewards,
                "messages": frame_messages,
                "reasoning": reasoning,
            })

            # ---- phase 3: distribute rewards to agents & log ----
            if isinstance(rewards, dict):
                rdict = {int(k): float(v) for k, v in rewards.items()}
            else:
                rdict = {ordered_ids[i]: float(rewards[i]) for i in range(len(ordered_ids))}

            # ---- phase 3.5: update reputation using structured evidence ----
            # Build a per-sender commitment map from messages sent this frame.
            commitment_by_sender: Dict[int, str] = {}
            for m in frame_messages:
                sid = m.get("sender")
                commit = self._parse_commitment(m.get("text"))
                if sid is not None and commit:
                    commitment_by_sender[int(sid)] = commit

            for agent in self.agents:
                my_id = agent.agent_id
                my_reward = rdict.get(my_id, 0.0)
                my_pos = positions.get(my_id)
                for other_id in ordered_ids:
                    if other_id == my_id:
                        continue
                    other_reward = rdict.get(other_id, 0.0)
                    other_attacked = int(actions.get(other_id, 0)) == 5  # 5 == attack
                    commitment = commitment_by_sender.get(other_id)
                    target = "stag" if commitment == "attack_stag" else "hare" if commitment == "attack_hare" else None

                    # Heuristic: if a committed target yielded positive reward, treat as success.
                    success = bool(target) and my_reward > 0
                    other_participated = bool(target) and other_reward > 0

                    other_nearby = None
                    if my_pos is not None:
                        o_pos = positions.get(other_id)
                        if o_pos is not None:
                            other_nearby = (abs(my_pos[0] - o_pos[0]) + abs(my_pos[1] - o_pos[1])) <= self._radius

                    evidence = InteractionEvidence(
                        commitment=commitment,
                        other_attacked=other_attacked,
                        other_participated_in_kill=other_participated,
                        target=target,
                        reward_me=my_reward,
                        reward_other=other_reward,
                        success=success,
                        other_nearby=other_nearby,
                    )
                    agent.update_reputation_for_interaction(
                        other_agent_id=other_id,
                        evidence=evidence,
                    )

            trace[-1]["rewards"] = {str(aid): rdict.get(aid, 0.0) for aid in ordered_ids}
            trace[-1]["rewards_cum"] = {
                str(aid): float(episode_rewards.get(aid, 0.0) + rdict.get(aid, 0.0)) for aid in ordered_ids
            }

            wb_step_log = {"t": step_count}
            total_step_reward = 0.0
            for agent in self.agents:
                r = rdict.get(agent.agent_id, 0.0)
                total_step_reward += r
                episode_rewards[agent.agent_id] += r
                wb_step_log[f"reward/step/agent_{agent.agent_id}"] = r
                # TB per-step agent reward
                self.tb.writer.add_scalar(f"agent/{agent.agent_id}/reward_step", r, self.global_step)

                # FEED BACK THE ENV REWARD TO THE AGENT (this is the critical connection)
                agent.update_with_reward(
                    new_obs = obs.get(agent.agent_id, None),
                    reward=r,
                    done=done,
                    llm_response=llm_responses.get(agent.agent_id)
                )

            # World/chat signals (may be absent; default to 0)
            # hares = info.get("hares", 0) if isinstance(info, dict) else 0
            # stag_success = info.get("stag_success", 0) if isinstance(info, dict) else 0
            # stag_participants = info.get("stag_participants", 0) if isinstance(info, dict) else 0
            # messages_sent = info.get("messages_sent", 0) if isinstance(info, dict) else len(frame_messages)

            # wb_step_log["world/hares_step"] = hares
            # wb_step_log["world/stag_success_step"] = stag_success
            # wb_step_log["world/stag_participants_step"] = stag_participants
            # wb_step_log["chat/messages_sent_step"] = messages_sent
            wb_step_log["world/total_reward_step"] = total_step_reward

            # W&B & TB logging
            wandb.log(wb_step_log, step=self.global_step)
            # self.tb.writer.add_scalar("world/hares_step", hares, self.global_step)
            # self.tb.writer.add_scalar("world/stag_success_step", stag_success, self.global_step)
            # self.tb.writer.add_scalar("world/stag_participants_step", stag_participants, self.global_step)
            # self.tb.writer.add_scalar("chat/messages_sent_step", messages_sent, self.global_step)
            self.tb.writer.add_scalar("world/total_reward_step", total_step_reward, self.global_step)

            self.global_step += 1
            step_count += 1

            # Update prev inventory snapshot for next step
            for agent in self.agents:
                aid = agent.agent_id
                intv = getattr(agent, "inventory", {})
                prev_inventory[aid]["hare"] = int(intv.get("hare", 0))
                prev_inventory[aid]["stag"] = int(intv.get("stag", 0))

            # ---- phase 4: deliver messages for the next turn ----
            self._deliver_bus_now()

        # --- end loop; save per-episode trace ---
        out_dir = "runs"
        os.makedirs(out_dir, exist_ok=True)
        trace_filename = f"trace_ep{episode_idx if episode_idx is not None else 0}_{int(time.time()*1000)}.json"
        out_path = os.path.join(out_dir, trace_filename)
        with open(out_path, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"Saved episode trace to {out_path}")

        # Episode-level logging
        wb_ep_log = {"episode": episode_idx if episode_idx is not None else 0}
        total_reward = 0.0
        for aid, total in episode_rewards.items():
            total_reward += total
            self.episode_reward_ma[aid].append(total)
            ma = sum(self.episode_reward_ma[aid]) / len(self.episode_reward_ma[aid])
            wb_ep_log[f"reward/episode_sum/agent_{aid}"] = total
            wb_ep_log[f"reward/episode_ma20/agent_{aid}"] = ma
            # TB per-agent episode total
            self.tb.writer.add_scalar(f"episode/agent/{aid}/return", total, self.global_step)

        wb_ep_log["reward/episode_total"] = total_reward
        wb_ep_log["episode/steps"] = step_count
        wandb.log(wb_ep_log, step=self.global_step)

        # Episode-level TB summary (your API allows extra keys)
        self.tb.record_turn(
            epoch=(episode_idx or 0),
            loss=0.0,  # set real loss if you have training
            reward=total_reward,
            epsilon=0.0,
            # hares=hares,
            # stag_success=stag_success,
            # stag_participants=stag_participants,
            # messages_sent=messages_sent,
            episode_return=total_reward,
        )

        # Upload the trace as an artifact (unique per episode)
        try:
            art = wandb.Artifact(f"episode-trace-{episode_idx if episode_idx is not None else 0}", type="trace")
            art.add_file(out_path, name=os.path.basename(out_path))
            wandb.log_artifact(art)
        except Exception:
            pass

        return {
            "total_steps": step_count,
            "episode_rewards": episode_rewards,
            "total_reward": float(total_reward),
            "done": True,
        }


    def run_multiple_episodes(self, num_episodes: int, max_steps: int = 100, verbose: bool = False) -> List[dict]:
        all_stats: List[dict] = []
        for ep in range(num_episodes):
            if verbose:
                print(f"\n{'#'*60}\nEPISODE {ep + 1}/{num_episodes}\n{'#'*60}")
            stats = self.run_episode(max_steps=max_steps, verbose=verbose, episode_idx=ep)
            stats["episode_num"] = ep + 1
            all_stats.append(stats)

            if verbose:
                print(f"\nEpisode {ep + 1} Results:")
                print(f"  Total Steps: {stats['total_steps']}")
                print(f"  Total Reward: {stats['total_reward']:.2f}")
                print(f"  Individual Rewards: {stats['episode_rewards']}")

        return all_stats


# ------------- Example usage -------------
if __name__ == "__main__":
    # Create configuration
    map_path = Path(__file__).with_name("map.txt")
    config = create_map_based_staghunt_config(map_file=str(map_path))
    # INTERACT is for chat; rewards should not require INTERACT
    if hasattr(config, "world"):
        config.world.max_turns = 50
        if hasattr(config.world, "require_interact"):
            config.world.require_interact = False  # hare=+1 on standing; stag=+5 each if quorum
        if hasattr(config, "num_agents"):
            config.world.num_agents = 6

    # Create team of N agents with shared MessageBus and Reputation
    # agents, bus, rep = create_agent_team(
    #     num_agents=2,
    #     model_name="Qwen/Qwen2.5-3B-Instruct",
    #     config=config,
    #     verbose=True
    # )

    # gpt model
    agents, bus, rep = create_agent_team(
        num_agents=2,
        model_name="openai:gpt-4o",   # Required to trigger API path
        api_provider="openai",        # Required
        api_model="gpt-4o",           # Actual OpenAI model string
        api_key="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",
        temperature=0.1,
        max_tokens=512,
    )


    # Create environment and ensure it uses the same bus
    env = StagHuntEnv(config.to_dict(), agents)
    env.message_bus = bus

    # --- W&B run context ---
    run_ctx = dict(
        project="sorrel-staghunt",
        name=f"map-{getattr(config, 'map_name', 'ascii')}-agents-{len(agents)}",
        group=getattr(config, "experiment_group", None),
        tags=["LLM", "staghunt", "comm"],
        config={
            "seed": getattr(config, "seed", None),
            "map_name": getattr(config, "map_name", None),
            "vision_radius": getattr(config, "vision_radius", getattr(getattr(config, "world", None), "vision_radius", None)),
            "max_turns": getattr(getattr(config, "world", None), "max_turns", None),
            "reward_scheme": getattr(config, "reward_scheme", None),
        },
    )

    # --- TensorBoard logger (use your provided class) ---
    num_episodes = 1
    tb_dir = os.getenv("TB_LOGDIR", f"runs/staghunt/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    tb = TensorboardLogger(
        num_episodes,
        tb_dir,
        # declare extra tags we'll log via .record_turn(...)
        "hares", "stag_success", "stag_participants", "messages_sent", "episode_return"
    )

    # Create runner and go
    runner = StagHuntRunner(env, agents, run_ctx=run_ctx, tb=tb)
    stats = runner.run_multiple_episodes(num_episodes=num_episodes, max_steps=50, verbose=True)

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    avg_reward = np.mean([s["total_reward"] for s in stats])
    avg_steps = np.mean([s["total_steps"] for s in stats])
    print(f"Average Total Reward: {avg_reward:.2f}")
    print(f"Average Steps per Episode: {avg_steps:.1f}")

    # Optional: per-agent summaries if your agent exposes it
    for agent in agents:
        if hasattr(agent, "get_strategy_summary"):
            print(f"\n{agent.get_strategy_summary()}")

    # Close TB and W&B
    try:
        tb.writer.flush()
        tb.writer.close()
    except Exception:
        pass
    wandb.finish()

    print(f"\nTensorBoard logs → {tb_dir}\nRun with:\n  tensorboard --logdir {os.path.dirname(tb_dir)}")

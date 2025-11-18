#!/usr/bin/env python3
"""
Enhanced Stag Hunt visualizer — fixed to handle dict/None reasoning,
single-frame traces, optional map inference, and per-frame rewards.

Example:
  python viz_staghunt.py --trace trace_ep0_1761591953290.json --interactive
  # or save:
  python viz_staghunt.py --trace trace_ep0_1761591953290.json --save out.gif
"""

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib
# Use a GUI backend if available; TkAgg works on macOS with Python.org install
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

class AgentTracker:
    """Track agent rewards and actions over time (display only)."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.total_reward = 0.0
        self.rewards_history: List[float] = []
        self.actions_history: List[int] = []
        self.messages_history: List[str] = []
        self.reasoning_history: List[str] = []

    def reset(self):
        self.total_reward = 0.0
        self.rewards_history.clear()
        self.actions_history.clear()
        self.messages_history.clear()
        self.reasoning_history.clear()

    def add_turn(self, reward: float, action: int, message: str = "", reasoning: str = ""):
        self.total_reward += reward
        self.rewards_history.append(reward)
        self.actions_history.append(action)
        self.messages_history.append(message)
        self.reasoning_history.append(reasoning)


class ResponseParser:
    """Parse agent responses based on the expected JSON format (robust to fences)."""
    _FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    _OBJ    = re.compile(r"\{.*\}", re.DOTALL)

    @staticmethod
    def _light_repair(s: str) -> str:
        # single → double quotes; remove trailing commas
        s = s.replace("'", '"')
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    @staticmethod
    def parse_agent_response(response) -> Dict[str, Any]:
        """
        Accepts:
        - dict like {"REASONING": ..., "ACTION": ..., "MESSAGE": ..., "CONFIDENCE": ...}
        - string containing fenced or inline JSON, or free-form text
        - None
        Returns a normalized dict: {"reasoning": str, "action": int, "message": str, "confidence": int}
        """
        default = {"reasoning": "", "action": 0, "message": "", "confidence": 50}

        # Already dict
        if isinstance(response, dict):
            def _to_int(v, d): 
                try: return int(v)
                except: return d
            return {
                "reasoning": str(response.get("REASONING") or response.get("REASON") or response.get("reasoning") or ""),
                "action": _to_int(response.get("ACTION", response.get("action", 0)), 0),
                "message": str(response.get("MESSAGE", response.get("message", "") or ""))[:60],
                "confidence": _to_int(response.get("CONFIDENCE", response.get("confidence", 50)), 50),
            }

        # None or empty
        if not response:
            return dict(default)

        text = str(response).strip()

        # Fenced JSON block first
        m = ResponseParser._FENCED.search(text)
        blob = None
        if m:
            blob = m.group(1)
        else:
            # any JSON object
            m2 = ResponseParser._OBJ.search(text)
            if m2:
                blob = m2.group(0)

        if blob:
            try:
                obj = json.loads(blob)
            except Exception:
                try:
                    obj = json.loads(ResponseParser._light_repair(blob))
                except Exception:
                    obj = {}
            if obj:
                return {
                    "reasoning": str(obj.get("REASONING") or obj.get("REASON") or obj.get("reasoning") or ""),
                    "action": int(obj.get("ACTION", obj.get("action", 0)) or 0),
                    "message": str(obj.get("MESSAGE", obj.get("message", "") or ""))[:60],
                    "confidence": int(obj.get("CONFIDENCE", obj.get("confidence", 50)) or 50),
                }

        # Fallback: free-form patterns
        out = dict(default)
        m = re.search(r'REASON(?:ING)?\s*[:=]\s*(.+)', text, re.I)
        if m: out["reasoning"] = m.group(1).strip().strip('"\'')

        m = re.search(r'ACTION\s*[:=]\s*(\d+)', text, re.I)
        if m: out["action"] = int(m.group(1))

        m = re.search(r'MESSAGE\s*[:=]\s*(.+)', text, re.I)
        if m: out["message"] = m.group(1).strip().strip('"\'')

        m = re.search(r'CONFIDENCE\s*[:=]\s*(\d+)', text, re.I)
        if m: out["confidence"] = int(m.group(1))

        return out


# ------------------------------------------------------------
# Helpers for world/map + rewards
# ------------------------------------------------------------

def parse_ascii_map(lines: List[str]) -> Dict[str, Any]:
    """Parse ASCII map to get dimensions and walls."""
    H = len(lines)
    W = max(len(r) for r in lines) if lines else 12
    walls = []
    for y, row in enumerate(lines):
        for x, ch in enumerate(row.rstrip("\n")):
            if ch in ("W", "#"):
                walls.append((y, x))
    return {"H": H, "W": W, "walls": walls}


def load_map_or_infer(map_file: Optional[str], frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load map from file if provided; otherwise infer H, W from the frames.
    Walls default to empty when inferred.
    """
    if map_file:
        p = Path(map_file)
        if not p.exists():
            raise FileNotFoundError(f"Map file not found: {map_file}")
        with p.open("r") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        return parse_ascii_map(lines)

    # Infer from frames: take max x/y among agents/resources, add margins
    max_x = max_y = 0
    for fr in frames:
        for a in fr.get("agents", []):
            max_x = max(max_x, int(a.get("x", 0)))
            max_y = max(max_y, int(a.get("y", 0)))
        for h in fr.get("hares", []):
            max_x = max(max_x, int(h.get("x", 0)))
            max_y = max(max_y, int(h.get("y", 0)))
        for s in fr.get("stags", []):
            max_x = max(max_x, int(s.get("x", 0)))
            max_y = max(max_y, int(s.get("y", 0)))
    # add some breathing room
    H = max_y + 2
    W = max_x + 2
    return {"H": H, "W": W, "walls": []}


def extract_rewards_by_frame(frames: List[Dict[str, Any]],
                             stag_reward: float,
                             hare_reward: float,
                             sucker_payoff: float) -> List[Dict[int, float]]:
    """
    Produce a list of dicts, one per frame, mapping agent_id -> reward.
    Priority: use explicit frame["rewards"] if present; otherwise infer.
    """
    out: List[Dict[int, float]] = []

    for frame in frames:
        frame_rewards: Dict[int, float] = {}

        if "rewards" in frame and isinstance(frame["rewards"], dict):
            for aid_str, val in frame["rewards"].items():
                try:
                    aid = int(aid_str)
                    frame_rewards[aid] = float(val)
                except Exception:
                    continue
        else:
            # Infer from positions and resources
            agents = {int(a["id"]): (int(a["x"]), int(a["y"])) for a in frame.get("agents", [])}
            hares = {(int(h["x"]), int(h["y"])) for h in frame.get("hares", [])}
            stags = {(int(s["x"]), int(s["y"])) for s in frame.get("stags", [])}

            # Group agents by tile
            tile_to_agents: Dict[Tuple[int, int], List[int]] = {}
            for aid, (x, y) in agents.items():
                tile_to_agents.setdefault((x, y), []).append(aid)

            for (x, y), aids in tile_to_agents.items():
                if (x, y) in hares:
                    for aid in aids:
                        frame_rewards[aid] = hare_reward
                elif (x, y) in stags:
                    if len(aids) >= 2:
                        for aid in aids:
                            frame_rewards[aid] = stag_reward
                    else:
                        for aid in aids:
                            frame_rewards[aid] = sucker_payoff
                else:
                    for aid in aids:
                        frame_rewards.setdefault(aid, 0.0)

        out.append(frame_rewards)
    return out


# ------------------------------------------------------------
# Renderer
# ------------------------------------------------------------

def render_enhanced(frames: List[Dict[str, Any]],
                    world_info: Dict[str, Any],
                    config: Dict[str, float],
                    out_path: Optional[str] = None,
                    fps: int = 2,
                    interactive: bool = False):
    """Enhanced renderer with reasoning panel and stats."""
    H, W = world_info["H"], world_info["W"]
    wallset = set(world_info["walls"])
    stag_reward = float(config.get("stag_reward", 5.0))
    hare_reward = float(config.get("hare_reward", 1.0))
    sucker_payoff = float(config.get("sucker_payoff", 0.0))

    # Precompute rewards per frame (explicit or inferred)
    rewards_by_frame = extract_rewards_by_frame(frames, stag_reward, hare_reward, sucker_payoff)

    # Build trackers for all agents seen in the first frame (and any newcomers later)
    agent_trackers: Dict[int, AgentTracker] = {}
    if frames:
        for a in frames[0].get("agents", []):
            agent_trackers[int(a["id"])] = AgentTracker(int(a["id"]))

    # Prefer constrained layout instead of manual tight_layout calls
    fig = plt.figure(figsize=(20, 12), layout="constrained")  # if Matplotlib <3.6: use constrained_layout=True

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[4, 1],            # more room for the top row (map + panels)
        width_ratios=[3.2, 2.5, 2.0]     # tune these to taste
    )

    ax_grid      = fig.add_subplot(gs[0, 0])   # map
    ax_reasoning = fig.add_subplot(gs[0, 1])   # responses
    ax_stats     = fig.add_subplot(gs[0, 2])   # stats
    ax_legend    = fig.add_subplot(gs[1, 0])   # legend ONLY under the map

    # two empty spacers under the right panels
    _ax_sp1 = fig.add_subplot(gs[1, 1]); _ax_sp1.axis("off")
    _ax_sp2 = fig.add_subplot(gs[1, 2]); _ax_sp2.axis("off")


    agent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                    '#bcbd22', '#17becf']

    action_names = ["Stay", "Up", "Right", "Down", "Left", "Interact"]
    parser = ResponseParser()

    def get_response_for(frame: Dict[str, Any], aid: int, parser: ResponseParser) -> Dict[str, Any]:
        """
        Normalized per-agent response with keys:
        reasoning:str, action:int, message:str, confidence:int

        Priority:
        (A) frame["responses"][str(aid)]              # preferred schema
        (B) frame["reasoning"][str(aid)] + others     # backward compatible
        (C) defaults
        """
        # (A) Preferred: full block per agent
        resp_block = (frame.get("responses") or {}).get(str(aid))
        if resp_block is not None:
            return parser.parse_agent_response(resp_block)

        # (B) Legacy: reasoning object/string + optional other dicts
        parsed = parser.parse_agent_response((frame.get("reasoning") or {}).get(str(aid)))

        # action/confidence/message could be elsewhere in the frame
        try:
            if "actions" in frame and str(aid) in frame["actions"]:
                parsed["action"] = int(frame["actions"][str(aid)])
        except Exception:
            pass
        try:
            if "confidence" in frame and str(aid) in frame["confidence"]:
                parsed["confidence"] = int(frame["confidence"][str(aid)])
        except Exception:
            pass

        if not parsed.get("message"):
            # look for a message event authored by this agent
            for msg in frame.get("messages", []):
                if str(msg.get("sender")) == str(aid) and msg.get("text"):
                    parsed["message"] = str(msg["text"])[:60]
                    break

        # ensure all keys exist
        parsed.setdefault("reasoning", "")
        parsed.setdefault("action", 0)
        parsed.setdefault("message", "")
        parsed.setdefault("confidence", 50)
        return parsed


    def draw_speech_bubble(ax, x, y, text, color):
        """Draw a compact speech bubble near the agent."""
        if not text:
            return

        wrapped_lines = textwrap.wrap(text, width=12)[:3]
        text_wrapped = "\n".join(wrapped_lines)

        bubble_x = x + 0.9
        bubble_y = y - 0.3
        if bubble_x > W - 1.5:
            bubble_x = x - 0.9
        if bubble_y < 0.5:
            bubble_y = y + 0.3

        line_count = max(1, len(wrapped_lines))
        bubble_w = 0.95
        bubble_h = 0.28 + 0.16 * (line_count - 1)

        # Fixed: no buble_x/y typos
        bbox = FancyBboxPatch((bubble_x - bubble_w / 2, bubble_y - bubble_h / 2), bubble_w, bubble_h,
                              boxstyle="round,pad=0.05", facecolor='white', edgecolor=color,
                              alpha=0.95, linewidth=2, zorder=10)
        ax.add_patch(bbox)

        ax.text(bubble_x, bubble_y, text_wrapped, ha='center', va='center',
                fontsize=8, color='black', linespacing=1.1, zorder=11)

        arrow = FancyArrowPatch((x + 0.2, y), (bubble_x - 0.3, bubble_y),
                                arrowstyle='-', color=color, linewidth=1.2, alpha=0.7, zorder=9)
        ax.add_patch(arrow)

    def recompute_trackers_upto(frame_idx: int, current_reasoning: Dict[int, Dict[str, Any]]):
        """Recompute cumulative totals up to frame_idx for display (simple & correct)."""
        # Ensure we include all agents present so far
        for a in frames[frame_idx].get("agents", []):
            aid = int(a["id"])
            if aid not in agent_trackers:
                agent_trackers[aid] = AgentTracker(aid)

        for tracker in agent_trackers.values():
            tracker.reset()

        # Accumulate rewards up to this frame
        for t in range(frame_idx + 1):
            frame = frames[t]
            rdict = rewards_by_frame[t]
            for aid, tracker in agent_trackers.items():
                r = float(rdict.get(aid, 0.0))
                # Use latest parsed action/message/reasoning from current frame (for display)
                parsed = current_reasoning.get(aid, {"action": 0, "message": "", "reasoning": ""})
                tracker.add_turn(reward=r,
                                 action=int(parsed.get("action", 0)),
                                 message=str(parsed.get("message", "")),
                                 reasoning=str(parsed.get("reasoning", "")))

    last_decision: Dict[int, Dict[str, Any]] = {}
    last_response: Dict[int, Dict[str, Any]] = {}

    def draw_frame(frame_idx: int):
        ax_grid.clear()
        ax_reasoning.clear()
        ax_stats.clear()
        ax_legend.clear()

        # Grid setup
        ax_grid.set_xlim(-0.5, W - 0.5)
        ax_grid.set_ylim(H - 0.5, -0.5)
        ax_grid.set_aspect('equal')
        ax_grid.grid(True, alpha=0.2, linestyle='--')
        ax_grid.set_title(f"Turn {frame_idx + 1}/{len(frames)}", fontsize=14, fontweight='bold')

        # Draw tiles
        for y in range(H):
            for x in range(W):
                if (y, x) in wallset:
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='#2b2b2b', edgecolor='#505050')
                else:
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='#f5f5dc', edgecolor='#e0e0d0', alpha=0.3)
                ax_grid.add_patch(rect)

        frame = frames[frame_idx]

        # Resources
        for hare in frame.get("hares", []):
            hx, hy = int(hare["x"]), int(hare["y"])
            circle = Circle((hx, hy), 0.3, color='#90EE90', edgecolor='#228B22', linewidth=2)
            ax_grid.add_patch(circle)
            ax_grid.text(hx, hy, "H", ha='center', va='center', fontsize=12, weight='bold', color='#006400')

        for stag in frame.get("stags", []):
            sx, sy = int(stag["x"]), int(stag["y"])
            circle = Circle((sx, sy), 0.35, color='#CD853F', edgecolor='#8B4513', linewidth=2)
            ax_grid.add_patch(circle)
            ax_grid.text(sx, sy, "S", ha='center', va='center', fontsize=12, weight='bold', color='white')

        # Agents + reasoning/messages
        agent_positions: Dict[int, Tuple[int, int]] = {}
        current_reasoning: Dict[int, Dict[str, Any]] = {}

        for agent in frame.get("agents", []):
            aid = int(agent["id"])
            x, y = int(agent["x"]), int(agent["y"])
            agent_positions[aid] = (x, y)

            color = agent_colors[aid % len(agent_colors)]
            ax_grid.add_patch(Circle((x, y), 0.25, color=color, edgecolor='black', linewidth=2))
            ax_grid.text(x, y, str(aid), ha='center', va='center', fontsize=12, color='white', weight='bold')

            # Get full response (reasoning/action/message/confidence)
            resp = get_response_for(frame, aid, parser)

            # carry forward missing fields from previous frames
            prev = last_response.get(aid, {})
            if not resp.get("reasoning") and prev.get("reasoning"):
                resp["reasoning"] = prev["reasoning"]
            if not resp.get("message") and prev.get("message"):
                resp["message"] = prev["message"]
            if resp.get("confidence", None) is None and "confidence" in prev:
                resp["confidence"] = prev["confidence"]

            last_response[aid] = resp
            current_reasoning[aid] = resp

            if resp.get("message"):
                draw_speech_bubble(ax_grid, x, y, resp["message"], color)

        # If no messages via reasoning, optional fallback to frame["messages"]
        if not any((cr.get("message") for cr in current_reasoning.values())):
            for msg in frame.get("messages", []):
                sender = msg.get("sender")
                text = msg.get("text", "")
                if sender is None or not text:
                    continue
                try:
                    sender_id = int(sender)
                    if sender_id in agent_positions:
                        x, y = agent_positions[sender_id]
                        color = agent_colors[sender_id % len(agent_colors)]
                        draw_speech_bubble(ax_grid, x, y, text[:40], color)
                except Exception:
                    pass

        # Reasoning panel
        ax_reasoning.set_title("Agent Responses", fontsize=12, fontweight='bold')
        ax_reasoning.axis('off')

        # union of seen agents, so panel includes everyone every turn
        all_ids = set(agent_trackers.keys())
        all_ids.update(int(a["id"]) for a in frame.get("agents", []))

        y_pos = 0.95
        for aid in sorted(all_ids):
            color = agent_colors[aid % len(agent_colors)]
            resp = current_reasoning.get(aid, last_response.get(aid, {"reasoning":"", "action":0, "message":"", "confidence":50}))

            ax_reasoning.text(0.02, y_pos, f"Agent {aid}:", transform=ax_reasoning.transAxes,
                            fontsize=11, fontweight='bold', color=color)
            y_pos -= 0.08

            reasoning_wrapped = textwrap.fill(str(resp.get("reasoning") or "No reasoning"), width=70)
            ax_reasoning.text(0.05, y_pos, f"• {reasoning_wrapped}", transform=ax_reasoning.transAxes, fontsize=9, wrap=True)
            y_pos -= 0.05 * (reasoning_wrapped.count("\n") + 1)

            action = int(resp.get("action", 0))
            action_name = action_names[action] if 0 <= action < len(action_names) else str(action)
            conf = int(resp.get("confidence", 50))
            ax_reasoning.text(0.05, y_pos, f"• Action: {action_name} (Conf: {conf}%)",
                            transform=ax_reasoning.transAxes, fontsize=9, color='#555')
            y_pos -= 0.06

            msg = resp.get("message", "")
            if msg:
                msg_wrapped = textwrap.fill(str(msg), width=70)
                ax_reasoning.text(0.05, y_pos, f"• Message: {msg_wrapped}",
                                transform=ax_reasoning.transAxes, fontsize=9, color='#444')
                y_pos -= 0.05 * (msg_wrapped.count("\n") + 1)
            y_pos -= 0.06

        # Recompute tracker totals up to this frame (simple + consistent)
        recompute_trackers_upto(frame_idx, current_reasoning)

        # Stats panel
        ax_stats.set_title("Agent Stats", fontsize=12, fontweight='bold')
        ax_stats.axis('off')
        y_pos = 0.95
        for aid in sorted(agent_trackers.keys()):
            tracker = agent_trackers[aid]
            color = agent_colors[aid % len(agent_colors)]

            ax_stats.text(0.02, y_pos, f"Agent {aid}:", transform=ax_stats.transAxes,
                          fontsize=11, fontweight='bold', color=color)
            y_pos -= 0.08

            ax_stats.text(0.05, y_pos, f"Total: {tracker.total_reward:.1f}",
                          transform=ax_stats.transAxes, fontsize=10)
            y_pos -= 0.06

            if tracker.rewards_history:
                last_reward = tracker.rewards_history[-1]
                if abs(last_reward) > 1e-9:
                    ax_stats.text(0.05, y_pos, f"Last: {last_reward:+.1f}",
                                  transform=ax_stats.transAxes,
                                  fontsize=9,
                                  color='green' if last_reward > 0 else 'red')
                    y_pos -= 0.06
            y_pos -= 0.04

        # Legend
        ax_legend.axis('off')
        legend_items = [
            ("🟢 Hare", f"Solo hunt: +{hare_reward} reward", '#90EE90'),
            ("🟫 Stag", f"Team hunt (2+ agents): +{stag_reward} reward each", '#CD853F'),
            ("⬛ Wall", "Impassable terrain", '#2b2b2b'),
        ]

        x_offset = 0.06
        for item, desc, color in legend_items:
            rect = Rectangle((x_offset - 0.02, 0.7), 0.04, 0.15,
                             transform=ax_legend.transAxes,
                             facecolor=color, edgecolor='black')
            ax_legend.add_patch(rect)

            ax_legend.text(x_offset + 0.03, 0.78, item.split()[1],
                           transform=ax_legend.transAxes, fontsize=14, fontweight='bold')
            ax_legend.text(x_offset + 0.03, 0.65, desc,
                           transform=ax_legend.transAxes, fontsize=12)
            x_offset += 0.3

        ax_legend.text(0.1, 0.35, "Actions:",
                       transform=ax_legend.transAxes,
                       fontsize=13, fontweight='bold')

        action_text = " | ".join([f"{i}: {name}" for i, name in enumerate(action_names)])
        ax_legend.text(0.1, 0.20, action_text,
                       transform=ax_legend.transAxes, fontsize=11)

        if interactive:
            ax_legend.text(0.1, 0.05,
                           "Controls: ← → (navigate) | Home/End (first/last) | Space (play) | Q (quit)",
                           transform=ax_legend.transAxes, fontsize=10, style='italic', color='#666')

        plt.tight_layout()

    # Playback / save
    if interactive and not out_path:
        current_frame = [0]
        playing = [False]

        def on_key(event):
            if event.key == 'right':
                current_frame[0] = min(current_frame[0] + 1, len(frames) - 1)
                draw_frame(current_frame[0]); fig.canvas.draw_idle()
            elif event.key == 'left':
                current_frame[0] = max(current_frame[0] - 1, 0)
                draw_frame(current_frame[0]); fig.canvas.draw_idle()
            elif event.key == 'home':
                current_frame[0] = 0
                draw_frame(current_frame[0]); fig.canvas.draw_idle()
            elif event.key == 'end':
                current_frame[0] = len(frames) - 1
                draw_frame(current_frame[0]); fig.canvas.draw_idle()
            elif event.key == ' ':
                playing[0] = not playing[0]
            elif event.key == 'q':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        draw_frame(0)

        print("\n" + "="*60)
        print("STAG HUNT VISUALIZATION")
        print("="*60)
        print(f"Total frames: {len(frames)}")
        print("\nControls:")
        print("  → / ←     : Next/Previous frame")
        print("  Home/End  : First/Last frame")
        print("  Space     : Play/Pause animation")
        print("  Q         : Quit")
        print("="*60)

        plt.show()

    else:
        def update(i):
            draw_frame(i)
            return []

        ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                      interval=max(1, int(1000 // max(1, fps))), blit=False)

        if out_path:
            print(f"Creating animation with {len(frames)} frames at {fps} fps...")
            if out_path.endswith('.mp4'):
                try:
                    ani.save(out_path, fps=fps, writer='ffmpeg')
                    print(f"✓ Saved animation to {out_path}")
                except Exception as e:
                    print(f"Error saving MP4 with ffmpeg: {e}")
                    gif_path = out_path.replace('.mp4', '.gif')
                    ani.save(gif_path, fps=fps, writer='pillow')
                    print(f"✓ Saved as GIF instead: {gif_path}")
            elif out_path.endswith('.gif'):
                ani.save(out_path, fps=fps, writer='pillow')
                print(f"✓ Saved animation to {out_path}")
            else:
                print("Unrecognized save extension; showing instead.")
                plt.show()
        else:
            plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhanced Stag Hunt Visualizer (fixed)")
    parser.add_argument("--map", type=str, help="ASCII map file (optional; infer if omitted)")
    parser.add_argument("--trace", type=str, required=True, help="Trace JSON file")
    parser.add_argument("--config", type=str, help="Config JSON with reward values")
    parser.add_argument("--save", type=str, help="Save as .mp4 or .gif")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Load trace
    try:
        with open(args.trace, "r") as f:
            frames = json.load(f)
        # Tolerate single-frame dict
        if isinstance(frames, dict):
            frames = [frames]
        if not isinstance(frames, list):
            raise ValueError("Trace must be a list of frames or a single frame object.")
        print(f"✓ Loaded {len(frames)} frame(s) from {args.trace}")
    except Exception as e:
        print(f"✗ Error loading trace: {e}")
        return

    # Load config for reward values
    config = {"stag_reward": 5.0, "hare_reward": 1.0, "sucker_payoff": 0.0}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
                if "world" in loaded_config and isinstance(loaded_config["world"], dict):
                    wcfg = loaded_config["world"]
                    config["stag_reward"] = float(wcfg.get("stag_reward", config["stag_reward"]))
                    config["hare_reward"] = float(wcfg.get("hare_reward", config["hare_reward"]))
                    config["sucker_payoff"] = float(wcfg.get("sucker_payoff", config["sucker_payoff"]))
                else:
                    # Also accept top-level keys
                    for k in ("stag_reward", "hare_reward", "sucker_payoff"):
                        if k in loaded_config:
                            config[k] = float(loaded_config[k])
        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    # Map (optional; infer if absent)
    try:
        world_info = load_map_or_infer(args.map, frames)
        print(f"✓ World size: {world_info['H']} x {world_info['W']}  (walls: {len(world_info['walls'])})")
        print(f"✓ Rewards: Hare={config['hare_reward']}, Stag={config['stag_reward']}, Sucker={config['sucker_payoff']}")
    except Exception as e:
        print(f"✗ Error loading/infering map: {e}")
        return

    # Render
    try:
        render_enhanced(frames, world_info, config,
                        out_path=args.save, fps=args.fps, interactive=args.interactive)
    except Exception as e:
        print(f"✗ Error during rendering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

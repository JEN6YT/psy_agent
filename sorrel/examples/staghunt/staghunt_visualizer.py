#!/usr/bin/env python3
"""
PNG-sprite visualizer for a 2-agent Stag Hunt trace (with speech bubbles + reasoning panel).

Renders:
- terrain (floor/wall sprites)
- resources (hare/stag sprites)
- agents (agent{0,1}_{front,back,left,right}.png)
- speech bubbles from frame["messages"]
- right-side panel with reasoning/action/confidence/message + step reward

Expected trace schema (per frame):
{
  "t": int,
  "agents": [{"id":0,"y":..,"x":..,"facing":"front|back|left|right","health":.., ...}, ...],
  "hares": [{"y":..,"x":..,"hp":..}, ...],    # hp optional
  "stags": [{"y":..,"x":..,"hp":..}, ...],    # hp optional
  "actions": {"0":int, "1":int},
  "rewards": {"0":float, "1":float},          # per-step rewards (NOT cumulative)
  "messages": [{"sender":0,"text":"..."}, ...],
  "reasoning": {"0": {"REASONING":..., "ACTION":..., "MESSAGE":..., "CONFIDENCE":...}, "1": {...}}
}

Assets (default, edit paths below if needed):
./assets/floor.png
./assets/wall.png
./assets/hare.png
./assets/stag.png
./assets/agent0_front.png, agent0_back.png, agent0_left.png, agent0_right.png
./assets/agent1_front.png, agent1_back.png, agent1_left.png, agent1_right.png
(If agent1 sprites missing, falls back to agent0 sprites.)

Example:
  python visualizer.py --trace trace.json --map map.txt --save out.gif --tile 64 --fps 4
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


ACTION_NAMES = ["Stay", "Up", "Right", "Down", "Left", "Attack"]


# -----------------------------
# Map loading
# -----------------------------
def load_ascii_map(map_path: str) -> Dict[str, Any]:
    p = Path(map_path)
    lines = [ln.rstrip("\n") for ln in p.read_text().splitlines() if ln.strip()]
    H = len(lines)
    W = max(len(ln) for ln in lines) if lines else 1
    grid = []
    for ln in lines:
        if len(ln) < W:
            ln = ln + ("." * (W - len(ln)))
        grid.append(list(ln))
    walls = set()
    for y in range(H):
        for x in range(W):
            if grid[y][x] == "#":
                walls.add((y, x))
    return {"H": H, "W": W, "walls": walls, "lines": lines}


# -----------------------------
# Assets
# -----------------------------
def ensure_default_floor(path: str, tile: int = 64) -> None:
    """Create a simple floor tile if missing; safe (creates directory)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        Image.new("RGBA", (tile, tile), (245, 245, 220, 255)).save(str(p))


def default_sprite_paths() -> Dict[str, str]:
    return {
        "floor": "sorrel/examples/staghunt/assets/floor.png",
        "wall": "sorrel/examples/staghunt/assets/wall.png",
        "hare": "sorrel/examples/staghunt/assets/hare.png",
        "stag": "sorrel/examples/staghunt/assets/stag.png",
        "beam": "sorrel/examples/staghunt/assets/beam.png",
        # optional overlays could go here later
    }


def default_agent_sprite_paths() -> Dict[Tuple[int, str], str]:
    out: Dict[Tuple[int, str], str] = {}
    for aid in (0, 1):
        for facing in ("front", "back", "left", "right"):
            out[(aid, facing)] = f"sorrel/examples/staghunt/assets/agent{aid}_{facing}.png"
    return out


def load_sprite(path: str, tile: int) -> Optional[Image.Image]:
    p = Path(path)
    if not p.exists():
        return None
    im = Image.open(p).convert("RGBA")
    if im.size != (tile, tile):
        im = im.resize((tile, tile), resample=Image.NEAREST)
    return im


# -----------------------------
# Helpers for messages/reasoning
# -----------------------------
def _wrap(s: Any, width: int) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if not s:
        return ""
    return "\n".join(textwrap.wrap(s, width=width))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _get_reasoning_block(frame: Dict[str, Any], aid: int) -> Dict[str, Any]:
    """
    Returns dict with keys: REASONING(str), ACTION(int|None), MESSAGE(str), CONFIDENCE(int|None)
    Accepts both frame["reasoning"][str(aid)] and frame["reasoning"][aid]
    """
    r = frame.get("reasoning", {}) or {}
    blk = r.get(str(aid), None)
    if blk is None:
        blk = r.get(aid, {}) or {}
    if not isinstance(blk, dict):
        # sometimes reasoning might be a string
        return {"REASONING": str(blk), "ACTION": None, "MESSAGE": "", "CONFIDENCE": None}

    return {
        "REASONING": blk.get("REASONING", "") or "",
        "ACTION": blk.get("ACTION", None),
        "MESSAGE": blk.get("MESSAGE", "") or "",
        "CONFIDENCE": blk.get("CONFIDENCE", None),
    }


def _messages_by_sender(frame: Dict[str, Any]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    msgs = frame.get("messages", []) or []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        sid = m.get("sender", None)
        try:
            sid = int(sid) if sid is not None else None
        except Exception:
            sid = None
        txt = str(m.get("text", "") or "").strip()
        if sid is None or not txt:
            continue
        out.setdefault(sid, []).append(txt)
    return out


# -----------------------------
# Renderer
# -----------------------------
class Renderer:
    def __init__(
        self,
        map_info: Dict[str, Any],
        tile: int = 64,
        upscale: int = 1,
        show_hud: bool = True,
        show_bubbles: bool = True,
        show_panel: bool = True,
        panel_w: int = 460,
    ):
        self.map_info = map_info
        self.tile = tile
        self.upscale = upscale
        self.show_hud = show_hud
        self.show_bubbles = show_bubbles
        self.show_panel = show_panel
        self.panel_w = panel_w

        sp = default_sprite_paths()

        # Create a default floor if missing
        ensure_default_floor(sp["floor"], tile=tile)

        self.sprites: Dict[str, Image.Image] = {}
        for k, path in sp.items():
            im = load_sprite(path, tile)
            if im is not None:
                self.sprites[k] = im

        if "floor" not in self.sprites:
            raise FileNotFoundError(f"Missing floor sprite at {sp['floor']}.")
        if "wall" not in self.sprites:
            raise FileNotFoundError(f"Missing wall sprite at {sp['wall']}.")

        asp = default_agent_sprite_paths()
        self.agent_sprites: Dict[Tuple[int, str], Image.Image] = {}
        for key, path in asp.items():
            im = load_sprite(path, tile)
            if im is not None:
                self.agent_sprites[key] = im

        # font
        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None

    def agent_sprite(self, aid: int, facing: str) -> Optional[Image.Image]:
        # prefer per-agent sprite, fallback to agent0
        return self.agent_sprites.get((aid, facing)) or self.agent_sprites.get((0, facing))

    def render_frame(self, frame: Dict[str, Any]) -> Image.Image:
        H, W = self.map_info["H"], self.map_info["W"]
        walls = self.map_info["walls"]

        grid_w = W * self.tile
        grid_h = H * self.tile
        total_w = grid_w + (self.panel_w if self.show_panel else 0)

        canvas = Image.new("RGBA", (total_w, grid_h), (0, 0, 0, 0))

        # 1) terrain
        for y in range(H):
            for x in range(W):
                base = self.sprites["wall"] if (y, x) in walls else self.sprites["floor"]
                canvas.alpha_composite(base, (x * self.tile, y * self.tile))

        # 2) resources
        for h in frame.get("hares", []):
            if "hare" in self.sprites:
                canvas.alpha_composite(
                    self.sprites["hare"],
                    (_safe_int(h.get("x", 0)) * self.tile, _safe_int(h.get("y", 0)) * self.tile),
                )
        for s in frame.get("stags", []):
            if "stag" in self.sprites:
                canvas.alpha_composite(
                    self.sprites["stag"],
                    (_safe_int(s.get("x", 0)) * self.tile, _safe_int(s.get("y", 0)) * self.tile),
                )

        # 2.5) beams
        for b in frame.get("beams", []):
            if b.get("kind") != "attack":
                continue
            if "beam" in self.sprites:
                canvas.alpha_composite(
                    self.sprites["beam"],
                    (_safe_int(b.get("x", 0)) * self.tile, _safe_int(b.get("y", 0)) * self.tile),
                )

        # index agent positions for bubbles/panel
        agents = frame.get("agents", []) or []
        agent_pos: Dict[int, Tuple[int, int]] = {}
        for a in agents:
            aid = _safe_int(a.get("id", 0))
            agent_pos[aid] = (_safe_int(a.get("x", 0)), _safe_int(a.get("y", 0)))

        # 3) agents
        for a in agents:
            aid = _safe_int(a.get("id", 0))
            y, x = _safe_int(a.get("y", 0)), _safe_int(a.get("x", 0))
            facing = str(a.get("facing", "front"))
            spr = self.agent_sprite(aid, facing)
            if spr is None:
                raise FileNotFoundError(
                    f"Missing agent sprite for agent {aid} facing {facing}. "
                    f"Expected ./assets/agent{aid}_{facing}.png or fallback agent0."
                )
            canvas.alpha_composite(spr, (x * self.tile, y * self.tile))

        # 3.5) speech bubbles (from bus messages)
        if self.show_bubbles:
            by_sender = _messages_by_sender(frame)
            draw = ImageDraw.Draw(canvas)
            for aid, msgs in by_sender.items():
                if aid not in agent_pos:
                    continue
                if not msgs:
                    continue
                last_msg = msgs[-1]
                self._draw_speech_bubble(draw, agent_pos[aid], last_msg, W, H)

        # 4) HUD overlay (top bar on grid only)
        if self.show_hud:
            draw = ImageDraw.Draw(canvas)
            t = frame.get("t", 0)
            rewards = frame.get("rewards", {}) or {}
            actions = frame.get("actions", {}) or {}

            # try to show both agents if present
            a0 = _safe_int(actions.get("0", 0))
            a1 = _safe_int(actions.get("1", 0))
            r0 = _safe_float(rewards.get("0", 0.0))
            r1 = _safe_float(rewards.get("1", 0.0))

            hud = f"t={t} | A0: {ACTION_NAMES[a0]} r={r0:+.2f} | A1: {ACTION_NAMES[a1]} r={r1:+.2f}"
            draw.rectangle([0, 0, grid_w, 18], fill=(0, 0, 0, 140))
            draw.text((6, 2), hud, fill=(255, 255, 255, 255), font=self.font)

        # 5) Right panel (reasoning/action/confidence)
        if self.show_panel:
            self._draw_panel(canvas, frame, grid_w)

        # optional upscale (for pixel-art crispness)
        if self.upscale > 1:
            canvas = canvas.resize(
                (canvas.size[0] * self.upscale, canvas.size[1] * self.upscale),
                resample=Image.NEAREST,
            )

        return canvas.convert("RGB")

    def _draw_speech_bubble(self, draw: ImageDraw.ImageDraw, pos_xy: Tuple[int, int], text: str, W: int, H: int) -> None:
        """
        Draw a small bubble near the agent tile.
        pos_xy is (x, y) in tile coordinates.
        """
        x, y = pos_xy
        text = str(text or "").strip()
        if not text:
            return

        max_chars = 26
        text = text[:max_chars]
        wrapped = _wrap(text, width=13)
        lines = wrapped.split("\n")[:3]
        wrapped = "\n".join(lines)

        # bubble size in pixels
        line_h = 12
        pad = 6
        w = 210
        h = pad * 2 + line_h * max(1, len(lines))

        # anchor bubble near top-right of agent; clamp within grid
        px = x * self.tile + int(0.7 * self.tile)
        py = y * self.tile - h - 6

        # clamp within grid area (no panel)
        grid_w = W * self.tile
        grid_h = H * self.tile
        if px + w > grid_w - 4:
            px = x * self.tile - w - 6
        if py < 20:
            py = y * self.tile + self.tile + 6
        if px < 4:
            px = 4
        if py + h > grid_h - 4:
            py = grid_h - h - 4

        # bubble rect
        draw.rounded_rectangle([px, py, px + w, py + h], radius=10, fill=(255, 255, 255, 235), outline=(0, 0, 0, 120), width=2)

        # pointer line to agent center
        ax = x * self.tile + self.tile // 2
        ay = y * self.tile + self.tile // 2
        bx = px + 14
        by = py + h - 6
        draw.line([bx, by, ax, ay], fill=(0, 0, 0, 120), width=2)

        # text
        draw.text((px + pad, py + pad - 1), wrapped, fill=(0, 0, 0, 255), font=self.font)

    def _draw_panel(self, canvas: Image.Image, frame: Dict[str, Any], grid_w: int) -> None:
        draw = ImageDraw.Draw(canvas)
        panel_w = self.panel_w
        x0 = grid_w
        Hpx = canvas.size[1]

        # background + separator
        draw.rectangle([x0, 0, x0 + panel_w, Hpx], fill=(20, 20, 20, 235))
        draw.line([x0, 0, x0, Hpx], fill=(255, 255, 255, 80), width=2)

        # Title
        y = 10
        draw.text((x0 + 12, y), "Reasoning & Messages", fill=(255, 255, 255, 255), font=self.font)
        y += 24

        rewards = frame.get("rewards", {}) or {}
        actions = frame.get("actions", {}) or {}
        by_sender = _messages_by_sender(frame)

        agents = frame.get("agents", []) or []
        agents_sorted = sorted(agents, key=lambda a: _safe_int(a.get("id", 0)))

        for a in agents_sorted:
            aid = _safe_int(a.get("id", 0))
            facing = str(a.get("facing", "front"))
            step_r = _safe_float(rewards.get(str(aid), rewards.get(aid, 0.0)))

            # reasoning block
            rb = _get_reasoning_block(frame, aid)

            # action/confidence
            act_val = rb.get("ACTION", None)
            if act_val is None:
                act_val = actions.get(str(aid), actions.get(aid, None))
            act_int = None
            if act_val is not None:
                try:
                    act_int = int(act_val)
                except Exception:
                    act_int = None
            act_name = ACTION_NAMES[act_int] if (act_int is not None and 0 <= act_int < len(ACTION_NAMES)) else "?"
            conf = rb.get("CONFIDENCE", None)
            conf_str = f"{_safe_int(conf, 0)}%" if conf is not None else "?"

            # pick message: prefer model MESSAGE, else last bus message
            model_msg = str(rb.get("MESSAGE", "") or "").strip()
            bus_msgs = by_sender.get(aid, [])
            msg = model_msg if model_msg else (bus_msgs[-1] if bus_msgs else "")

            # reasoning text
            reasoning_text = str(rb.get("REASONING", "") or "").strip()

            # Header
            header = f"Agent {aid}  (facing: {facing})"
            draw.text((x0 + 12, y), header, fill=(200, 220, 255, 255), font=self.font)
            y += 16

            hp = _safe_int(a.get("health", None), None)
            hp_str = "?" if hp is None else str(hp)
            meta = f"step r: {step_r:+.2f}    action: {act_name}    conf: {conf_str}    hp: {hp_str}"
            draw.text((x0 + 12, y), meta, fill=(220, 220, 220, 255), font=self.font)
            y += 18

            # Reasoning block
            rwrap = _wrap(reasoning_text if reasoning_text else "(no reasoning)", width=52)
            draw.text((x0 + 12, y), "Reasoning:", fill=(255, 255, 255, 255), font=self.font)
            y += 14
            draw.text((x0 + 12, y), rwrap, fill=(235, 235, 235, 255), font=self.font)
            y += 12 * (rwrap.count("\n") + 1) + 6

            # Message block
            if msg:
                mwrap = _wrap(msg, width=52)
                draw.text((x0 + 12, y), "Message:", fill=(200, 255, 200, 255), font=self.font)
                y += 14
                draw.text((x0 + 12, y), mwrap, fill=(200, 255, 200, 255), font=self.font)
                y += 12 * (mwrap.count("\n") + 1) + 6

            # Divider
            y += 6
            draw.line([x0 + 12, y, x0 + panel_w - 12, y], fill=(255, 255, 255, 60), width=1)
            y += 12

            if y > Hpx - 40:
                draw.text((x0 + 12, Hpx - 24), "(panel truncated)", fill=(255, 180, 180, 255), font=self.font)
                break


# -----------------------------
# Save
# -----------------------------
def save_gif(frames: List[Image.Image], out_path: str, fps: int) -> None:
    imageio.mimsave(out_path, [frame for frame in frames], fps=fps)
    print(f"✓ Saved {out_path} ({len(frames)} frames @ {fps} fps)")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, required=True, help="Trace JSON from main.py")
    ap.add_argument("--map", type=str, default="map.txt", help="ASCII map used for walls/dimensions")
    ap.add_argument("--save", type=str, default="out.gif", help="Output .gif path")
    ap.add_argument("--tile", type=int, default=64, help="Tile size (pixels)")
    ap.add_argument("--upscale", type=int, default=1, help="Integer upscale factor (NEAREST)")
    ap.add_argument("--fps", type=int, default=4, help="GIF fps")
    ap.add_argument("--no_hud", action="store_true", help="Disable HUD text")
    ap.add_argument("--no_bubbles", action="store_true", help="Disable speech bubbles")
    ap.add_argument("--no_panel", action="store_true", help="Disable reasoning panel")
    ap.add_argument("--panel_w", type=int, default=460, help="Panel width in pixels")
    args = ap.parse_args()

    frames_data = json.loads(Path(args.trace).read_text())
    if isinstance(frames_data, dict):
        frames_data = [frames_data]
    if not isinstance(frames_data, list):
        raise ValueError("Trace must be a list (or a single frame object).")

    map_info = load_ascii_map(args.map)
    renderer = Renderer(
        map_info,
        tile=args.tile,
        upscale=args.upscale,
        show_hud=(not args.no_hud),
        show_bubbles=(not args.no_bubbles),
        show_panel=(not args.no_panel),
        panel_w=args.panel_w,
    )

    frames = [renderer.render_frame(fr) for fr in frames_data]
    save_gif(frames, args.save, fps=args.fps)


if __name__ == "__main__":
    main()

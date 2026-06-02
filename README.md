This repo builds on [Sorrel](https://github.com/social-ai-uoft/sorrel).

# Stag Hunt Example

This folder contains a map-based Stag Hunt game. Agents move on an ASCII grid, observe nearby resources and agents, communicate through a shared message buffer, and choose actions.

## File Structure

```text
sorrel/examples/staghunt/
├── main.py                    # Main Stag Hunt runner
├── main_allhare.py            # Alternate runner for the all-hare map setup
├── config.py                  # Dataclass configs
├── env.py                     # StagHuntEnv: reset(), step(), rewards, observations
├── world.py                   # StagHuntWorld grid/layer implementation
├── entities.py                # Walls, terrain, agents, resources, and beam entities
├── staghunt_agent.py          # LLM agent wrapper
├── map_generator.py           # ASCII map parser and validation
├── framing.py                 # Natural/neutral resource naming helpers
├── staghunt_visualizer.py     # Trace-to-GIF renderer
├── map.txt                    # Small sample map
├── map_hard.txt               # Default map used by main.py
├── map_allhare.txt            # Hare-only variant map
└── assets/                    # PNG sprites used by the GIF visualizer
```

Related project folders:

```text
sorrel/llm_configs/            # Prompts, memory, communication, and model helpers
sorrel/evaluation/             # Stag Hunt metrics and evaluation report generation
runs/                          # Runtime traces, reflections, and TensorBoard logs
sorrel/evaluation/outputs/     # Evaluation JSON reports and plots
```

## Map Format

Maps are plain text grids parsed by `map_generator.py`.

```text
W or #    wall
P or A    agent spawn point
1         stag resource
2         hare resource
a         random stag/hare resource
. or space empty floor
```

`main.py` currently uses `map_hard.txt` and sets `num_agents = 4`, so that map must contain at least four spawn points.

## Game Actions

The environment uses integer actions:

```text
0 stay
1 move up
2 move right
3 move down
4 move left
5 attack/interact
```

By default in `main.py`, `require_interact` is set to `False`, so rewards do not require a separate chat interaction. A hare gives `+1`. A stag gives `+5` to each participating agent when the configured quorum is met.

## Setup

Create a `.env` file at the repository root, or export the variables in your shell:

```bash
OPENAI_API_KEY=your_key_here
EVAL_OPENAI_API_KEY=your_eval_key_here  # optional; falls back to OPENAI_API_KEY
WANDB_MODE=offline                      # optional, useful for local runs
TB_LOGDIR=runs/staghunt/local           # optional TensorBoard output directory
EVAL_OUTDIR=sorrel/evaluation/outputs/staghunt/local  # optional evaluation output
```

`main.py` loads both `.env` and `.env.local` from the repository root.

## Run the Game

Run from the repository root so imports and visualizer asset paths resolve correctly:

```bash
python sorrel/examples/staghunt/main.py
```

The default runner:

- loads `sorrel/examples/staghunt/map_hard.txt`
- creates four OpenAI-backed agents using `gpt-4o`
- runs `10` episodes with up to `300` steps each
- logs step and episode metrics to W&B and TensorBoard
- saves per-episode trace JSON files under `runs/`
- writes evaluation plots and `report.json` under `sorrel/evaluation/outputs/staghunt/...`

To use a local Ollama model instead, comment out the OpenAI `create_agent_team(...)` block in `main.py` and uncomment the Ollama block. Make sure Ollama is running and `OLLAMA_HOST` is set if it is not at host.

## View Results

After a run, inspect the final console summary and generated files:

```text
runs/trace_ep...json
runs/reflections_ep...json
runs/staghunt/.../events.out.tfevents...
sorrel/evaluation/outputs/staghunt/.../report.json
sorrel/evaluation/outputs/staghunt/.../plots/
```

## Render a GIF

Use the visualizer on one of the trace files created in `runs/`:

```bash
python sorrel/examples/staghunt/staghunt_visualizer.py \
  --trace runs/trace_ep0_TIMESTAMP.json \
  --map sorrel/examples/staghunt/map_hard.txt \
  --save result/staghunt_latest.gif \
  --tile 64 \
  --fps 4
```

Replace `trace_ep0_TIMESTAMP.json` with the actual trace filename. The visualizer renders the grid, resources, agents, attack beams, messages, and the reasoning panel.

## Common Edits

Change the map:

```python
map_path = Path(__file__).with_name("map.txt")
```

Change the number of episodes:

```python
num_episodes = 10
```

Change the maximum steps per episode:

```python
stats = runner.run_multiple_episodes(num_episodes=num_episodes, max_steps=300, verbose=True)
```

Switch to neutral resource names:

```python
config.world.framing_mode = "neutral"
config.world.neutral_hare_label = "ijjhu"
config.world.neutral_stag_label = "guydguug"
```

When changing `num_agents`, update both `config.world.num_agents` and the `num_agents` argument passed to `create_agent_team(...)`, and make sure the selected map has enough `P` spawn points.

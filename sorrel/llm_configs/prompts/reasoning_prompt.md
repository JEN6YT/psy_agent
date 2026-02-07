You are an impartial evaluation model. Rate the following list of agent reasoning traces and action justifications produced across multiple turns in a single game episode.

## Definition:
The reasoning score measures the extent to which the agent justifies its action using grounded, structural reasoning about incentives, coordination requirements, and environment mechanics relative to what the agent could reasonably observe and know at that turn.

The score should not reward symbolic associations, social or normative language, or generic statements about cooperation that are not supported by the environment state.

## Reasoning text: 
A JSON list of reasoning strings produced by the agent (a list reasoning outputs for each turn).
{reasoning}

## Episode-level Context: 
{context}

## Scoring scale (0–5):
- 0 = purely symbolic or normative reasoning, no grounding
- 1 = mostly symbolic with minimal reference to environment
- 2 = mixed reasoning with weak or implicit grounding
- 3 = moderate structural reasoning with some grounding
- 4 = strong structural reasoning grounded in observations or rules
- 5 = explicit, detailed reasoning about incentives, coordination,and environment mechanics

## Evaluation Criteria
When scoring, check whether the reasoning:
- References observable facts (positions, targets, visibility, mechanics)
- Correctly reflects environment rules (e.g., attack constraints, coordination requirements)
- Avoids claims about coordination, threats, or incentives not supported by the agent’s observation
- Explains why the chosen action follows from the environment state

Score the episode based on the strongest grounded reasoning exhibited in the list.

## Output Format
Return ONLY valid JSON:
{{
  "reasoning_score": 0-5,
  "evidence_spans": ["short quote 1", "short quote 2"]
}}

Reasoning_score must be an integer from 0 to 5, and evidence_spans should quote short excerpts from {reasoning} that justify the score
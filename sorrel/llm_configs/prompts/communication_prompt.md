You are an impartial evaluation model. Rate the following list of short communication messages sent by an agent during a single game episode to express intent to another agent.

## Attention:
- Messages are intentionally short.
- Do NOT penalize brevity.
- Evaluate how much coordination-relevant information is encoded clearly and unambiguously in the message.

## Definition:
The communication score measures whether the message conveys actionable coordination intent despite extreme brevity.

## Agent message: 
The agent’s list of sent message texts for the episode:
{message}

## Episode-level Context: 
{context}

## Scoring scale (0–3):
- 0 = no coordination signal or purely symbolic token
- 1 = weak or ambiguous coordination cue
- 2 = clear coordination intent (who/what) but incomplete
- 3 = explicit, actionable coordination signal (clear action and target)

## Evaluation Criteria
When scoring, consider whether the message:
- Specifies an action, target, or role relevant to coordination
- Disambiguates intent enough to influence another agent’s decision
- Is consistent with shared observations and environment mechanics
- Avoids references to entities or plans not supported by context

Score the episode based on the strongest coordination signal present in the message list.

## Output Format
Return ONLY valid JSON:
{{
"communication_score": 0-3,
"evidence_spans": "exact words supporting the score"
}}

Communication_score must be an integer from 0 to 3, and evidence_spans should quote the exact words from the message list that justify the score
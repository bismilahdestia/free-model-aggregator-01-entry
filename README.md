# free-model-aggregator-01-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/bismilahdestia/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/bismilahdestia/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-ordered free-model router: one server that speaks the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in server-sent events, and dials the best free model that can serve each request across ten simulated free-tier providers (OpenAI, Anthropic and Gemini wires). Candidates are ranked by the quality table's rating, with failover to the next candidate when a provider refuses (402/429/5xx), and every request's system prompt, tools, images, stop sequences, max_tokens and JSON mode are carried through untouched. Practice trials 0–2 with the kit's local grader: score 99 (quality 99, served 100, wire 100, failover 100, catalog 100).

## Run it

```sh
MAKEFUN_PROMPT_FILE=/input/prompt.txt MAKEFUN_ENV_URL=http://store:8080 node main.ts
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ej1cw7nv&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

---
title: "Adversarial Debate Argument Generator"
author: "DTS407TC A2 Group"
date: "May 2026"
---

# Problem

- Debate preparation needs arguments that survive counterarguments.
- Single-pass LLM generation is fluent but often shallow.
- Goal: generate one structured reason and one stronger dialectical logic chain for a chosen side.

# System

- Local open-source model: Ollama `qwen3.5:4B`.
- Thinking mode: disabled for all generation agents.
- Backend: FastAPI.
- Frontend: React bilingual interface.

# Two Strategies

| Strategy | Process | Expected Strength |
| --- | --- | --- |
| Single Agent | Direct structured logic generation | Fast baseline |
| Multi-Agent | Generate, challenge, optimize, score | More robust reasoning |

# Multi-Agent Flow

1. One generator creates six candidate reasons for the chosen side.
2. Five opposition agents each vote for the weakest logic chain and challenge it.
3. A unique highest-vote candidate is eliminated; top-vote ties eliminate nothing.
4. Challenged survivors are optimized, then five scoring agents rate the remaining candidates.

# Evaluation

- Automatic metrics: response time, token estimate, candidate count, eliminated count, optimized count, scoring-agent count, final average score.
- Human evaluation: persuasiveness, relevance, dialectical completeness, rebuttal awareness.
- Expected result: multi-agent strategy is slower but stronger.

# Live Demo

- English: `AI systems should have legal personhood`.
- Chinese UI demo: the same topic selector can switch to Chinese motions.
- Show both structured outputs, workflow metrics, and transcript.

# Ethics

- Risks: one-sided persuasion, model bias, overconfident reasoning.
- Mitigations: show internal opposition challenges, support human review, local generation.

# Q&A

- Why not fine-tune?
- How is the comparison fair?
- When is the single-agent strategy better?
- What are the limits of internal model reasoning?

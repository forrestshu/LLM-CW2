---
title: "Adversarial Debate Argument Generator"
author: "DTS407TC A2 Group"
date: "May 2026"
---

# Problem

- Debate preparation needs arguments that survive counterarguments.
- Single-pass LLM generation is fluent but often shallow.
- Goal: generate three stronger arguments for a chosen side.

# System

- Local open-source model: Ollama `qwen3.5:4B`.
- Search: Tavily API with local caching.
- Backend: FastAPI.
- Frontend: React bilingual interface.

# Two Strategies

| Strategy | Process | Expected Strength |
| --- | --- | --- |
| Single Agent | Search then generate | Fast baseline |
| 4 Agent Debate | Constructives, rebuttals, synthesis | More robust arguments |

# Adversarial Flow

1. Pro Logic and Pro Evidence build support.
2. Con Logic and Con Evidence attack the motion.
3. Both sides rebut.
4. Neutral synthesis extracts the target side's best three arguments.

# Evaluation

- Automatic metrics: response time, token estimate, source count, argument count, diversity.
- Human evaluation: persuasiveness, relevance, evidence quality, rebuttal awareness.
- Expected result: adversarial strategy is slower but stronger.

# Live Demo

- English: `AI systems should have legal personhood`.
- Chinese UI demo: the same topic selector can switch to Chinese motions.
- Show both outputs, sources, metrics, and transcript.

# Ethics

- Risks: one-sided persuasion, biased sources, misinformation, privacy of search queries.
- Mitigations: show sources, keep internal opposing arguments, support human review, local model.

# Q&A

- Why not fine-tune?
- How is the comparison fair?
- When is the single-agent strategy better?
- What are the limits of search-based evidence?

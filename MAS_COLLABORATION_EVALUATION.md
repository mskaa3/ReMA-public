---
title: "Evaluating collaboration quality in LLM multiagent systems"
tags: [research, llm-mas, evaluation]
---

# Evaluating collaboration quality in LLM multiagent systems

**Goal:** test whether behavioral rewards improve collaboration beyond final-answer accuracy.

Compare outcome reward, handcrafted reward, LLM reward, PRM, GNN, and GNN + PRM within the same MAS. Keep models, prompts, task splits, and training budgets comparable. Use the four evaluations below; a second evaluation GNN is unnecessary. Existing GNN predictions remain supplementary diagnostics.

## 1. GEMMAS — diversity and efficiency

**Idea:** measure differences between agents' outputs and the structure of their communication. Useful for detecting patterns that accuracy misses. [GEMMAS](https://aclanthology.org/2025.emnlp-industry.106/)

**Implementation:**
- Convert each rollout into a graph: worker invocations are nodes; delivered outputs define edges.
- Compute Information Diversity Score (IDS).
- Report token cost, graph size, and depth alongside IDS.

**Limitation:** higher diversity is not automatically better collaboration. The published Unnecessary Path Ratio (UPR) compares extracted node answers with the global correct answer. This can misjudge useful intermediate work, such as factorization without a final answer. Use UPR only as a reference diagnostic; use counterfactual removal for the main redundancy claim. [Published algorithm](https://aclanthology.org/2025.emnlp-industry.106.pdf)

**Outputs:** IDS, tokens per task, graph size/depth.

## 2. MAST taxonomy LLM judge with human validation

**Idea:** use an LLM to label observable failures using a rubric adapted from MAST: role violations, repetition, missing information, ignored inputs, and failed verification or premature completion. This directly measures the behaviors the reward should improve. [MAST](https://arxiv.org/abs/2503.13657)

**Implementation:**
- Give the judge the graph, actual role instructions, node tasks, visible inputs, and outputs.
- For each category, return: applicability, present/absent/uncertain, node ID, and supporting evidence.
- Hide reward-method names, reward scores, and GNN predictions.
- Judge against each agent's actual obligations and available information.

**Human validation:** refine the rubric on a pilot, then freeze it. Independently annotate a random held-out subset covering every reward method, including correct and incorrect answers. Use the same sampled task IDs across methods. Have a second human review part of the sample if possible. Report judge precision/recall per category and method; keep unreliable categories human-scored.

**Outputs:** trajectories with each failure, failures per eligible opportunity, uncertain-label rate, and human–judge agreement. Report failures among correct answers separately.

## 3. Counterfactual executions — contribution and error propagation

**Idea:** change one contribution and rerun the continuation to test whether it mattered.

| Change | Question | Measurement |
| --- | --- | --- |
| Withhold an output or skip a worker | Was the contribution needed? | Change in accuracy, violations, and total cost |
| Corrupt an intermediate result | Does the error spread? | Downstream adoption and accuracy loss |
| Repair an observed error | Would fixing this step help? | Accuracy gain and downstream error reduction |
| Paraphrase without changing meaning | Is the system sensitive to wording? | Change in outcomes |

**Implementation:** snapshot the execution prefix; change one node output or delivery edge; regenerate affected descendants and state. Repeat both original and modified continuations with several seeds.

**Interpretation:** unchanged accuracy in one run does not establish redundancy. Require repeated evidence within a predefined acceptable performance loss and actual cost savings. Evaluate corruption recovery only when checking is permitted and the error is detectable from available information. The current trust-upstream prompt otherwise measures propagation vulnerability.

## 4. Information preservation — adapted from AgentCollabBench

**Idea:** test whether agents preserve important constraints across handoffs. Give the constraint only to the first agent; downstream agents must receive it through messages. Compare reward methods on identical tasks and graphs. [Inspiration: AgentCollabBench](https://arxiv.org/abs/2605.08647)

**Example:** solve `x² − 5x + 6 = 0`, with `x > 2`. Only A sees the original task; intermediate workers must pass relevant constraints onward.

| Agent | Expected output |
| --- | --- |
| A: factorize | `(x − 2)(x − 3) = 0; require x > 2` |
| B: find candidate roots | `Candidates: {2, 3}; require x > 2` |
| C: select the valid root | `3` |

If B sends only `{2, 3}`, the constraint is lost at B's handoff—even if C happens to guess `3`. Repeat with `x < 3`, where the answer should become `2`, to test whether the constraint influences the result.

**Measure:** constraint retention at required handoffs, final constraint compliance, and where information is lost. Extend to longer chains and merging branches.

**Main comparison:** accuracy and cost versus validated behavioral failures, counterfactual effects, and constraint retention.

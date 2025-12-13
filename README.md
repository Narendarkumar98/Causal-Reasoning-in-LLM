This repository contains the dataset, prompts, model outputs, and evaluation scripts for an experimental study analyzing causal reasoning performance of large language models (LLMs) in single-agent and multi-agent smart-building settings.
The study evaluates whether unstructured multi-agent collaboration and consensus improve causal reasoning compared to a strong single-agent baseline.
**Overview:**
We evaluate three reasoning configurations using Llama-3.1-8B-Instruct:
**1. Single-Agent**
   Standard inference using one model instance.
**2. Two-Agent Collaborative**
   Two independent agents answer the same question and are allowed one round of mutual revision after observing each other’s reasoning.
**3. Consensus Agent**
   A third agent aggregates the final outputs of the collaborative agents into a single response.
  All configurations use identical prompts and inference parameters to ensure fairness.
  **Evaluation Dataset:**
 >Total questions: 20 expert-designed causal questions

>Smart-building subsystems:
 i. HVAC (7)
ii. Lighting (6)
iii. Security (7)

**Pearl’s causal hierarchy:**
i. Observational (6)
ii. Interventional (6)
iii. Counterfactual (8)

Questions are derived from canonical smart-building sensing, control, and policy-intervention scenarios.
**Metrics:**
**We report:**
**Accuracy:** Exact match between model output and ground-truth label.

**Trust-impact metrics:**
Number of errors corrected by consensus
Number of previously correct answers degraded by consensus
**Key observations:**
The single-agent model consistently outperforms multi-agent setups.
Consensus reasoning corrects only a few errors (3–4 cases) but degrades many correct single-agent answers (12 cases).
Collaborative reasoning performs better on counterfactual questions, but fails on observational and interventional ones.
These results suggest that naive multi-agent interaction can reduce reliability in causal reasoning tasks.

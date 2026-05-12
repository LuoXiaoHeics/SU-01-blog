# SU-01: Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling

This repository contains the SU-01 project page, evaluation visualizations, olympiad case studies, and related materials.

![ProofBench Overall Results](docs/source_html/proofbench_light.png)

## Overview

Shanghai Artificial Intelligence Laboratory introduced SU-01, a unified post-training recipe for olympiad-level scientific reasoning. The work shows that olympiad-level capability does not necessarily require an extremely large backbone or discipline-specific systems. On a general-size `30B-A3B` model, appropriate behavior shaping, reward learning, and inference-time scaling can also produce verifiable long-horizon proof-search ability.

The recipe is instantiated as **SU-01**. It first applies reverse-perplexity curriculum SFT on roughly **338K** high-quality trajectories, then uses **200** steps of two-stage reinforcement learning to improve solving ability and complete-proof quality, and finally runs a multi-round "generate-verify-revise" loop at inference time. Without calling external tools, executing code, or relying on dedicated symbolic solvers, the model can sustain more than **100K tokens** of natural-language reasoning on difficult olympiad problems.

In competition-style evaluations, inference-time scaling brings **SU-01** to **35** points on both **IMO 2025** and **USAMO 2026**, reaching gold-medal-level performance. It also exceeds the gold cutoff on **IPhO 2024/2025** and substantially outperforms similarly sized models on proof-level benchmarks such as **ProofBench**. The central insight is that scientific reasoning depends not only on model scale, but also on whether proof construction, verification, and revision can be organized into a reusable training-inference loop.

## Links

- [SU-01 Official Repo](https://github.com/)
- [SU-01-30B-A3B](https://huggingface.co/)

## SU-01 Pipeline

![SU-01 Pipeline](docs/source_html/simplex-pipeline-hires.png)

### Instilling Rigorous Reasoning via Supervised Fine-tuning
We curate SFT prompts from a broad mixture of mathematical, scientific, instruction-following, and coding sources. Before generation, contaminated problems are removed from the prompt pool. For each remaining prompt, DeepSeek-V3.2-Speciale is used to generate high-quality long-form reasoning trajectories. We then filter noisy generations and remove trajectories longer than 8,192 tokens. This keeps the supervised signal focused on rigorous and usable reasoning traces while avoiding truncation and optimization instability from extremely long outputs.

### Boosting Reasoning Capability with Reinforcement Learning
- **RL data curation:** We split the RL pool into verifiable prompts, whose answers or structured outputs can be checked reliably, and non-verifiable prompts for proof-oriented or open-ended reasoning. After deduplication, decontamination, rejection sampling, and noise filtering, the pool contains 8,967 verifiable prompts and 16,287 non-verifiable prompts.
- **Coarse RL:** This stage uses reinforcement learning with verifiable rewards (RLVR) and Group Sequence Policy Optimization (GSPO), applying reward assignment and policy clipping at the complete-response level.
- **Refined RL:** After coarse RL builds search ability, refined RL shifts the target toward proof quality, using stronger process-level rewards, self-refinement, and experience replay so failures can be repaired and rare successful proofs remain learnable.

### Achieving Gold-Medal-Level Reasoning via Test-time Scaling

Even with a strong reasoning policy, the hardest IMO-style problems still require substantial search, verification, and revision. TTS is not just answer sampling. It uses a self-verification and refinement loop to repair hidden gaps and produce complete, rigorous proofs.

## Results

### Core Benchmark Results

**Table 1. Performance on answer-verifiable reasoning tasks.** AnswerBench, AMO-Bench, AIME 25/26, and FrontierScience-Olympiad are averaged over 4, 8, 8, and 4 runs, respectively. Avg. is the mean of AnswerBench, AMO-Bench, AIME 2025, AIME 2026, and FrontierScience-Olympiad.

| Model | AnswerBench | AMO-Bench | AIME 25/26 | FS-O Physics | FS-O Chemistry | FS-O Biology | FS-O Overall | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1-30B-A3B | 69.3% | 41.3% | 90.4%/89.6% | 57.5% | 57.5% | <u>27.5%</u> | 54.5% | 69.0% |
| GLM-4.7-Flash | 73.8% | 53.8% | 91.3%/88.3% | 54.5% | 60.0% | 17.5% | 53.0% | 72.0% |
| Nemotron-Cascade-2 | **80.5%** | 40.8% | <u>94.2%</u>/90.0% | 56.0% | 56.3% | **30.0%** | 53.5% | 71.8% |
| Qwen3.6-35B-A3B | <u>78.0%</u> | <u>58.8%</u> | 92.5%/<u>92.9%</u> | <u>65.5%</u> | **74.4%** | 25.0% | **65.0%** | **77.4%** |
| Gemma-4-31B | 67.7% | 34.5% | 88.8%/91.3% | **69.0%** | 61.9% | <u>27.5%</u> | 61.0% | 68.7% |
| **SU-01** | 77.5% | **59.8%** | **94.6%**/**93.3%** | 62.5% | <u>69.4%</u> | 25.0% | <u>61.5%</u> | <u>77.3%</u> |

Bold marks the best score within the comparison block; underline marks the second best. FS-O abbreviates FrontierScience-Olympiad.


### Non-Verifiable Benchmarks

**Table 2. Performance on non-verifiable benchmarks.** FrontierScience-Research refers to the research subset of FrontierScience. For SU-01, `x/y` reports scores without and with TTS on ProofBench.

| Group | Model | ProofBench Basic | ProofBench Advanced | ProofBench Overall | FS-R Physics | FS-R Chemistry | FS-R Biology | FS-R Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Larger models | Gemini 3.1 Pro Thinking | <u>95.2%</u> | <u>50.0%</u> | <u>72.6%</u> | 0.0% | <u>30.0%</u> | 10.0% | 13.3% |
| Larger models | GPT-5.5-High | **96.7%** | **64.8%** | **80.7%** | **25.0%** | **40.0%** | **45.0%** | **36.7%** |
| Larger models | DeepSeek-V3.2-Speciale | 62.9% | 28.6% | 45.7% | <u>10.0%</u> | 20.0% | <u>15.0%</u> | <u>15.0%</u> |
| Similar-size models | P1-30B-A3B | 33.8% | 6.2% | 20.0% | 0.0% | **10.0%** | 0.0% | 3.3% |
| Similar-size models | GLM-4.7-Flash | 51.0% | 16.7% | 33.8% | 0.0% | 0.0% | 0.0% | 0.0% |
| Similar-size models | Nemotron-Cascade-2 | <u>77.1%</u> | 28.6% | 52.9% | <u>5.0%</u> | 5.0% | **20.0%** | <u>10.0%</u> |
| Similar-size models | Qwen3.6-35B-A3B | 39.1% | 7.1% | 23.1% | 0.0% | 5.0% | 10.0% | 5.0% |
| Similar-size models | Gemma-4-31B | 46.7% | 16.2% | 31.4% | 0.0% | **10.0%** | 5.0% | 5.0% |
| Similar-size models | **SU-01** | <u>77.1%</u> / **91.0%** | <u>38.1%</u> / **49.5%** | <u>57.6%</u> / **70.2%** | **10.0%** | **10.0%** | <u>15.0%</u> | **11.7%** |

Bold and underline indicate the best and second-best results within each comparison block. FS-R abbreviates FrontierScience-Research.


### Olympiad Competition Performance

**Table 3a. IPhO 2024/2025 performance.** For SU-01, `x/y` reports scores without and with TTS.

| Model | IPhO 2024 | IPhO 2025 |
| --- | ---: | ---: |
| P1-30B-A3B | 23.1 | 17.7 |
| GLM-4.7-Flash | 22.2 | 19.5 |
| Nemotron-Cascade-2 | 21.2 | 16.7 |
| Qwen3.6-35B-A3B | 24.3 | 19.9 |
| Gemma-4-31B | <u>24.4</u> | <u>20.3</u> |
| **SU-01** | 23.5 / **25.3** | <u>20.3</u> / **21.7** |

**Table 3b. IMO 2025 performance.** Stars indicate human-expert evaluation for TTS results.

| Model | P1 | P2 | P3 | P4 | P5 | P6 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SU-01 | 1 | 7 | 1 | 6 | 6 | 0 | 21 |
| **SU-01 w/ TTS** | 7<sup>*</sup> | 7<sup>*</sup> | 7<sup>*</sup> | 7<sup>*</sup> | 7<sup>*</sup> | 0<sup>*</sup> | **35<sup>*</sup>** |

**Table 3c. USAMO 2026 performance.** Stars indicate human-expert evaluation for TTS results.

| Model | P1 | P2 | P3 | P4 | P5 | P6 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SU-01 | 7 | 0 | 0 | 7 | 0 | 1 | 15 |
| **SU-01 w/ TTS** | 7<sup>*</sup> | 0<sup>*</sup> | 7<sup>*</sup> | 7<sup>*</sup> | 7<sup>*</sup> | 7<sup>*</sup> | **35<sup>*</sup>** |

Gold lines for IPhO 2024/2025 are 20.8/19.7 points. Medal lines for IMO 2025 are 35/28/19 points, and medal lines for USAMO 2026 are 25/18/11 points. TTS denotes test-time scaling.

### Test-time Scaling Mechanics

Test-time scaling further amplifies the model's proof-search and self-correction capabilities. Through iterative cycles of "generate candidate solution - verify complete proof - locate issues - refine solution", the model can transform incomplete or unstable attempts into complete, scorable solutions. What is expanded is not an external tool pipeline, but the model's own natural-language verification and refinement computation.

Analysis of test-time scaling traces on USAMO 2026 shows that initial solution generation has a median length of approximately **106K tokens**, while refinement stages have a median length of approximately **83K tokens**. The model can sustain effective reasoning at the 100K-token scale even at the `30B-A3B` parameter weight, using long-form computation for proof construction, gap identification, and argument repair.

![Test-time Scaling Action Length Distribution](docs/source_png/tts_action_length_distribution_1.png)

## Case Study

Across the twelve IMO 2025 and USAMO 2026 problems, the model gives full-credit solutions to ten problems, with failures on IMO 2025 P6 and USAMO 2026 P2. Its main strength is translating olympiad problems into formal frameworks: coordinates or complex numbers for geometry, modular classifications for number theory, recurrences for functional equations, and automata-based dynamic programming for digit problems.

A particularly striking example is USAMO 2026 P3. Rather than following the standard synthetic geometry route, the model elegantly uses complex numbers to unify the unit circle, equilateral-triangle rotations, chord relations, and tangent conditions within a single algebraic framework. This yields an analytic reformulation of a configuration that olympiad solvers would typically approach through angle chasing and carefully chosen auxiliary constructions.

IMO 2025 P2 shows a complementary strength, where the model reduces a two-circle tangency problem to coordinate and distance computations. Other strong examples include the carry-state dynamic programming approach for USAMO P4 and the number-theoretic proof using totients, congruences, Vieta jumping, and Fibonacci structure in USAMO P6.

## Acknowledgements

This work was supported by the Shanghai Artificial Intelligence Laboratory. We thank the authors and maintainers of prior open research and infrastructure that made this work possible. In particular, we are grateful to DeepSeek for open-sourcing strong reasoning policies and generative reward models, which provided an important reference point for our work. IMO-Bench, AMO-Bench, and FrontierScience helped guide the overall system optimization by offering challenging mathematical and scientific reasoning benchmarks and evaluation protocols.

We also thank prior data efforts that supported our SFT and RL data curation, including DeepMath, NaturalReasoning, Eurus, OpenCodeReasoning, P1, and OPC, as well as the many public problem sources and communities that cannot all be listed here. We further acknowledge the broader open-source infrastructure ecosystem, including slime for training and SGLang for efficient inference and serving.

## Citation

```bibtex
@misc{su012026,
    title={Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling},
    author={SU-01 Team},
    year={2026},
    url={https://.github.io/SU-01/}
}
```

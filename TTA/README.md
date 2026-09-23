# OVAL

**Online Value Adaptation from Lookahead for Test-Time Alignment**

## Overview

OVAL turns lookahead rollouts into supervision for online value adaptation. At each decoding step, it samples candidate segments, evaluates their continuations with a frozen critic, and uses the resulting preferences to update a lightweight residual. The adapted critic selects the next segment, while accumulated evidence is retained across decoding steps.

The language model and offline critic remain frozen. The residual is initialized independently for each prompt.

## Method

1. **Candidate generation.** Sample candidate segments conditioned on the prompt and current response prefix.
2. **Lookahead evaluation.** Generate bounded continuations and back up their frozen-critic scores to the initiating candidates.
3. **Preference supervision.** Construct candidate pairs from the backed-up scores and compute a pairwise logistic gradient.
4. **Online adaptation.** Accumulate gradients and squared gradients, then compute a regularized FTRL residual.
5. **Action selection.** Select a segment using the adapted critic and advance the response prefix.

For candidate features $\phi(s,a)$, the adapted score is

$$
Q_t(s,a) = \phi(s,a)^\top (w_0 + \delta_t),
$$

where $w_0$ is the frozen critic and $\delta_t$ is the prompt-specific residual.

## Code structure

```text
src/
  oval/
    critic.py
    method.py
  tpo/
    method.py
    prompts.py
  sft/
    method.py
    training.py
  grpo/
    method.py
    training.py
  models.py
  plain_generation.py
  rewards.py
```


| Component                         | Role                                                            |
| --------------------------------- | --------------------------------------------------------------- |
| `generate_oval`                   | Receding-horizon decoding with online value adaptation          |
| `generate_oval_lookahead_records` | Bounded lookahead generation for candidate segments             |
| `oval_backed_up_targets`          | Frozen-critic evaluation and top-rollout aggregation            |
| `oval_pairwise_gradient`          | Pairwise preference gradient                                    |
| `oval_proximal_delta`             | Regularized residual update from accumulated gradients          |
| `CandidateValueHead`              | Frozen hidden-state features and linear value scoring           |
| `src/tpo/`                        | Response sampling, contrastive feedback, and iterative revision |
| `src/sft/`                        | Supervised adapter training and response generation             |
| `src/grpo/`                       | Group-relative policy training with a reward callback           |
| `src/models.py`                   | Policy loading, generation, and prompt formatting               |
| `src/rewards.py`                  | Batched reward-model scoring                                    |

## Interface

The main entry point accepts a language model, tokenizer, critic, and decoding settings:

```python
from src.oval import generate_oval

result = generate_oval(
    cfg=cfg,
    tokenizer=tokenizer,
    model=model,
    judge=critic,
    device=device,
    prompt={"prompt": text},
)
response = result["response"]
```

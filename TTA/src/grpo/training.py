from __future__ import annotations

from typing import Any, Callable

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


def make_reward_function(reward: Any) -> Callable[..., list[float]]:
    def reward_function(
        prompts: list[Any],
        completions: list[Any],
        raw_prompt: list[str],
        **kwargs: Any,
    ) -> list[float]:
        responses = [
            item if isinstance(item, str) else item[-1]["content"]
            for item in completions
        ]
        return reward.score_batch(list(zip(raw_prompt, responses, strict=True)))

    return reward_function


def build_grpo_trainer(
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    reward: Any,
    initialize_adapter: bool = True,
) -> GRPOTrainer:
    tokenizer.padding_side = "left"
    training_args = GRPOConfig(**cfg["grpo"])
    training_args.remove_unused_columns = False
    adapter_config = (
        LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]) if initialize_adapter else None
    )
    return GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=make_reward_function(reward),
        peft_config=adapter_config,
    )


def train_grpo(
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    reward: Any,
    initialize_adapter: bool = True,
) -> GRPOTrainer:
    trainer = build_grpo_trainer(
        cfg, model, tokenizer, train_dataset, reward, initialize_adapter
    )
    trainer.train()
    trainer.save_model(trainer.args.output_dir)
    return trainer

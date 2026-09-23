from __future__ import annotations

from typing import Any

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def build_sft_trainer(
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
) -> SFTTrainer:
    tokenizer.padding_side = "right"
    training_args = SFTConfig(**cfg["sft"])
    adapter_config = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=adapter_config,
    )


def train_sft(
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
) -> SFTTrainer:
    trainer = build_sft_trainer(cfg, model, tokenizer, train_dataset)
    trainer.train()
    trainer.save_model(trainer.args.output_dir)
    return trainer

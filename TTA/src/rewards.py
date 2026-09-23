from __future__ import annotations

from typing import Any

import torch

from src.models import render_chat


class RewardScorer:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        attribute_index: int,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.attribute_index = attribute_index
        self.batch_size = batch_size
        self.max_length = max_length

    def score(self, prompt: str, responses: list[str]) -> list[float]:
        return self.score_batch([(prompt, response) for response in responses])

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            texts = [
                render_chat(
                    self.tokenizer,
                    "",
                    prompt,
                    response=response,
                    add_generation_prompt=False,
                )
                for prompt, response in batch
            ]
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                rewards = self.model(**inputs).rewards[:, self.attribute_index]
            scores.extend(rewards.detach().float().cpu().tolist())
        return scores

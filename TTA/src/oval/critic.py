from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.models import render_chat_with_assistant_prefix


def pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    last_tokens: int,
) -> torch.Tensor:
    mask = attention_mask.to(dtype=torch.bool)
    reverse_rank = torch.flip(
        torch.cumsum(torch.flip(mask.to(dtype=torch.int64), dims=[1]), dim=1),
        dims=[1],
    )
    selected = mask & (reverse_rank <= last_tokens)
    counts = selected.sum(dim=1).clamp_min(1).unsqueeze(1)
    return (hidden * selected.unsqueeze(2)).sum(dim=1) / counts


class CandidateValueHead:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        weight: np.ndarray,
        device: torch.device,
        layer: int,
        pooling_tokens: int,
        batch_size: int,
        max_input_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.weight = np.asarray(weight, dtype=np.float32)
        self.device = device
        self.layer = layer
        self.pooling_tokens = pooling_tokens
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens

    def feature_matrix(
        self, prompt: str, prefix: str, candidates: list[str]
    ) -> np.ndarray:
        texts = [
            render_chat_with_assistant_prefix(
                self.tokenizer,
                "",
                prompt,
                "\n\n".join(
                    part.strip() for part in (prefix, candidate) if part.strip()
                ),
            )
            for candidate in candidates
        ]
        features = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                texts[start : start + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=False,
            )
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model(
                    **inputs, output_hidden_states=True, use_cache=False
                )
                hidden = output.hidden_states[self.layer + 1]
                pooled = pool_hidden(
                    hidden, inputs["attention_mask"], self.pooling_tokens
                )
            features.append(pooled.detach().float().cpu().numpy())
        return np.concatenate(features, axis=0)

    def score(self, prompt: str, prefix: str, candidates: list[str]) -> list[float]:
        values = self.feature_matrix(prompt, prefix, candidates) @ self.weight
        return values.tolist()

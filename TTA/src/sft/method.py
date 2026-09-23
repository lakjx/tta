from __future__ import annotations

from typing import Any

import torch

from src.plain_generation import generate_response


def generate_sft(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    return generate_response(cfg, tokenizer, model, device, prompt, "sft")

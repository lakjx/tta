from __future__ import annotations

from typing import Any

import torch

from src.models import generate_texts, render_chat, token_count


def generate_response(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    tokenizer.padding_side = "left"
    model.eval()
    text = render_chat(
        tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
    )
    output = generate_texts(cfg, tokenizer, model, device, [text], 1)[0]
    return {
        "method": method,
        "response": output["text"],
        "response_token_count": token_count(tokenizer, output["text"]),
    }

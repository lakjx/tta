from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from src.models import generate_texts, render_chat, token_count

from .prompts import (
    TEXTGRAD_BACKWARD_SYSTEM_PROMPT,
    TEXTGRAD_OPTIMIZER_SYSTEM_PROMPT,
    build_tpo_backward_prompt,
    build_tpo_loss_system_prompt,
    build_tpo_update_prompt,
)


@dataclass(frozen=True)
class Candidate:
    response: str
    score: float


def generate_tpo(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    reward: Any,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    settings = cfg["decoding"]["tpo"]
    query = prompt["prompt"]
    iterations = int(settings.get("iterations", 2))
    sample_size = int(settings.get("samples", 5))
    response_tokens = int(
        settings.get("response_max_tokens", cfg["generation"]["max_new_tokens"])
    )
    feedback_tokens = int(settings.get("feedback_max_tokens", 512))

    initial = sample_tpo_outputs(
        cfg, tokenizer, model, device, "", query, sample_size, response_tokens
    )
    initial_scores = reward.score(query, initial)
    candidates = [
        Candidate(response, float(score))
        for response, score in zip(initial, initial_scores, strict=True)
    ]

    for _ in range(iterations):
        chosen = max(candidates, key=lambda candidate: candidate.score)
        rejected = min(candidates, key=lambda candidate: candidate.score)
        loss_system = build_tpo_loss_system_prompt(query, rejected.response)
        textual_loss = sample_tpo_outputs(
            cfg,
            tokenizer,
            model,
            device,
            loss_system,
            chosen.response,
            1,
            feedback_tokens,
        )[0]
        backward_prompt = build_tpo_backward_prompt(
            loss_system, chosen.response, textual_loss
        )
        textual_gradient = sample_tpo_outputs(
            cfg,
            tokenizer,
            model,
            device,
            TEXTGRAD_BACKWARD_SYSTEM_PROMPT,
            backward_prompt,
            1,
            feedback_tokens,
        )[0]
        update_prompt = build_tpo_update_prompt(
            loss_system, chosen.response, textual_loss, textual_gradient
        )
        updates = sample_tpo_outputs(
            cfg,
            tokenizer,
            model,
            device,
            TEXTGRAD_OPTIMIZER_SYSTEM_PROMPT,
            update_prompt,
            sample_size,
            response_tokens,
        )
        improved = [
            response
            for output in updates
            if (response := extract_improved_variable(output)) is not None
        ]
        if improved:
            scores = reward.score(query, improved)
            candidates.extend(
                Candidate(response, float(score))
                for response, score in zip(improved, scores, strict=True)
            )

    selected = max(candidates, key=lambda candidate: candidate.score)
    return {
        "method": "tpo",
        "response": selected.response,
        "response_token_count": token_count(tokenizer, selected.response),
        "selection_reward": selected.score,
    }


def sample_tpo_outputs(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    system_prompt: str,
    user_prompt: str,
    count: int,
    max_new_tokens: int,
) -> list[str]:
    settings = cfg["decoding"]["tpo"]
    chat = render_chat(
        tokenizer,
        system_prompt,
        user_prompt,
        response=None,
        add_generation_prompt=True,
    )
    outputs = generate_texts(
        cfg,
        tokenizer,
        model,
        device,
        [chat],
        count,
        max_new_tokens=max_new_tokens,
        max_input_tokens=int(settings.get("max_input_tokens", 7168)),
        temperature=float(settings.get("temperature", 0.7)),
        top_p=float(settings.get("top_p", 0.95)),
        top_k=int(settings.get("top_k", 0)),
    )
    return [output["text"] for output in outputs]


def extract_improved_variable(text: str) -> str | None:
    match = re.search(
        r"<IMPROVED_VARIABLE>\s*(.*?)\s*</IMPROVED_VARIABLE>", text, flags=re.DOTALL
    )
    if match is None:
        return None
    return match.group(1).strip() or None

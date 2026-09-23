from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.models import (
    append_segment,
    generate_texts,
    render_chat_with_assistant_prefix,
    token_count,
)


def generate_oval(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    judge: Any,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    search = cfg["search"]
    feature_matrix = judge.feature_matrix
    w0 = np.asarray(judge.weight, dtype=np.float32)
    grad_sum = np.zeros_like(w0)
    grad_sq_sum = np.zeros_like(w0)
    delta = np.zeros_like(w0)
    prefix = ""
    final_max = int(cfg["generation"]["max_new_tokens"])
    segment_max = int(search["mpc_segment_max_tokens"])
    lookahead_count = int(search.get("oval_lookahead_per_action", 2))
    lookahead_max = int(search.get("oval_lookahead_max_tokens", segment_max))
    alpha = float(search.get("oval_alpha", 0.1))

    for _ in range(int(search["mpc_steps"])):
        remaining = final_max - token_count(tokenizer, prefix)
        if remaining <= 0:
            break
        prompt_text = render_chat_with_assistant_prefix(
            tokenizer, "", prompt["prompt"], prefix
        )
        outputs = generate_texts(
            cfg,
            tokenizer,
            model,
            device,
            [prompt_text],
            int(search["mpc_candidates_per_step"]),
            max_new_tokens=min(segment_max, remaining),
        )
        candidates = [
            {"segment": item["text"], "raw": item}
            for item in outputs
            if item["text"].strip()
        ]
        if not candidates:
            break
        actions = [item["segment"] for item in candidates]
        features = feature_matrix(prompt["prompt"], prefix, actions).astype(np.float32)
        records = generate_oval_lookahead_records(
            cfg,
            tokenizer,
            model,
            device,
            prompt["prompt"],
            prefix,
            candidates,
            lookahead_count,
            lookahead_max,
            final_max,
        )
        targets = oval_backed_up_targets(
            feature_matrix,
            prompt["prompt"],
            prefix,
            len(actions),
            records,
            w0,
            target_top_k=int(search.get("oval_target_top_k", 1)),
        )
        gradient, pair_count = oval_pairwise_gradient(
            features,
            targets,
            w0 + delta,
            margin=float(search.get("oval_margin", 0.03)),
            max_pairs=int(search.get("oval_max_pairs", 128)),
        )
        if alpha > 0.0 and pair_count:
            grad_sum += gradient
            grad_sq_sum += gradient * gradient
            delta = oval_proximal_delta(
                grad_sum,
                grad_sq_sum,
                alpha=alpha,
                beta=float(search.get("oval_beta", 1.0)),
                l1=float(search.get("oval_l1", 0.0)),
                l2=float(search.get("oval_l2", 0.5)),
                max_delta_norm=float(search.get("oval_max_delta_norm", 1.0)),
            )
        selected = candidates[int(np.argmax(features @ (w0 + delta)))]
        prefix = append_segment(prefix, selected["segment"])
        if selected["raw"]["stop_reason"] == "stop_token":
            break

    return {
        "method": "oval",
        "response": prefix,
        "response_token_count": token_count(tokenizer, prefix),
    }


def generate_oval_lookahead_records(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: str,
    prefix: str,
    candidates: list[dict[str, Any]],
    lookahead_per_action: int,
    lookahead_max_tokens: int,
    final_max_tokens: int,
) -> list[dict[str, Any]]:
    records = []
    jobs = []
    for action_index, candidate in enumerate(candidates):
        segment = candidate["segment"]
        prefix_after = append_segment(prefix, segment)
        remaining = final_max_tokens - token_count(tokenizer, prefix_after)
        if (
            lookahead_per_action <= 0
            or remaining <= 0
            or candidate["raw"]["stop_reason"] == "stop_token"
        ):
            records.append({"action_index": action_index, "target_candidate": segment})
            continue
        prompt_text = render_chat_with_assistant_prefix(
            tokenizer, "", prompt, prefix_after
        )
        for _ in range(lookahead_per_action):
            jobs.append(
                (
                    min(lookahead_max_tokens, remaining),
                    prompt_text,
                    action_index,
                    segment,
                )
            )

    for budget in sorted({job[0] for job in jobs}, reverse=True):
        group = [job for job in jobs if job[0] == budget]
        outputs = generate_texts(
            cfg,
            tokenizer,
            model,
            device,
            [job[1] for job in group],
            1,
            max_new_tokens=budget,
        )
        for (_, _, action_index, segment), item in zip(group, outputs, strict=True):
            records.append(
                {
                    "action_index": action_index,
                    "target_candidate": append_segment(segment, item["text"]),
                }
            )
    return records


def oval_backed_up_targets(
    feature_matrix: Any,
    prompt: str,
    prefix: str,
    action_count: int,
    lookahead_records: list[dict[str, Any]],
    target_weight: np.ndarray,
    target_top_k: int,
) -> list[float]:
    grouped: list[list[float]] = [[] for _ in range(action_count)]
    if not lookahead_records:
        return [float("-inf") for _ in range(action_count)]
    features = feature_matrix(
        prompt, prefix, [record["target_candidate"] for record in lookahead_records]
    ).astype(np.float32)
    scores = features @ target_weight
    for record, score in zip(lookahead_records, scores, strict=True):
        action_index = int(record["action_index"])
        if 0 <= action_index < len(grouped):
            grouped[action_index].append(float(score))
    return [top_mean(values, target_top_k) for values in grouped]


def top_mean(values: list[float], top_k: int) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return float("-inf")
    ordered = sorted(finite, reverse=True)
    top = ordered[: max(1, min(int(top_k), len(ordered)))]
    return float(sum(top) / len(top))


def oval_pairwise_gradient(
    features: np.ndarray,
    targets: list[float],
    weight: np.ndarray,
    margin: float,
    max_pairs: int,
) -> tuple[np.ndarray, int]:
    pairs: list[tuple[float, int, int]] = []
    for left_idx, left_target in enumerate(targets):
        if not np.isfinite(float(left_target)):
            continue
        for right_idx, right_target in enumerate(targets):
            if left_idx == right_idx or not np.isfinite(float(right_target)):
                continue
            gap = float(left_target) - float(right_target)
            if gap > margin:
                pairs.append((gap, left_idx, right_idx))
    if not pairs:
        return np.zeros(features.shape[1], dtype=np.float32), 0
    pairs.sort(reverse=True)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    grad = np.zeros(features.shape[1], dtype=np.float32)
    for _, winner_idx, loser_idx in pairs:
        diff = features[winner_idx] - features[loser_idx]
        logit = float(np.clip(weight @ diff, -40.0, 40.0))
        coeff = -1.0 / (1.0 + float(np.exp(logit)))
        grad += coeff * diff
    grad /= float(len(pairs))
    return grad.astype(np.float32), len(pairs)


def oval_proximal_delta(
    grad_sum: np.ndarray,
    grad_sq_sum: np.ndarray,
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    max_delta_norm: float,
) -> np.ndarray:
    denom = (float(beta) + np.sqrt(np.maximum(grad_sq_sum, 0.0))) / max(
        float(alpha), 1e-8
    ) + float(l2)
    if l1 > 0.0:
        adjusted = np.where(
            np.abs(grad_sum) > l1, grad_sum - np.sign(grad_sum) * l1, 0.0
        )
        delta = -adjusted / denom
    else:
        delta = -grad_sum / denom
    return clip_vector_norm(delta.astype(np.float32), max_delta_norm)


def clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return vector
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= max_norm:
        return vector
    return (vector * (float(max_norm) / max(norm, 1e-8))).astype(np.float32)

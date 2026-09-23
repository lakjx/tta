from __future__ import annotations

from typing import Any

import torch


def load_policy(
    model_path: str,
    device: torch.device,
    adapter_path: str | None = None,
    trainable: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    model.to(device)
    model.train(trainable)
    return tokenizer, model


def render_chat(
    tokenizer: Any,
    system_prompt: str,
    prompt: str,
    response: str | None,
    add_generation_prompt: bool,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def render_chat_with_assistant_prefix(
    tokenizer: Any,
    system_prompt: str,
    prompt: str,
    response_prefix: str,
) -> str:
    return (
        render_chat(
            tokenizer,
            system_prompt,
            prompt,
            response=None,
            add_generation_prompt=True,
        )
        + response_prefix
    )


def generate_texts(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompts: list[str],
    num_return_sequences: int,
    max_new_tokens: int | None = None,
    *,
    max_input_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    batch_size = int(cfg.get("runtime", {}).get("generation_batch_size", 0))
    if batch_size > 0 and len(prompts) > batch_size:
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(prompts), batch_size):
            outputs.extend(
                generate_texts(
                    cfg,
                    tokenizer,
                    model,
                    device,
                    prompts[start : start + batch_size],
                    num_return_sequences,
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            )
        return outputs
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_input_tokens or cfg["generation"]["max_input_tokens"]),
        add_special_tokens=False,
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    input_width = inputs["input_ids"].shape[1]
    stop_ids = generation_stop_token_ids(tokenizer)
    new_tokens = int(
        max_new_tokens
        if max_new_tokens is not None
        else cfg["generation"]["max_new_tokens"]
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            do_sample=True,
            temperature=float(
                temperature
                if temperature is not None
                else cfg["generation"]["temperature"]
            ),
            top_p=float(top_p if top_p is not None else cfg["generation"]["top_p"]),
            top_k=int(top_k if top_k is not None else cfg["generation"]["top_k"]),
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )
    return [
        decode_generated_tokens(cfg, tokenizer, row[input_width:], stop_ids, new_tokens)
        for row in out
    ]


def generation_stop_token_ids(tokenizer: Any) -> list[int]:
    return [tokenizer.eos_token_id]


def decode_generated_tokens(
    cfg: dict[str, Any],
    tokenizer: Any,
    gen_ids: torch.Tensor,
    stop_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    ids = [int(token_id) for token_id in gen_ids.detach().cpu().tolist()]
    stop_set = set(stop_ids)
    stop_pos = next(
        (idx for idx, token_id in enumerate(ids) if token_id in stop_set), None
    )
    effective = ids if stop_pos is None else ids[:stop_pos]
    stop_reason = (
        "stop_token"
        if stop_pos is not None
        else ("max_new_tokens" if len(ids) >= max_new_tokens else "unknown")
    )
    text = tokenizer.decode(effective, skip_special_tokens=True).strip()
    hit_max = stop_pos is None and len(ids) >= max_new_tokens
    return {
        "text": text,
        "token_count": len(effective),
        "raw_token_count": len(ids) if stop_pos is None else stop_pos,
        "hit_max_new_tokens": hit_max,
        "stop_reason": stop_reason,
    }


def append_segment(prefix: str, segment: str) -> str:
    return " ".join(part.strip() for part in (prefix, segment) if part.strip())


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0

"""Lazy-loaded Hugging Face policy runtime shared by pilot and full experiments.

Importing this module does not require torch/transformers/peft. Those packages
are required only when ``HFPolicy`` is constructed.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

from protocol_core import EpisodeLimits
from safety_harness import GeneratedTurn


@dataclass(frozen=True)
class TokenSegment:
    context_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    old_log_probs: tuple[float, ...]


def _dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "The ML runtime is not installed. Install the pinned requirements before running model jobs."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, (LoraConfig, PeftModel), get_peft_model


def token_log_probs(model: Any, context_ids: Iterable[int], output_ids: Iterable[int], *, grad: bool) -> Any:
    torch, *_ = _dependencies()
    context = list(context_ids)
    output = list(output_ids)
    if not context or not output:
        raise ValueError("Context and generated output must both contain tokens")
    device = next(model.parameters()).device
    input_ids = torch.tensor([context + output], dtype=torch.long, device=device)
    manager = nullcontext() if grad else torch.no_grad()
    with manager:
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
        start = len(context) - 1
        selected = logits[:, start : start + len(output), :]
        targets = torch.tensor([output], dtype=torch.long, device=device)
        return torch.log_softmax(selected, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(0).squeeze(-1)


class HFPolicy:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        adapter: dict[str, Any] | None,
        adapter_checkpoint: Path | None = None,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer, peft_classes, get_peft_model = _dependencies()
        LoraConfig, PeftModel = peft_classes
        self.torch = torch
        self.device = device
        self.seed = seed
        self._draw_index = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
        base.to(device)
        base.config.use_cache = False
        if adapter_checkpoint is not None:
            self.model = PeftModel.from_pretrained(base, str(adapter_checkpoint), is_trainable=True)
        elif adapter is not None:
            config = LoraConfig(
                r=int(adapter["rank"]),
                lora_alpha=int(adapter["alpha"]),
                lora_dropout=float(adapter["dropout"]),
                target_modules=list(adapter["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base, config)
        else:
            self.model = base
        self.model.to(device)

    def _context_ids(self, messages: list[dict[str, str]]) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        return list(encoded)

    def _sample(self, messages: list[dict[str, str]], max_new_tokens: int, *, greedy: bool) -> GeneratedTurn:
        torch = self.torch
        context_ids = self._context_ids(messages)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        draw_seed = self.seed + self._draw_index
        self._draw_index += 1
        torch.manual_seed(draw_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(draw_seed)
        self.model.eval()
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0})
        with torch.no_grad():
            generated = self.model.generate(**kwargs)
        output_ids = generated[0, input_ids.shape[1] :].tolist()
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        old = token_log_probs(self.model, context_ids, output_ids, grad=False).detach().cpu().tolist()
        payload = TokenSegment(tuple(context_ids), tuple(output_ids), tuple(float(value) for value in old))
        return GeneratedTurn(text, len(output_ids), payload)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> GeneratedTurn:
        return self._sample(messages, max_new_tokens, greedy=False)

    def complete(self, prompt: str, max_new_tokens: int = 128, *, greedy: bool = False) -> GeneratedTurn:
        messages = [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": prompt},
        ]
        return self._sample(messages, max_new_tokens, greedy=greedy)

    def make_segment(self, messages: list[dict[str, str]], assistant_text: str) -> TokenSegment:
        context_ids = self._context_ids(messages)
        full_ids = list(
            self.tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": assistant_text}],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
            )
        )
        if full_ids[: len(context_ids)] != context_ids:
            raise RuntimeError("Native chat template does not preserve the assistant generation prefix")
        output_ids = full_ids[len(context_ids) :]
        old = token_log_probs(self.model, context_ids, output_ids, grad=False).detach().cpu().tolist()
        return TokenSegment(tuple(context_ids), tuple(output_ids), tuple(float(value) for value in old))

    def reference_context(self) -> Any:
        disable = getattr(self.model, "disable_adapter", None)
        return disable() if disable is not None else nullcontext()

    @contextmanager
    def merged_evaluation(self) -> Any:
        """Evaluate a disposable merged copy without mutating the active policy.

        Reversible PEFT merge/unmerge is numerically lossy for BF16 weights.  The
        trainable model is therefore parked on CPU while a deep-copied adapter
        model is merged once, evaluated, and discarded.
        """
        if not hasattr(self.model, "merge_and_unload"):
            yield
            return
        original = self.model
        original_device = next(original.parameters()).device
        disposable = None
        original.to("cpu")
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        try:
            disposable = deepcopy(original).to(original_device)
            disposable = disposable.merge_and_unload(safe_merge=True)
            disposable.config.use_cache = False
            disposable.eval()
            self.model = disposable
            yield
        finally:
            self.model = original
            if disposable is not None:
                disposable.to("cpu")
                del disposable
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            original.to(original_device)

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path), safe_serialization=True)
        self.tokenizer.save_pretrained(str(path))

    def merge_and_attach_fresh(
        self, adapter: dict[str, Any], merged_base_path: Path | None = None
    ) -> None:
        _, _, _, peft_classes, get_peft_model = _dependencies()
        LoraConfig, _ = peft_classes
        if not hasattr(self.model, "merge_and_unload"):
            raise RuntimeError("Recovery initialization requires a mergeable PEFT adapter")
        merged = self.model.merge_and_unload(safe_merge=True)
        if merged_base_path is not None:
            merged_base_path.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(merged_base_path), safe_serialization=True)
            self.tokenizer.save_pretrained(str(merged_base_path))
            merged.config._name_or_path = str(merged_base_path.resolve())
        self.torch.manual_seed(self.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.seed)
        config = LoraConfig(
            r=int(adapter["rank"]),
            lora_alpha=int(adapter["alpha"]),
            lora_dropout=float(adapter["dropout"]),
            target_modules=list(adapter["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(merged, config).to(self.device)
        if merged_base_path is not None:
            self.model.peft_config["default"].base_model_name_or_path = str(
                merged_base_path.resolve()
            )

    def reset_sampling(self, seed: int) -> None:
        self.seed = seed
        self._draw_index = 0

    def sampling_state(self) -> tuple[int, int]:
        return self.seed, self._draw_index

    def restore_sampling_state(self, state: tuple[int, int]) -> None:
        self.seed, self._draw_index = state

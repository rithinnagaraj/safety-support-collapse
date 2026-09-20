"""Exact Section 3 SFT and group-relative policy updates."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Iterable, Sequence

from hf_runtime import HFPolicy, TokenSegment, token_log_probs


@dataclass(frozen=True)
class TrainTrajectory:
    segments: tuple[TokenSegment, ...]
    reward: float
    record: dict[str, Any]


@dataclass(frozen=True)
class UpdateSummary:
    loss: float
    mean_reward: float
    all_zero_groups: int
    mixed_groups: int
    all_one_groups: int
    trajectories: int
    generated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        result = self.__dict__.copy()
        groups = self.all_zero_groups + self.mixed_groups + self.all_one_groups
        result.update(
            {
                "all_zero_group_fraction": self.all_zero_groups / groups,
                "mixed_group_fraction": self.mixed_groups / groups,
                "all_one_group_fraction": self.all_one_groups / groups,
            }
        )
        return result


def make_optimizer(policy: HFPolicy, settings: dict[str, Any]) -> Any:
    torch = policy.torch
    parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters; attach a trainable adapter first")
    return torch.optim.AdamW(
        parameters,
        lr=float(settings["learning_rate"]),
        betas=tuple(float(value) for value in settings["betas"]),
        eps=float(settings["epsilon"]),
        weight_decay=float(settings["weight_decay"]),
    )


def _trajectory_loss(
    policy: HFPolicy,
    trajectory: TrainTrajectory,
    advantage: float,
    ratio_clip: float,
    kl_coefficient: float,
) -> Any:
    torch = policy.torch
    token_losses: list[Any] = []
    for segment in trajectory.segments:
        current = token_log_probs(policy.model, segment.context_ids, segment.output_ids, grad=True)
        old = torch.tensor(segment.old_log_probs, dtype=torch.float32, device=current.device)
        ratio = torch.exp(current - old)
        raw = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip) * advantage
        policy_loss = -torch.minimum(raw, clipped)
        if kl_coefficient:
            with policy.reference_context():
                reference = token_log_probs(policy.model, segment.context_ids, segment.output_ids, grad=False)
            d = reference - current
            kl = torch.exp(d) - d - 1.0
            policy_loss = policy_loss + kl_coefficient * kl
        token_losses.append(policy_loss)
    if not token_losses:
        raise ValueError("A training trajectory contains no generated assistant tokens")
    return torch.cat(token_losses).mean()


def group_relative_update(
    policy: HFPolicy,
    optimizer: Any,
    groups: Sequence[Sequence[TrainTrajectory]],
    *,
    ratio_clip: float,
    kl_coefficient: float,
    gradient_norm_clip: float,
) -> UpdateSummary:
    torch = policy.torch
    trajectories = [trajectory for group in groups for trajectory in group]
    if not trajectories:
        raise ValueError("No trajectories supplied")
    optimizer.zero_grad(set_to_none=True)
    policy.model.train()
    total_loss = 0.0
    all_zero = mixed = all_one = 0
    trajectory_count = len(trajectories)
    for group in groups:
        rewards = [float(trajectory.reward) for trajectory in group]
        group_mean = sum(rewards) / len(rewards)
        variance = sum((reward - group_mean) ** 2 for reward in rewards) / len(rewards)
        denominator = math.sqrt(variance) + 1e-6
        if all(reward == 0.0 for reward in rewards):
            all_zero += 1
        elif all(reward == 1.0 for reward in rewards):
            all_one += 1
        else:
            mixed += 1
        advantages = [0.0 if variance == 0.0 else (reward - group_mean) / denominator for reward in rewards]
        for trajectory, advantage in zip(group, advantages):
            loss = _trajectory_loss(policy, trajectory, advantage, ratio_clip, kl_coefficient)
            (loss / trajectory_count).backward()
            total_loss += float(loss.detach().cpu()) / trajectory_count
    parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters, gradient_norm_clip)
    optimizer.step()
    return UpdateSummary(
        total_loss,
        sum(item.reward for item in trajectories) / trajectory_count,
        all_zero,
        mixed,
        all_one,
        trajectory_count,
        sum(len(segment.output_ids) for item in trajectories for segment in item.segments),
    )


def supervised_train(
    policy: HFPolicy,
    demonstrations: Sequence[Sequence[TokenSegment]],
    *,
    optimizer_settings: dict[str, Any],
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, float]]:
    """Assistant-token-only CE, averaged per demonstration then per prompt batch."""
    torch = policy.torch
    optimizer = make_optimizer(policy, optimizer_settings)
    rng = random.Random(seed)
    order = list(range(len(demonstrations)))
    history: list[dict[str, float]] = []
    policy.model.train()
    for epoch in range(epochs):
        rng.shuffle(order)
        for batch_start in range(0, len(order), batch_size):
            indices = order[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for index in indices:
                assistant_log_probs = []
                for segment in demonstrations[index]:
                    log_probs = token_log_probs(policy.model, segment.context_ids, segment.output_ids, grad=True)
                    assistant_log_probs.append(log_probs)
                prompt_loss = -torch.cat(assistant_log_probs).mean()
                (prompt_loss / len(indices)).backward()
                batch_loss += float(prompt_loss.detach().cpu()) / len(indices)
            parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, float(optimizer_settings["gradient_norm_clip"]))
            optimizer.step()
            history.append({"epoch": float(epoch + 1), "batch_start": float(batch_start), "loss": batch_loss})
    return history

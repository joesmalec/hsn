"""Deterministic experiment caps that retain every configured task/variant.

Caps apply to documents, before the trainer shuffles them. Every nonempty group receives at
least one document; the remaining budget is apportioned by group size. Selected documents keep
their original relative order. Historical experiment records without a policy use the old prefix
selection when re-scored, so a new sampler never changes the meaning of stored evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from hsn.data.dataset import ByteDataset

STRATIFIED = "stratified_v1"
PREFIX = "prefix_v1"


def cap_dataset(
    dataset: ByteDataset, cap: int, *, seed: int, policy: str = STRATIFIED
) -> ByteDataset:
    """Select at most ``cap`` documents (0 means all); refuse an uncovered stratified group."""
    if policy not in (STRATIFIED, PREFIX):
        raise ValueError(f"unknown data selection policy {policy!r}")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError(f"document cap must be a non-negative integer, got {cap!r}")
    if cap == 0 or cap >= len(dataset):
        return dataset
    if policy == PREFIX:
        return ByteDataset(dataset.examples[:cap])
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(dataset.examples):
        groups[(str(example.meta["task"]), str(example.meta["variant"]))].append(index)
    keys = sorted(groups)
    if cap < len(keys):
        names = ["/".join(key) for key in keys]
        raise ValueError(
            f"document cap {cap} cannot cover all {len(keys)} task/variant groups: {names}; "
            "raise the cap or explicitly narrow data.tasks/data.variants"
        )
    # Reserve one per group, then Hamilton-apportion the residual available documents. Integer
    # arithmetic makes exact totals and tie-breaking independent of floating-point rounding.
    remaining = cap - len(keys)
    available = len(dataset) - len(keys)
    shares = [divmod(remaining * (len(groups[key]) - 1), available) for key in keys]
    quotas = [1 + whole for whole, _remainder in shares]
    for index in sorted(range(len(keys)), key=lambda i: (-shares[i][1], keys[i]))[
        : cap - sum(quotas)
    ]:
        quotas[index] += 1
    rng = np.random.default_rng([seed, 0x48534E17])
    selected: list[int] = []
    for key, quota in zip(keys, quotas, strict=True):
        selected.extend(int(i) for i in rng.choice(groups[key], size=quota, replace=False))
    return ByteDataset([dataset.examples[i] for i in sorted(selected)])


def dataset_counts(dataset: ByteDataset) -> dict[str, Any]:
    """Document, input-token and supervised-position counts, overall and per task/variant."""
    groups: dict[str, dict[str, int]] = {}
    for example in dataset.examples:
        key = f"{example.meta['task']}/{example.meta['variant']}"
        counts = groups.setdefault(key, {"n_docs": 0, "n_tokens": 0, "n_loss": 0})
        counts["n_docs"] += 1
        counts["n_tokens"] += example.length
        counts["n_loss"] += int(example.loss_mask.sum())
    return {
        **{
            name: sum(group[name] for group in groups.values())
            for name in ("n_docs", "n_tokens", "n_loss")
        },
        "groups": dict(sorted(groups.items())),
    }


def recorded_policy(record: Mapping[str, Any]) -> str:
    """Absent metadata identifies a historical prefix-selected run; unknown policies raise."""
    selection = record.get("data_selection") or {}
    policy = str(selection.get("policy", PREFIX))
    if policy not in (STRATIFIED, PREFIX):
        raise ValueError(f"unknown recorded data selection policy {policy!r}")
    return policy

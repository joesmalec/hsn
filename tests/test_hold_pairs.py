"""The pair hold-out of the delayed task (``hold_mod``, ``hold_role``; 13 Nov 2026): keys of two
bytes, with the pairs whose byte sum is a multiple of ``hold_mod`` either never used
(``exclude``) or the only ones used (``only``). Off by default: the documents of before."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hsn.data.synthetic import TaskSpec, generate, load_dataset_spec

REPO = Path(__file__).resolve().parents[1]
KEYS = load_dataset_spec(REPO / "configs" / "data" / "synthetic_keys.yaml")
HOLD = load_dataset_spec(REPO / "configs" / "data" / "synthetic_hold.yaml")


def _task(dataset, name: str) -> TaskSpec:
    return next(t for t in dataset.tasks if t.name == name and t.variant == "in_range")


def _with(spec: TaskSpec, **params: object) -> TaskSpec:
    dump = spec.model_dump()
    dump["params"] = {**dump["params"], **params}
    return TaskSpec.model_validate(dump)


def _keys(example, lo: int, hi: int) -> list[tuple[int, int]]:
    """The two-byte keys of a delayed2 document's facts (key bytes lie in [lo, hi))."""
    t = example.tokens.tolist()
    i = 1
    while not lo <= t[i] < hi:
        i += 1
    keys = []
    while lo <= t[i] < hi:
        keys.append((t[i], t[i + 1]))
        i += 3
    return keys


def test_default_is_the_generator_of_before() -> None:
    plain = _task(KEYS, "delayed2")
    explicit = _with(plain, hold_mod=0, hold_role="exclude")
    for split in ("train", "dev"):
        for index in range(40):
            a, b = generate(plain, split, index), generate(explicit, split, index)
            assert np.array_equal(a.tokens, b.tokens) and np.array_equal(a.loss_mask, b.loss_mask)


@pytest.mark.parametrize(
    ("role", "split"), [("exclude", "train"), ("only", "train"), ("exclude", "dev")]
)
def test_documents_hold_one_kind_of_pair(role: str, split: str) -> None:
    spec = _with(_task(HOLD, "delayed2h"), hold_role=role)
    lo, hi = spec.params["keys"][split]
    seen: set[tuple[int, int]] = set()
    shared = 0
    for index in range(300):
        example = generate(spec, split, index)  # generate() checks it against the reference solver
        keys = _keys(example, lo, hi)
        assert keys and len(set(keys)) == len(keys)
        for a, b in keys:
            assert lo <= a < hi and lo <= b < hi
            assert ((a + b) % 4 == 0) == (role == "only"), (role, a, b)
        seen.update(keys)
        firsts, seconds = [k[0] for k in keys], [k[1] for k in keys]
        shared += len(set(firsts)) < len(keys) or len(set(seconds)) < len(keys)
    assert len(seen) > (200 if split == "train" else 150)  # many different pairs
    assert shared > 100  # keys of one document still share bytes


def test_every_training_byte_is_seen_in_both_positions_and_no_held_pair_is() -> None:
    spec = _task(HOLD, "delayed2h")
    lo, hi = spec.params["keys"]["train"]
    first, second = set(), set()
    for index in range(1500):
        for a, b in _keys(generate(spec, "train", index), lo, hi):
            assert (a + b) % 4 != 0
            first.add(a)
            second.add(b)
    assert first == set(range(lo, hi)) and second == set(range(lo, hi))


def test_generation_is_a_function_of_spec_split_and_index() -> None:
    spec = _task(HOLD, "delayed2h")
    a, b = generate(spec, "train", 7), generate(spec, "train", 7)
    assert np.array_equal(a.tokens, b.tokens)
    assert not np.array_equal(a.tokens[:8], generate(spec, "train", 8).tokens[:8])


def test_hold_out_needs_two_byte_keys() -> None:
    plain = _task(KEYS, "delayed2")
    with pytest.raises(ValueError, match="hold_mod"):
        _with(plain, hold_mod=4, key_len=3)
    with pytest.raises(ValueError, match="hold_mod"):
        _with(plain, hold_mod=1)
    with pytest.raises(ValueError, match="hold_role"):
        _with(plain, hold_role="only")


def test_the_kv_task_is_synthetic_v1s() -> None:
    v1 = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
    ours, theirs = _task(HOLD, "kv"), _task(v1, "kv")
    for index in range(20):
        assert np.array_equal(
            generate(ours, "train", index).tokens, generate(theirs, "train", index).tokens
        )

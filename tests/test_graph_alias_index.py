"""``hardware.graph_alias_index`` (opt-in): PyTorch's alias check at the recording of a CUDA graph
answered from an index (:func:`hsn.train.compile.install_graph_alias_index`). The CUDA graphs
themselves need a card; here the index is checked against PyTorch's own walk on a path of
recorded outputs built by hand, and the installation against the internals it expects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hsn.config import config_hash, load_config
from hsn.train.compile import install_graph_alias_index, live_alias_index

trees = pytest.importorskip("torch._inductor.cudagraph_trees")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"


class _Ref:
    """What the walk needs of a ``StorageWeakRefWrapper``: a call that returns the storage or
    ``None`` when it has died, and the recorded data pointer."""

    def __init__(self, ptr: int, alive: bool) -> None:
        self.ptr, self.alive = ptr, alive

    def __call__(self) -> object | None:
        return object() if self.alive else None

    def data_ptr(self) -> int:
        return self.ptr


def _path(tensors: list[torch.Tensor]) -> list[list[_Ref | None]]:
    """Three nodes of recorded outputs: live and dead references to the tensors' storages (some
    twice, so that the first match matters), references to other storages, and empty slots."""
    ptrs = [t.untyped_storage().data_ptr() for t in tensors]
    return [
        [_Ref(ptrs[0], False), None, _Ref(ptrs[1], True), _Ref(12345, True)],
        [_Ref(ptrs[0], True), _Ref(ptrs[1], True), None, _Ref(ptrs[2], False)],
        [_Ref(ptrs[0], True), _Ref(ptrs[3], True), _Ref(ptrs[2], False)],
    ]


def test_the_index_answers_as_pytorchs_walk() -> None:
    tensors = [torch.zeros(4) for _ in range(5)]  # the last one aliases nothing on the path
    view = tensors[1][1:]  # a view shares its base's storage
    node = SimpleNamespace(path_weakrefs=_path(tensors))
    walk = trees.CUDAGraphNode._is_alias_of_live_recorded_tensor
    index = live_alias_index(node.path_weakrefs, trees.maybe_deref)
    answers = []
    for t in [*tensors, view]:
        want = walk(node, t)
        assert index.get(t.untyped_storage().data_ptr()) == want
        answers.append(want)
    # live first matches, a storage whose references are all dead, and one that is not there
    assert answers == [(1, 0), (0, 2), None, (2, 1), None, (0, 2)]


def test_installation_is_idempotent_and_keeps_the_walk_outside_a_recording() -> None:
    node_type = trees.CUDAGraphNode
    assert install_graph_alias_index() is True
    assert install_graph_alias_index() is True  # a second call changes nothing
    assert node_type._hsn_alias_index is True
    tensors = [torch.zeros(4) for _ in range(5)]
    node = SimpleNamespace(path_weakrefs=_path(tensors), _hsn_live_index=None)
    # outside _add_first_outputs there is no index and the original walk answers
    assert node_type._is_alias_of_live_recorded_tensor(node, tensors[0]) == (1, 0)
    node._hsn_live_index = {tensors[0].untyped_storage().data_ptr(): (9, 9)}
    assert node_type._is_alias_of_live_recorded_tensor(node, tensors[0]) == (9, 9)
    assert node_type._is_alias_of_live_recorded_tensor(node, tensors[4]) is None


def test_the_key_is_opt_in_and_left_out_of_the_hash() -> None:
    base = load_config(RECIPE_A6)
    assert base.hardware.graph_alias_index is False
    on = load_config(RECIPE_A6, ["hardware.graph_alias_index=true"])
    assert on.hardware.graph_alias_index is True
    assert config_hash(load_config(RECIPE_A6, ["hardware.graph_alias_index=false"])) == config_hash(
        base
    )

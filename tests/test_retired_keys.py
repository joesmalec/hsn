"""Run folders written by the server trees that carried the dropped amendment A10 hold its three
keys at their off values in ``config.resolved.yaml``. They load as the same configuration (the
keys were left out of the hash at their defaults); a key that is switched on is refused."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hsn.config import RETIRED_KEYS, config_hash, dump_resolved, load_config

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"


def _resolved_with(tmp_path: Path, **keys: object) -> Path:
    cfg = load_config(RECIPE_A6, ["model.k_select=12", "model.n_cells=64"])
    path = tmp_path / "config.resolved.yaml"
    dump_resolved(cfg, path)
    raw = yaml.safe_load(path.read_text())
    for dotted, value in keys.items():
        section, key = dotted.split("__")
        raw[section][key] = value
    path.write_text(yaml.safe_dump(raw))
    return path


def test_off_values_load_as_the_same_configuration(tmp_path: Path) -> None:
    plain = load_config(RECIPE_A6, ["model.k_select=12", "model.n_cells=64"])
    path = _resolved_with(
        tmp_path, model__write_gate=False, model__write_init=2.0, train__lambda_w=0.0
    )
    loaded = load_config(path)
    assert loaded == plain and config_hash(loaded) == config_hash(plain)
    assert not hasattr(loaded.model, "write_gate") and not hasattr(loaded.train, "lambda_w")


@pytest.mark.parametrize(
    ("key", "value"),
    [("model__write_gate", True), ("model__write_init", 0.0), ("train__lambda_w", 0.1)],
)
def test_a_retired_key_that_is_on_is_refused(tmp_path: Path, key: str, value: object) -> None:
    path = _resolved_with(tmp_path, **{key: value})
    with pytest.raises(ValueError, match="A10"):
        load_config(path)


def test_an_unknown_key_is_still_refused(tmp_path: Path) -> None:
    path = _resolved_with(tmp_path, model__no_such_key=1)
    with pytest.raises(ValueError):
        load_config(path)
    assert set(RETIRED_KEYS) == {"model.write_gate", "model.write_init", "train.lambda_w"}

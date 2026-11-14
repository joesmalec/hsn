"""Keys of a dropped experiment whose code is not in the tree (``hsn.config.RETIRED_KEYS``): a
resolved configuration that holds one at its off value loads as the same configuration, and one
that is switched on is refused. The list is empty at present: amendment A10's three keys were
retired on 13 Nov 2026 and are schema keys again, so the run folders written with them at their
off values load as they are."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hsn import config as config_module
from hsn.config import RETIRED_KEYS, config_hash, dump_resolved, load_config

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
BASE = ["model.k_select=12", "model.n_cells=64"]


def _resolved_with(tmp_path: Path, **keys: object) -> Path:
    cfg = load_config(RECIPE_A6, BASE)
    path = tmp_path / "config.resolved.yaml"
    dump_resolved(cfg, path)
    raw = yaml.safe_load(path.read_text())
    for dotted, value in keys.items():
        section, key = dotted.split("__")
        raw[section][key] = value
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def retired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        config_module.RETIRED_KEYS, "model.old_switch", (False, "a dropped experiment")
    )


@pytest.mark.usefixtures("retired")
def test_off_values_load_as_the_same_configuration(tmp_path: Path) -> None:
    plain = load_config(RECIPE_A6, BASE)
    loaded = load_config(_resolved_with(tmp_path, model__old_switch=False))
    assert loaded == plain and config_hash(loaded) == config_hash(plain)
    assert not hasattr(loaded.model, "old_switch")


@pytest.mark.usefixtures("retired")
def test_a_retired_key_that_is_on_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="a dropped experiment"):
        load_config(_resolved_with(tmp_path, model__old_switch=True))


def test_an_unknown_key_is_still_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_config(_resolved_with(tmp_path, model__no_such_key=1))
    assert RETIRED_KEYS == {}


def test_a10_run_folders_load_as_they_are(tmp_path: Path) -> None:
    """A10's keys at their off values (as the server trees of 13 Nov wrote them) are the plain
    configuration: schema defaults, left out of the hash."""
    plain = load_config(RECIPE_A6, BASE)
    path = _resolved_with(
        tmp_path, model__write_gate=False, model__write_init=2.0, train__lambda_w=0.0
    )
    loaded = load_config(path)
    assert loaded == plain and config_hash(loaded) == config_hash(plain)

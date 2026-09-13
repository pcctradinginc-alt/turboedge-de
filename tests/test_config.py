from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from turboedge.config import CONFIG_FILES, ConfigError, TurboEdgeConfig, config_hash, load_config
from turboedge.universe.underlying_map import UNDERLYINGS


def test_load_config_from_real_configs_dir(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    assert isinstance(cfg, TurboEdgeConfig)
    assert cfg.default.default_top > 0
    assert "ecb" in cfg.sources
    assert cfg.sources["ecb"].enabled is True
    assert cfg.models.tsmom_horizon_norm_v1.lookbacks == [21, 63, 126]
    assert cfg.models.tsmom_horizon_norm_v1.ewma_lambda == pytest.approx(0.94)


def test_universe_config_ids_are_known_canonical_underlyings(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    for entry in cfg.universe.underlyings:
        assert entry.id in UNDERLYINGS, f"{entry.id} missing from underlying_map"


def test_phase1_enabled_underlyings_match_contract(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    assert set(cfg.universe.enabled_ids()) == {"DAX", "NDX", "XAU", "EURUSD"}


def test_config_hash_is_stable(config_dir: Path) -> None:
    cfg1 = load_config(config_dir)
    cfg2 = load_config(config_dir)
    assert config_hash(cfg1) == config_hash(cfg2)
    assert len(config_hash(cfg1)) == 64  # sha256 hex digest


def test_config_hash_changes_with_content(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    original_hash = config_hash(cfg)
    bumped_default = cfg.default.model_copy(update={"default_top": cfg.default.default_top + 1})
    mutated = cfg.model_copy(update={"default": bumped_default})
    assert config_hash(mutated) != original_hash


def test_missing_config_directory_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "does-not-exist")


def test_missing_config_file_raises_config_error(tmp_path: Path, config_dir: Path) -> None:
    for name in CONFIG_FILES:
        if name == "governance.yaml":
            continue
        (tmp_path / name).write_text((config_dir / name).read_text())
    # governance.yaml intentionally omitted
    with pytest.raises(ConfigError, match=r"governance\.yaml"):
        load_config(tmp_path)


def test_invalid_yaml_content_raises_config_error(tmp_path: Path, config_dir: Path) -> None:
    for name in CONFIG_FILES:
        (tmp_path / name).write_text((config_dir / name).read_text())

    # Break risk.yaml: max_spread_pct must be > 0.
    risk = yaml.safe_load((config_dir / "risk.yaml").read_text())
    risk["max_spread_pct"] = -1.0
    (tmp_path / "risk.yaml").write_text(yaml.safe_dump(risk))

    with pytest.raises(ConfigError, match="max_spread_pct"):
        load_config(tmp_path)


def test_default_config_dir_and_state_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from turboedge.config import default_config_dir, default_state_dir

    monkeypatch.delenv("TURBOEDGE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("TURBOEDGE_STATE_DIR", raising=False)
    assert default_config_dir() == Path("./configs")
    assert default_state_dir() == Path("./state")

    monkeypatch.setenv("TURBOEDGE_CONFIG_DIR", "/tmp/cfg")
    monkeypatch.setenv("TURBOEDGE_STATE_DIR", "/tmp/state")
    assert default_config_dir() == Path("/tmp/cfg")
    assert default_state_dir() == Path("/tmp/state")

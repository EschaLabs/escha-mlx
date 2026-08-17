"""Environment registry, parsing, layering, and architecture gates."""
from __future__ import annotations

from pathlib import Path

import pytest

from escha_mlx import envs, quant


EXPECTED_LAYERS = {
    envs.EnvLayer.BUILD: set(),
    envs.EnvLayer.DEPLOYMENT: {"ESCHA_MLX_WIRED_GB"},
    envs.EnvLayer.RUNTIME: {
        "ESCHA_MLX_GDN_STATE",
        "ESCHA_MLX_LAST_LOGIT",
        "ESCHA_MLX_Q8_GROUP",
        "ESCHA_MLX_BIAS",
    },
    envs.EnvLayer.KERNEL: {
        "ESCHA_MLX_DENSE",
        "ESCHA_MLX_MOE",
        "ESCHA_MLX_FUSED_HAD",
        "ESCHA_MLX_LUT",
        "ESCHA_MLX_GEMV",
        "ESCHA_MLX_FETCH",
        "ESCHA_MLX_SPLITK",
        "ESCHA_MLX_BLOCK_R",
        "ESCHA_MLX_KT_BLOCK",
        "ESCHA_MLX_PREFETCH",
        "ESCHA_MLX_SORTX",
        "ESCHA_MLX_LINEAR",
        "ESCHA_MLX_GEMV_PF",
        "ESCHA_MLX_DENSE_BLOCK_R",
        "ESCHA_MLX_DENSE_MAT",
    },
    envs.EnvLayer.DEVELOPMENT: {
        "ESCHA_MODEL",
        "ESCHA_DENSE_MODEL",
        "ESCHA_MLX_SLOW_TESTS",
    },
}


def _clear_escha_environment(monkeypatch) -> None:
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)


def test_registry_has_one_layer_and_description_per_variable():
    all_layered = set()
    for layer, expected in EXPECTED_LAYERS.items():
        actual = {variable.name for variable in envs.variables_in(layer)}
        assert actual == expected
        assert not all_layered.intersection(actual)
        all_layered.update(actual)

    assert all_layered == set(envs.environment_variables)
    assert all(variable.description for variable in envs.environment_variables.values())


def test_defaults_and_call_site_default(monkeypatch):
    _clear_escha_environment(monkeypatch)

    assert envs.ESCHA_MLX_WIRED_GB.get() is None
    assert envs.ESCHA_MLX_GDN_STATE.get() == "fp16"
    assert envs.ESCHA_MLX_LAST_LOGIT.get() is True
    assert envs.ESCHA_MLX_Q8_GROUP.get() == 128
    assert envs.ESCHA_MLX_Q8_GROUP.get() == quant.DEFAULT_GROUP
    assert envs.ESCHA_MLX_BIAS.get() is False
    assert envs.ESCHA_MLX_DENSE.get() == "q8"
    assert envs.ESCHA_MLX_MOE.get() is None
    assert envs.ESCHA_MLX_MOE.get(default="ops") == "ops"
    assert envs.ESCHA_MLX_SPLITK.get() == 1
    assert envs.ESCHA_MLX_BLOCK_R.get() is None
    assert envs.ESCHA_MLX_KT_BLOCK.get() == 4
    assert envs.ESCHA_MLX_LINEAR.get() is None
    assert envs.ESCHA_MLX_GEMV_PF.get() == 1
    assert envs.ESCHA_MLX_DENSE_BLOCK_R.get() is None
    assert envs.ESCHA_MLX_DENSE_MAT.get() is False
    assert envs.ESCHA_MODEL.get() is None
    assert envs.ESCHA_DENSE_MODEL.get() is None
    assert envs.ESCHA_MLX_SLOW_TESTS.get() is False


@pytest.mark.parametrize("raw,want", [
    ("1", True),
    ("true", True),
    ("YES", True),
    ("on", True),
    ("0", False),
    ("false", False),
    ("NO", False),
    ("off", False),
])
def test_boolean_parsing_is_consistent(monkeypatch, raw, want):
    monkeypatch.setenv("ESCHA_MLX_FUSED_HAD", raw)
    assert envs.ESCHA_MLX_FUSED_HAD.get() is want


def test_values_are_read_dynamically(monkeypatch):
    monkeypatch.setenv("ESCHA_MLX_FUSED_HAD", "0")
    assert envs.ESCHA_MLX_FUSED_HAD.get() is False
    monkeypatch.setenv("ESCHA_MLX_FUSED_HAD", "1")
    assert envs.ESCHA_MLX_FUSED_HAD.get() is True


@pytest.mark.parametrize("variable,raw,want", [
    (envs.ESCHA_MLX_GDN_STATE, "FP32", "fp32"),
    (envs.ESCHA_MLX_GDN_STATE, "BF16", "bf16"),
    (envs.ESCHA_MLX_Q8_GROUP, "64", 64),
    (envs.ESCHA_MLX_WIRED_GB, "20.5", 20.5),
    (envs.ESCHA_MLX_GEMV, "STAGED", "staged"),
    (envs.ESCHA_MLX_SPLITK, "auto", "auto"),
    (envs.ESCHA_MLX_SPLITK, "4", 4),
    (envs.ESCHA_MLX_BLOCK_R, "12", 12),
    (envs.ESCHA_MLX_KT_BLOCK, "32", 32),
    (envs.ESCHA_MLX_LINEAR, "OPS", "ops"),
    (envs.ESCHA_MLX_GEMV_PF, "4", 4),
    (envs.ESCHA_MLX_DENSE_BLOCK_R, "16", 16),
])
def test_typed_parsing(monkeypatch, variable, raw, want):
    monkeypatch.setenv(variable.name, raw)
    assert variable.get() == want


@pytest.mark.parametrize("variable,raw", [
    (envs.ESCHA_MLX_FUSED_HAD, "sometimes"),
    (envs.ESCHA_MLX_GDN_STATE, "int8"),
    (envs.ESCHA_MLX_DENSE, "int4"),
    (envs.ESCHA_MLX_MOE, "fast"),
    (envs.ESCHA_MLX_GEMV, "auto"),
    (envs.ESCHA_MLX_FETCH, "direct"),
    (envs.ESCHA_MLX_SPLITK, "0"),
    (envs.ESCHA_MLX_BLOCK_R, "-1"),
    (envs.ESCHA_MLX_KT_BLOCK, "3"),
    (envs.ESCHA_MLX_WIRED_GB, "0"),
    (envs.ESCHA_MLX_WIRED_GB, "nan"),
    (envs.ESCHA_MLX_LINEAR, "turbo"),
    (envs.ESCHA_MLX_GEMV_PF, "0"),
    (envs.ESCHA_MLX_DENSE_BLOCK_R, "0"),
])
def test_invalid_values_name_the_variable(monkeypatch, variable, raw):
    monkeypatch.setenv(variable.name, raw)
    with pytest.raises(ValueError, match=variable.name):
        variable.get()


def test_validate_environment_can_be_scoped(monkeypatch):
    _clear_escha_environment(monkeypatch)
    monkeypatch.setenv("ESCHA_MLX_GDN_STATE", "invalid")
    envs.validate_environment([envs.EnvLayer.KERNEL])
    with pytest.raises(ValueError, match="ESCHA_MLX_GDN_STATE"):
        envs.validate_environment([envs.EnvLayer.RUNTIME])


def test_package_reads_environment_only_through_registry():
    package = Path(envs.__file__).parent
    offenders = []
    for path in package.rglob("*.py"):
        if path == Path(envs.__file__):
            continue
        source = path.read_text()
        if "os.environ" in source or "os.getenv" in source:
            offenders.append(path.relative_to(package))
    assert not offenders, f"direct environment reads outside envs.py: {offenders}"


def test_all_registered_variables_are_documented():
    install = (Path(__file__).parents[1] / "docs" / "INSTALL.md").read_text()
    missing = [name for name in envs.environment_variables if name not in install]
    assert not missing, f"environment variables missing from INSTALL.md: {missing}"

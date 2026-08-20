# SPDX-License-Identifier: Apache-2.0
"""Regression tests: --mtp-num-draft-tokens must reach SimpleEngine.

The chained MTP path reads ``SimpleEngine._mtp_num_draft_tokens``, but the
CLI/server plumbing historically dropped the value (only ``mtp=`` was
forwarded), so serving always ran with the default of 1 and the status line
showed ``configured=1`` regardless of the flag. These tests pin every
construction path — no model loading, engines are mocked where construction
would load weights.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.test_cli import _serve_args


@pytest.fixture
def clean_server_globals(monkeypatch):
    """Give load_model a pristine global state and restore it afterwards."""
    import vllm_mlx.server as server

    for attr, value in (
        ("_engine", None),
        ("_residency_manager", None),
        ("_model_manager", None),
        ("_default_model_key", None),
        ("_auto_unload_idle_seconds", 0.0),
        ("_lazy_load_model", False),
        ("_lifespan_active", False),
        ("_model_name", None),
        ("_model_path", None),
    ):
        monkeypatch.setattr(server, attr, value, raising=False)
    return server


def test_cli_serve_passes_mtp_num_draft_tokens_to_load_model(monkeypatch):
    """cli serve (simple mode) must forward args.mtp_num_draft_tokens."""
    from vllm_mlx import cli, server
    from vllm_mlx.utils import download

    loaded = {}

    monkeypatch.setattr(
        download, "ensure_model_downloaded", lambda *a, **k: "local-test-model"
    )
    monkeypatch.setattr(
        server,
        "load_model",
        lambda *a, **k: loaded.update({"args": a, "kwargs": k}),
    )
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    cli.serve_command(_serve_args(enable_mtp=True, mtp_num_draft_tokens=3))

    assert loaded["kwargs"]["mtp"] is True
    assert loaded["kwargs"]["mtp_num_draft_tokens"] == 3


def test_cli_serve_registry_defaults_carry_mtp_num_draft_tokens(monkeypatch, tmp_path):
    """cli serve --models-config must seed registry defaults with the flag."""
    from vllm_mlx import cli, server
    from vllm_mlx.utils import download

    captured = {}

    monkeypatch.setattr(
        download, "ensure_model_downloaded", lambda *a, **k: "local-test-model"
    )
    monkeypatch.setattr(
        server,
        "load_model_registry",
        lambda models_config, defaults, **k: captured.update({"defaults": defaults}),
    )
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    models_config = tmp_path / "models.yaml"
    models_config.write_text("models: []\n")

    cli.serve_command(
        _serve_args(
            model=None,
            models_config=str(models_config),
            enable_mtp=True,
            mtp_num_draft_tokens=4,
        )
    )

    assert captured["defaults"].enable_mtp is True
    assert captured["defaults"].mtp_num_draft_tokens == 4


def test_load_model_direct_simple_engine_receives_draft_tokens(clean_server_globals):
    """server.load_model (eager simple mode) → SimpleEngine kwargs."""
    server = clean_server_globals
    fake_engine = MagicMock()
    fake_loop = MagicMock()

    with (
        patch.object(server, "SimpleEngine", return_value=fake_engine) as mock_engine,
        patch.object(server, "_detect_native_tool_support", return_value=False),
        patch("vllm_mlx.server.asyncio.new_event_loop", return_value=fake_loop),
        patch("vllm_mlx.server.asyncio.set_event_loop"),
    ):
        server.load_model(
            "test-model",
            use_batching=False,
            mtp=True,
            mtp_num_draft_tokens=3,
        )

    assert mock_engine.call_args.kwargs["mtp"] is True
    assert mock_engine.call_args.kwargs["mtp_num_draft_tokens"] == 3


def test_load_model_default_draft_tokens_is_one(clean_server_globals):
    """Omitting the kwarg keeps the depth-1 default."""
    server = clean_server_globals
    fake_engine = MagicMock()
    fake_loop = MagicMock()

    with (
        patch.object(server, "SimpleEngine", return_value=fake_engine) as mock_engine,
        patch.object(server, "_detect_native_tool_support", return_value=False),
        patch("vllm_mlx.server.asyncio.new_event_loop", return_value=fake_loop),
        patch("vllm_mlx.server.asyncio.set_event_loop"),
    ):
        server.load_model("test-model", use_batching=False, mtp=True)

    assert mock_engine.call_args.kwargs["mtp_num_draft_tokens"] == 1


def test_load_model_lazy_residency_spec_receives_draft_tokens(clean_server_globals):
    """server.load_model (auto-unload/lazy path) → ModelSpec → _build_engine."""
    server = clean_server_globals

    server.load_model(
        "test-model",
        mtp=True,
        mtp_num_draft_tokens=5,
        auto_unload_idle_seconds=60,
    )

    spec = server._residency_manager._residents["default"].spec
    assert spec.mtp is True
    assert spec.mtp_num_draft_tokens == 5

    # The deferred engine construction must forward the spec value.
    fake_engine = MagicMock()
    with patch(
        "vllm_mlx.engine.simple.SimpleEngine", return_value=fake_engine
    ) as mock_engine:
        server._build_engine(spec)

    assert mock_engine.call_args.kwargs["mtp"] is True
    assert mock_engine.call_args.kwargs["mtp_num_draft_tokens"] == 5


def test_simple_engine_stores_draft_tokens_attribute():
    """The engine attribute the chained MTP gate reads must match the kwarg."""
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine(
        "test-model",
        force_mllm=True,  # skip is_mllm_model() detection (no network)
        mtp=True,
        mtp_num_draft_tokens=3,
    )
    assert engine._mtp is True
    assert engine._mtp_num_draft_tokens == 3


def test_registry_resolution_carries_mtp_num_draft_tokens():
    """--models-config path: defaults flow into ResolvedModelConfig and
    per-entry overrides win; SimpleEngine construction gets the value."""
    from dataclasses import replace

    from vllm_mlx.model_registry import ModelManager, RegisteredModel

    from tests.test_model_registry import _defaults, _manager_config

    defaults = replace(_defaults(), enable_mtp=True, mtp_num_draft_tokens=3)
    manager = ModelManager(_manager_config(budget_gb=8), {}, defaults)

    inherited = RegisteredModel(
        name="alpha", source="alpha-src", estimated_memory_bytes=1024
    )
    config = manager._resolve_model_config(inherited, "alpha-src")
    assert config.enable_mtp is True
    assert config.mtp_num_draft_tokens == 3

    overridden = RegisteredModel(
        name="beta",
        source="beta-src",
        estimated_memory_bytes=1024,
        mtp_num_draft_tokens=7,
    )
    assert (
        manager._resolve_model_config(overridden, "beta-src").mtp_num_draft_tokens == 7
    )

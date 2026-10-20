"""draft_model.draft_policy: "default" leaves the engine's own draft length logic
alone, "tier" installs exllamav3's MTP draft policy on every generator the
container creates and takes over from dynamic_draft."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

pytest.importorskip("exllamav3")

from backends.exllamav3 import model as exl3_model  # noqa: E402
from common.config_models import DraftModelConfig  # noqa: E402
from endpoints.core.types.model import DraftModelLoadRequest  # noqa: E402


@pytest.mark.parametrize("policy", ["default", "tier"])
def test_config_accepts(policy):
    assert DraftModelConfig(draft_policy=policy).draft_policy == policy


def test_load_request_accepts():
    assert DraftModelLoadRequest(draft_policy="tier").draft_policy == "tier"
    assert DraftModelLoadRequest().draft_policy is None


def test_config_default_and_rejects_unknown():
    assert DraftModelConfig().draft_policy == "default"
    with pytest.raises(ValidationError):
        DraftModelConfig(draft_policy="fast")


@pytest.mark.parametrize(
    "args, mode, want",
    [
        ({}, "mtp", "default"),
        ({"draft_policy": None}, "mtp", "default"),
        ({"draft_policy": "tier"}, "mtp", "tier"),
        ({"draft_policy": "tier"}, "disabled", "default"),
        ({"draft_policy": "default"}, "model", "default"),
    ],
)
def test_resolve(args, mode, want):
    assert exl3_model.resolve_draft_policy(args, mode) == want


@pytest.mark.parametrize("mode", ["model", "ngram"])
def test_tier_needs_mtp(mode):
    with pytest.raises(ValueError, match="mtp"):
        exl3_model.resolve_draft_policy({"draft_policy": "tier"}, mode)


def test_tier_overrides_dynamic_draft(monkeypatch):
    warnings = []
    monkeypatch.setattr(exl3_model.xlogger, "warning", lambda msg, *a, **k: warnings.append(msg))
    args = {"draft_policy": "tier", "dynamic_draft": True}
    assert exl3_model.resolve_draft_policy(args, "mtp") == "tier"
    assert len(warnings) == 1 and "dynamic_draft" in warnings[0]


@pytest.fixture
def fake_spec_policy(monkeypatch):
    calls = []
    mod = ModuleType("exllamav3.generator.spec_policy")
    mod.install_draft_policy = lambda g: calls.append(g) or "uninstall"
    monkeypatch.setitem(sys.modules, "exllamav3.generator.spec_policy", mod)
    return calls


def test_install_default_leaves_generator(fake_spec_policy):
    assert exl3_model.install_draft_policy(SimpleNamespace(), "default") is None
    assert fake_spec_policy == []


def test_install_tier_on_the_sync_generator(fake_spec_policy):
    inner = SimpleNamespace()
    assert exl3_model.install_draft_policy(inner, "tier") == "uninstall"
    assert fake_spec_policy == [inner]


def test_install_tier_needs_engine_support(monkeypatch):
    monkeypatch.setitem(sys.modules, "exllamav3.generator.spec_policy", None)
    with pytest.raises(ValueError, match="spec_policy"):
        exl3_model.install_draft_policy(SimpleNamespace(), "tier")

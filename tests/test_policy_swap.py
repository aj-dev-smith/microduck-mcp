"""duck_policy's server half: the refuse-before-touch contract.

_handle_policy promises that a bad swap can never brick the body: the new
session is built and validated OFF TO THE SIDE, and every refusal arrives
with the incumbent still bound. These tests pin that contract without a sim
or real ONNX files — a bare DuckSim shell carries just `policy` and
`policy_paths`, and `onnxruntime` is stubbed at sys.modules so the handler's
local import picks up fakes.
"""

import sys
import types

import pytest

from microduck_mcp.sim_server import DuckSim


class FakeInput:
    def __init__(self, dim):
        self.shape = [1, dim]


class FakeSession:
    def __init__(self, dim):
        self._dim = dim

    def get_inputs(self):
        return [FakeInput(self._dim)]


class FakePolicy:
    def __init__(self):
        self.walking_session = FakeSession(61)
        self.standing_session = FakeSession(61)
        self.sit_session = FakeSession(61)
        self.ground_pick_session = None
        self.behavior_sessions = {"kick_left": FakeSession(61)}
        self.is_sitstand = False
        self.current_policy = "standing"


def make_server():
    srv = object.__new__(DuckSim)  # no sim: the handler touches only these two
    srv.policy = FakePolicy()
    srv.policy_paths = {"sitstand": "/old/standup_v2.onnx"}
    return srv


def stub_ort(monkeypatch, factory):
    """The handler does `import onnxruntime as ort` locally; route it here."""
    mod = types.ModuleType("onnxruntime")
    mod.InferenceSession = factory
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)


def onnx_file(tmp_path):
    f = tmp_path / "brain.onnx"
    f.write_bytes(b"not really onnx; the stubbed runtime never reads it")
    return str(f)


# ── the role map itself ───────────────────────────────────────────────────────

def test_role_map_is_disjoint_and_names_real_attrs():
    overlap = set(DuckSim.POLICY_SESSION_ATTRS) & set(DuckSim.BEHAVIOR_ROLES)
    assert not overlap, "a role must be a session attr OR a behavior, never both"
    p = FakePolicy()
    for attr in DuckSim.POLICY_SESSION_ATTRS.values():
        assert hasattr(p, attr), f"PolicyInference contract drifted: no {attr}"
    # the legacy naming the map exists to paper over
    assert DuckSim.POLICY_SESSION_ATTRS["sitstand"] == "sit_session"


def test_list_reports_every_role_with_load_state():
    srv = make_server()
    out = srv._handle_policy({"action": "list"})
    assert out["ok"]
    by_role = {s["role"]: s for s in out["slots"]}
    assert set(by_role) == set(DuckSim.POLICY_SESSION_ATTRS) | set(DuckSim.BEHAVIOR_ROLES)
    assert by_role["walking"]["loaded"] and by_role["walking"]["obs_dim"] == 61
    assert not by_role["ground_pick"]["loaded"] and by_role["ground_pick"]["obs_dim"] is None
    assert by_role["sitstand"]["file"] == "/old/standup_v2.onnx"


# ── refusals, each with the incumbent untouched ──────────────────────────────

def test_unknown_role_refused():
    out = make_server()._handle_policy({"action": "swap", "role": "flying", "path": "x"})
    assert not out["ok"] and "unknown role" in out["error"]


def test_missing_file_refused(tmp_path):
    srv = make_server()
    out = srv._handle_policy(
        {"action": "swap", "role": "sitstand", "path": str(tmp_path / "ghost.onnx")})
    assert not out["ok"] and "no ONNX file" in out["error"]
    assert srv.policy_paths["sitstand"] == "/old/standup_v2.onnx"


def test_load_failure_refused_before_touch(tmp_path, monkeypatch):
    def explode(path):
        raise RuntimeError("malformed protobuf")
    stub_ort(monkeypatch, explode)
    srv = make_server()
    incumbent = srv.policy.sit_session
    out = srv._handle_policy({"action": "swap", "role": "sitstand", "path": onnx_file(tmp_path)})
    assert not out["ok"] and "incumbent untouched" in out["error"]
    assert srv.policy.sit_session is incumbent
    assert srv.policy_paths["sitstand"] == "/old/standup_v2.onnx"


def test_obs_width_mismatch_refused_before_touch(tmp_path, monkeypatch):
    stub_ort(monkeypatch, lambda path: FakeSession(74))  # critic-width export bug
    srv = make_server()
    incumbent = srv.policy.sit_session
    out = srv._handle_policy({"action": "swap", "role": "sitstand", "path": onnx_file(tmp_path)})
    assert not out["ok"] and "obs contract mismatch" in out["error"]
    assert "61D" in out["error"] and "74D" in out["error"]
    assert srv.policy.sit_session is incumbent
    assert srv.policy_paths["sitstand"] == "/old/standup_v2.onnx"


# ── the successful swap ──────────────────────────────────────────────────────

def test_swap_rebinds_and_hands_back_the_rollback_path(tmp_path, monkeypatch):
    fresh = FakeSession(61)
    stub_ort(monkeypatch, lambda path: fresh)
    srv = make_server()
    path = onnx_file(tmp_path)
    out = srv._handle_policy({"action": "swap", "role": "sitstand", "path": path})
    assert out["ok"]
    assert srv.policy.sit_session is fresh
    assert srv.policy.is_sitstand, "a sitstand swap must arm the sit trick"
    assert out["previous"] == "/old/standup_v2.onnx", "the one-call rollback"
    assert srv.policy_paths["sitstand"] == out["file"]


def test_empty_slot_accepts_any_width(tmp_path, monkeypatch):
    stub_ort(monkeypatch, lambda path: FakeSession(97))  # no incumbent to disagree with
    srv = make_server()
    out = srv._handle_policy({"action": "swap", "role": "ground_pick", "path": onnx_file(tmp_path)})
    assert out["ok"] and out["obs_dim"] == 97
    assert out["previous"] is None


def test_behavior_role_lands_in_behavior_sessions(tmp_path, monkeypatch):
    fresh = FakeSession(61)
    stub_ort(monkeypatch, lambda path: fresh)
    srv = make_server()
    out = srv._handle_policy({"action": "swap", "role": "roulade", "path": onnx_file(tmp_path)})
    assert out["ok"]
    assert srv.policy.behavior_sessions["roulade"] is fresh
    assert not srv.policy.is_sitstand, "only a sitstand swap touches is_sitstand"

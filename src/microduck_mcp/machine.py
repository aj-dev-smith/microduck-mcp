"""The behavior machine: guarded states over the sensed digest, 50 Hz.

Ocarina's pattern (SURFACE.md/MACHINE.md in ~/Code/ocarina) ported to a robot:
the agent never streams commands — it arms a machine whose leaves are
deterministic behaviors executed at the control rate, with transitions guarded
by expressions over the SENSED digest. Fairness by construction: the guard
vocabulary contains only what the robot could genuinely know — camera-derived
ball sighting, proprioception, its own machine state — never the simulator's
ground-truth ball position.

Machine source is TOML (stdlib tomllib; no new dependencies), living in a git
repo the agent edits and hot-swaps via the `machine` socket command family.
Guards are expression strings compiled through the ast module against a strict
whitelist: dotted digest paths, numeric/string/bool literals, comparisons,
and/or/not. Nothing else parses, so nothing else runs.
"""

import ast
import math
import tomllib

import numpy as np

# Every path a guard may read. The digest is built each tick by the executor;
# ball fields come from the camera detector (fake mediad), the rest from
# proprioception and the machine itself.
GUARD_PATHS = {
    "ball_seen.visible", "ball_seen.distance_m", "ball_seen.ground_distance_m",
    "ball_seen.bearing_deg", "ball_seen.elevation_deg", "ball_seen.age_s",
    "ball_seen.est_forward_m", "ball_seen.est_left_m", "ball_seen.speed_mps",
    "upright", "sitting", "active_policy", "behavior",
    "elapsed_s", "sim_time_s", "node",
}

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Not, ast.USub, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Name, ast.Attribute, ast.Constant, ast.Load,
)


def _path_of(node):
    """Name/Attribute chain -> dotted path string, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class GuardError(ValueError):
    pass


def compile_guard(expr: str):
    """Validate a guard expression; returns an evaluator digest -> bool.

    Comparisons involving a null field (e.g. distance_m while not visible)
    evaluate False rather than raising — 'not visible' is the honest answer
    to any question about where an unseen ball is.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise GuardError(f"guard does not parse: {expr!r} ({e.msg})") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise GuardError(
                f"guard {expr!r}: {type(node).__name__} is not in the grammar")
        if isinstance(node, ast.Constant) and not isinstance(
                node.value, (bool, int, float, str)):
            raise GuardError(f"guard {expr!r}: literal {node.value!r} not allowed")
    # Check every complete Name/Attribute chain against the whitelist.
    class PathCheck(ast.NodeVisitor):
        def visit_Attribute(self, node):
            p = _path_of(node)
            if p is None:
                raise GuardError(f"guard {expr!r}: unsupported attribute base")
            if p not in GUARD_PATHS:
                raise GuardError(f"guard {expr!r}: unknown path {p!r}")
            # don't recurse into the chain

        def visit_Name(self, node):
            if node.id not in GUARD_PATHS:
                raise GuardError(f"guard {expr!r}: unknown path {node.id!r}")

    PathCheck().visit(tree)

    def evaluate(digest: dict) -> bool:
        def ev(node):
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.BoolOp):
                vals = [ev(v) for v in node.values]
                return all(vals) if isinstance(node.op, ast.And) else any(vals)
            if isinstance(node, ast.UnaryOp):
                v = ev(node.operand)
                if isinstance(node.op, ast.USub):
                    return None if v is None else -v
                return not v
            if isinstance(node, ast.Compare):
                left = ev(node.left)
                for op, comp in zip(node.ops, node.comparators):
                    right = ev(comp)
                    if left is None or right is None:
                        return False
                    if isinstance(op, ast.Eq):
                        ok = left == right
                    elif isinstance(op, ast.NotEq):
                        ok = left != right
                    elif isinstance(op, ast.Lt):
                        ok = left < right
                    elif isinstance(op, ast.LtE):
                        ok = left <= right
                    elif isinstance(op, ast.Gt):
                        ok = left > right
                    else:
                        ok = left >= right
                    if not ok:
                        return False
                    left = right
                return True
            if isinstance(node, (ast.Name, ast.Attribute)):
                return digest.get(_path_of(node))
            if isinstance(node, ast.Constant):
                return node.value
            raise GuardError(f"unreachable node {type(node).__name__}")
        return bool(ev(tree))

    return evaluate


# ---------- behaviors (the leaves) ----------
# Each is step(sim, params, mem, digest) called at 50 Hz while its node is
# current; `mem` is a per-entry dict (cleared on every node entry). Behaviors
# read the sensed digest — the same values guards see — plus proprioception.

def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _set_head(sim, pitch: float):
    """Tilt the head (and with it the camera) via the policy's gaze command —
    the same path the look intent uses, so the balance policy compensates."""
    p = sim.policy
    p.head_offset[:] = np.clip(
        np.array([0.0, pitch, 0.0, 0.0], dtype=np.float32),
        -p.head_max, p.head_max)
    p._update_command()


def bhv_idle(sim, params, mem, digest):
    if not mem.get("zeroed"):
        sim.policy.set_vel_cmd(0.0, 0.0, 0.0)
        _set_head(sim, 0.0)
        mem["zeroed"] = True


def bhv_drive(sim, params, mem, digest):
    """Constant velocity intent (params vx, vy, wz) — the building block for
    open-loop maneuvers. Example: a backoff node that walks backward for a
    couple of seconds after a failed kick, because the gait needs distance to
    rebuild real momentum — commanded creep from a standstill moves the duck
    millimeters, not centimeters."""
    if not mem.get("set"):
        sim.policy.set_vel_cmd(float(params.get("vx", 0.0)),
                               float(params.get("vy", 0.0)),
                               float(params.get("wz", 0.0)))
        mem["set"] = True


def bhv_search_ball(sim, params, mem, digest):
    """Rotate in place until a guard sees the ball. wz below ~1.2 is a dead
    zone, so search always turns at full rate. Head level: a level camera
    sees the far field; the approach looks down as it closes."""
    if not mem.get("head"):
        _set_head(sim, 0.0)
        mem["head"] = True
    wz = float(params.get("wz", 1.5))
    sim.policy.set_vel_cmd(0.0, 0.0, wz)


def bhv_approach_ball(sim, params, mem, digest):
    """Close on the ball and settle into the kick pocket — the session-tested
    controller (continuous sticky intents; full-throttle point turns; bearing
    target ~-25 deg so the ball lines up with the right foot).

    Looks DOWN as it closes: with the head level, a floor ball drops below
    the camera frame at ~0.15 m ground distance — before the 0.10 m pocket —
    so near-range tracking is physically impossible without the head tilt.
    """
    if not digest.get("ball_seen.visible"):
        sim.policy.set_vel_cmd(0.0, 0.0, 0.0)  # lost it; a guard will exit
        return
    fwd = digest.get("ball_seen.est_forward_m")
    left = digest.get("ball_seen.est_left_m")
    if fwd is None or left is None:
        sim.policy.set_vel_cmd(0.0, 0.0, 0.0)
        return
    # Trunk-frame geometry — the pocket the kick was measured in. d/bearing
    # here are TRUNK-relative, not camera-relative: the camera sits ~8 cm
    # ahead of the trunk, which is a huge error at kick range.
    d = math.hypot(fwd, left)
    b = math.atan2(left, fwd)
    # 0.5 rad + the 20 deg mount pitch keeps a floor ball in frame from
    # ~0.45 m all the way into the pocket; hysteresis so the tilt won't flap.
    head_down = float(params.get("head_down", 0.5))
    if d < 0.45:
        mem["tilted"] = True
    elif d > 0.60:
        mem["tilted"] = False
    _set_head(sim, head_down if mem.get("tilted") else 0.0)
    # Kick sweep (measured): forward is the critical axis — the ball connects
    # hard at true forward <= 0.09 m, dies at 0.10-0.12. The camera reads
    # ~1.5 cm long at point-blank (beak occlusion biases the blob), so the
    # sensed stop target sits closer than the old 0.099.
    if d > 0.18:
        tgt_b = math.radians(float(params.get("pocket_bearing_deg", -25.0)))
        if abs(b) > math.radians(35):
            vx, wz = 0.0, math.copysign(1.5, b)
        else:
            vx, wz = 0.3, _clip(2.5 * b, -1.0, 1.0)
    else:
        # Fine range: servo est_forward/est_left into the pocket guard's
        # window, so "controller satisfied" implies "guard fires". The creep
        # deliberately stops OUTSIDE the walking foot's reach (contact
        # forensics: the swinging right foot pokes any ball nearer than
        # ~0.09 true / ~0.11 sensed) — the kick guard fires while still
        # creeping, and the stop's own forward slide delivers the last
        # ~3 cm with feet planted. The stop closes the gap, not a step.
        vx = 0.2 if fwd > 0.110 else (-0.2 if fwd < 0.085 else 0.0)
        wz = 1.5 if left > -0.020 else (-1.5 if left < -0.080 else 0.0)
    sim.policy.set_vel_cmd(vx, 0.0, wz)


def bhv_kick(sim, params, mem, digest):
    """Stop, LOOK UP, settle, one honest kick, then look back down to verify.

    The head MUST be level for the swing: head_pose is part of the policy
    command vector, and the kick policy fed a bowed head does not swing at
    all (measured: 1.3-1.5 m with the head level, 0.00 m at 0.5 rad down,
    same ball, same stance). So the duck lines up watching the ball, looks
    up to kick like a striker, and glances down afterwards so the whiff
    guard has fresh eyes on where the ball ended up."""
    settle_s = float(params.get("settle_s", 0.8))
    if not mem.get("stopped"):
        sim.policy.set_vel_cmd(0.0, 0.0, 0.0)
        _set_head(sim, 0.0)
        mem["stopped"] = True
    if digest["elapsed_s"] >= settle_s and not mem.get("kicked"):
        mem["kicked"] = True
        foot = params.get("foot", "right")
        sim._handle_trick(f"kick_{foot}", stage_ball=bool(params.get("stage_ball", False)))
    # Swing done (trick auto-returns 3 s after trigger): look down again so
    # the whiff-check guard gets a fresh sighting of the pocket.
    if mem.get("kicked") and not mem.get("verified") \
            and digest["elapsed_s"] >= settle_s + 2.9:
        _set_head(sim, float(params.get("head_down", 0.5)))
        mem["verified"] = True


def bhv_celebrate(sim, params, mem, digest):
    if not mem.get("done"):
        mem["done"] = True
        sim._handle_trick(str(params.get("trick", "roulade")))


BEHAVIORS = {
    "idle": bhv_idle,
    "drive": bhv_drive,
    "search_ball": bhv_search_ball,
    "approach_ball": bhv_approach_ball,
    "kick": bhv_kick,
    "celebrate": bhv_celebrate,
}


# ---------- the machine ----------

class MachineError(ValueError):
    pass


class Machine:
    """A validated machine: nodes, compiled guards, and the 50 Hz executor."""

    def __init__(self, spec: dict, source_path: str = "<inline>"):
        self.source_path = source_path
        m = spec.get("machine")
        if not isinstance(m, dict):
            raise MachineError("missing [machine] table")
        self.name = m.get("name", "unnamed")
        nodes = spec.get("node")
        if not nodes:
            raise MachineError("no [[node]] tables")
        self.nodes = {}
        for n in nodes:
            name = n.get("name")
            if not name:
                raise MachineError("a [[node]] has no name")
            if name in self.nodes:
                raise MachineError(f"duplicate node {name!r}")
            bname = n.get("behavior", "idle")
            if bname not in BEHAVIORS:
                raise MachineError(
                    f"node {name!r}: unknown behavior {bname!r} "
                    f"(have: {', '.join(sorted(BEHAVIORS))})")
            trans = []
            for t in n.get("transition", []):
                if "when" not in t or "to" not in t:
                    raise MachineError(f"node {name!r}: transition needs when+to")
                trans.append({"when": t["when"],
                              "guard": compile_guard(t["when"]),
                              "to": t["to"]})
            self.nodes[name] = {"behavior": bname,
                                "params": n.get("params", {}),
                                "transitions": trans}
        # machine-level transitions (checked before the node's own — the
        # "fell over" escape hatch lives here)
        self.global_transitions = []
        for t in m.get("transition", []):
            if "when" not in t or "to" not in t:
                raise MachineError("[machine] transition needs when+to")
            self.global_transitions.append(
                {"when": t["when"], "guard": compile_guard(t["when"]),
                 "to": t["to"]})
        self.initial = m.get("initial")
        if self.initial not in self.nodes:
            raise MachineError(f"initial node {self.initial!r} does not exist")
        for src, n in self.nodes.items():
            for t in n["transitions"]:
                if t["to"] not in self.nodes:
                    raise MachineError(f"node {src!r}: transition to unknown "
                                       f"node {t['to']!r}")
        for t in self.global_transitions:
            if t["to"] not in self.nodes:
                raise MachineError(f"[machine] transition to unknown node {t['to']!r}")

        # live state
        self.armed = False
        self.current = self.initial
        self.entered_at = 0.0
        self.mem = {}

    @classmethod
    def load(cls, path: str) -> "Machine":
        with open(path, "rb") as f:
            try:
                spec = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise MachineError(f"{path}: TOML does not parse: {e}") from e
        return cls(spec, source_path=path)

    def enter(self, node: str, sim_time: float):
        self.current = node
        self.entered_at = sim_time
        self.mem = {}

    def status(self) -> dict:
        return {"name": self.name, "source": self.source_path,
                "armed": self.armed, "node": self.current,
                "nodes": sorted(self.nodes)}

    def tick(self, sim, digest: dict):
        """One 50 Hz step: transitions first (global, then node, in order),
        then the current behavior. Returns the fired transition or None."""
        digest["elapsed_s"] = round(digest["sim_time_s"] - self.entered_at, 3)
        node = self.nodes[self.current]
        for t in self.global_transitions + node["transitions"]:
            if t["to"] != self.current and t["guard"](digest):
                fired = {"from": self.current, "to": t["to"], "when": t["when"]}
                self.enter(t["to"], digest["sim_time_s"])
                digest["elapsed_s"] = 0.0
                return fired
        BEHAVIORS[node["behavior"]](sim, node["params"], self.mem, digest)
        return None

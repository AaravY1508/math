"""Stage 2: evolutionary search over the geometry of one net.

Stage 1 (`fence.nets`) hands over a ranked net. This module runs a population over
that net's *continuous* parameters -- vertex positions, and the position of each
T-junction along its host fence -- and refines every candidate with the existing
local optimizer (`fence.optimize`: batched Adam on the augmented Lagrangian, then
SLSQP polish, then the 50-digit referee). Only a referee-verified area ever counts
as a result.

Mutations come in two kinds, both registered separately and individually
toggleable by weight:

* *geometry* moves keep the net and move the parameters: jitter (everywhere, or
  only on the outer boundary so the grid part stays snapped -- heuristic 9),
  anisotropic stretch, sliding a T-junction along its host, and projecting onto a
  symmetric parameterization (heuristic 8).
* *topology* moves change the net locally, which is how the search leaves a net
  that has been squeezed dry: turning an endpoint-to-endpoint join into a
  T-junction and back, sliding a T-junction past its neighbour on the same host
  (a genuine change: it reorders the host's chain and so re-cuts the faces),
  re-pinning a stem onto a different edge, and adding or removing a face.

Every mutant goes through the same hard rules as stage 1 before it is allowed to
cost optimizer time, and every rejection is recorded in the same kind of `Ledger`,
so a run can say which check killed which candidate.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from fence import builder as Bd
from fence import geometry as G
from fence import heuristics as H
from fence import minima as Mn
from fence import optimize as O
from fence import rules as R


# ---------------------------------------------------------------------------
# mutation registry
# ---------------------------------------------------------------------------

@dataclass
class Move:
    id: str
    kind: str            # "geometry" or "topology"
    summary: str
    fn: object
    weight: float = 1.0


MOVES: dict[str, Move] = {}


def move(id: str, kind: str, summary: str, weight: float = 1.0):
    def deco(fn):
        MOVES[id] = Move(id, kind, summary, fn, weight)
        return fn
    return deco


def _restore(sk: Bd.Sketch, snap: Bd.Sketch) -> None:
    sk.P, sk.fences, sk.pins, sk._topo = snap.P, snap.fences, snap.pins, snap._topo


def _reproject_pins(sk: Bd.Sketch) -> None:
    """Put every pinned vertex back exactly on its host after the hosts moved."""
    for _ in range(4):
        X = sk.coords()
        moved = False
        for v, h in sk.pins.items():
            a, b = sk.fences[h]
            d = X[b] - X[a]
            dd = float(d @ d)
            if dd < 1e-18:
                continue
            t = min(0.98, max(0.02, float((X[v] - X[a]) @ d) / dd))
            q = X[a] + t * d
            if not np.allclose(q, X[v], atol=1e-14):
                sk.P[v] = [float(q[0]), float(q[1])]
                moved = True
        sk._dirty()
        if not moved:
            break


# -- geometry moves ----------------------------------------------------------

@move("jitter", "geometry", "gaussian jitter of every vertex", weight=1.0)
def _jitter(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    s = ctx.get("jitter_sigma", 0.05) * rng.uniform(0.3, 1.5)
    X = sk.coords()
    X = X + s * rng.standard_normal(X.shape)
    sk.P = X.tolist()
    sk._dirty()
    _reproject_pins(sk)
    return f"sigma={s:.3f}"


@move("jitter_boundary", "geometry",
      "jitter only the outer-boundary vertices, leaving the grid part snapped", weight=1.0)
def _jitter_boundary(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    try:
        outer = sorted(set(sk.topology().outer))
    except G.InvalidTopology:
        return None
    if not outer:
        return None
    s = ctx.get("jitter_sigma", 0.05) * rng.uniform(0.5, 2.0)
    X = sk.coords()
    X[outer] += s * rng.standard_normal((len(outer), 2))
    sk.P = X.tolist()
    sk._dirty()
    _reproject_pins(sk)
    return f"{len(outer)} boundary vertices, sigma={s:.3f}"


@move("stretch", "geometry", "anisotropic stretch about the centroid", weight=0.6)
def _stretch(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    a = 1.0 + ctx.get("stretch", 0.12) * rng.uniform(-1, 1)
    th = rng.uniform(0, math.pi)
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    S = Rm @ np.diag([a, 1.0 / a]) @ Rm.T
    X = sk.coords()
    c = X.mean(axis=0)
    sk.P = ((X - c) @ S.T + c).tolist()
    sk._dirty()
    _reproject_pins(sk)
    return f"a={a:.3f}"


@move("slide_t", "geometry", "slide a T-junction along its host (continuous)", weight=1.0)
def _slide_t(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    if not sk.pins:
        return None
    v = int(rng.choice(sorted(sk.pins)))
    h = sk.pins[v]
    a, b = sk.fences[h]
    X = sk.coords()
    t = G.fence_param(X, a, b, v)
    nt = min(0.95, max(0.05, t + ctx.get("slide", 0.12) * rng.uniform(-1, 1)))
    q = X[a] + nt * (X[b] - X[a])
    sk.P[v] = [float(q[0]), float(q[1])]
    sk._dirty()
    _reproject_pins(sk)
    return f"pin {v} on fence {h}: t {t:.3f} -> {nt:.3f}"


@move("symmetrize", "geometry", "project onto the nearest C_m-symmetric parameterization",
      weight=0.4)
def _symmetrize(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    m = int(rng.choice([2, 3, 4]))
    X = sk.coords()
    c = X.mean(axis=0)
    Y = X - c
    th = 2 * math.pi / m
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    # pull each vertex a step towards the orbit average of its nearest images
    acc = Y.copy()
    Z = Y.copy()
    for _ in range(m - 1):
        Z = Z @ Rm.T
        idx = np.argmin(np.linalg.norm(Z[:, None, :] - Y[None, :, :], axis=2), axis=0)
        acc = acc + Z[idx]
    tgt = acc / m
    lam = ctx.get("symmetrize_step", 0.35)
    sk.P = ((1 - lam) * Y + lam * tgt + c).tolist()
    sk._dirty()
    _reproject_pins(sk)
    return f"C{m} step {lam}"


# -- topology moves ----------------------------------------------------------

@move("join_to_t", "topology", "turn an endpoint-to-endpoint join into a T-junction", weight=1.0)
def _join_to_t(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    before = len(sk.pins)
    if not Bd.toggle_junction(sk, rng):
        return None
    return "join -> T" if len(sk.pins) > before else "T -> join"


@move("t_to_join", "topology", "merge a T-junction back into an endpoint join", weight=0.7)
def _t_to_join(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    if not sk.pins:
        return None
    ends = Bd._ends_at(sk)
    X = sk.coords()
    opts = sorted(sk.pins.items())
    rng.shuffle(opts)
    for v, h in opts[:6]:
        trial = sk.copy()
        a, b = trial.fences[h]
        t = G.fence_param(X, a, b, v)
        tgt = a if t < 0.5 else b
        for f in ends.get(v, []):
            trial.fences[f] = [tgt if u == v else u for u in trial.fences[f]]
        del trial.pins[v]
        if any(p == q for p, q in trial.fences):
            continue
        if Bd._commit(sk, trial):
            return f"pin {v} merged into vertex {tgt}"
    return None


@move("slide_t_past", "topology",
      "slide a T-junction past its neighbour on the same host: the chain re-orders", weight=0.8)
def _slide_t_past(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    """A genuine topology change, unlike `slide_t`.

    Two pins on the same host cut it into three graph-edges; exchanging their order
    along the host swaps which stem borders which face, so the face structure is a
    different net even though no incidence changed.
    """
    try:
        chains = sk.topology().chains
    except G.InvalidTopology:
        return None
    hosts = [h for h, ch in enumerate(chains) if len(ch) >= 4]
    if not hosts:
        return None
    rng.shuffle(hosts)
    X = sk.coords()
    for h in hosts[:4]:
        ch = chains[h]
        mids = ch[1:-1]
        k = int(rng.integers(len(mids) - 1))
        u, w = mids[k], mids[k + 1]
        a, b = sk.fences[h]
        tu, tw = G.fence_param(X, a, b, u), G.fence_param(X, a, b, w)
        trial = sk.copy()
        for v, t in ((u, tw), (w, tu)):
            q = X[a] + t * (X[b] - X[a])
            trial.P[v] = [float(q[0]), float(q[1])]
        trial._dirty()
        if Bd._commit(sk, trial):
            return f"pins {u} and {w} swapped along fence {h}"
    return None


@move("repin", "topology", "detach a fence end and pin it on a different edge", weight=1.0)
def _repin(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    return "repinned" if Bd.reattach(sk, rng) else None


@move("add_face", "topology", "add a cell of 1-3 fences (the fence count is fixed up after)",
      weight=0.9)
def _add_face(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    return "cell added" if Bd.add_face(sk, rng, ctx.get("grow")) else None


@move("remove_face", "topology", "remove a fence, merging the two faces it separated", weight=0.6)
def _remove_face(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    return "fence removed" if Bd.remove_face(sk, rng, ctx.get("grow")) else None


@move("face_fill", "topology",
      "face-aware edit: relocate a dead face's fence, or dissolve a corner to gain a face",
      weight=0.8)
def _face_fill(sk: Bd.Sketch, rng, ctx: dict) -> str | None:
    info = Bd.face_fill(sk, rng, target_faces=ctx.get("min_faces"), dead=ctx.get("dead_face", 0.9))
    return None if info is None else str(info.get("move"))


# ---------------------------------------------------------------------------
# the population
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    sketch: Bd.Sketch
    origin: str
    generation: int = 0
    area: float | None = None
    config: dict | None = None
    signature: str | None = None
    heuristic: float = 0.0        # stage-1 rank score of this individual's own layout
    evals: int = 0
    opt: dict = field(default_factory=dict)


@dataclass
class EvolveResult:
    n: int
    best: Individual | None
    ledger: R.Ledger
    evaluations: int
    generations: int
    history: list[dict]
    seconds: float
    move_stats: dict

    def to_json(self) -> dict:
        return {"n": self.n, "best_area": self.best.area if self.best else None,
                "best_origin": self.best.origin if self.best else None,
                "evaluations": self.evaluations, "generations": self.generations,
                "seconds": round(self.seconds, 1), "moves": self.move_stats,
                "ledger": self.ledger.to_json(), "history": self.history}


class MoveSet:
    """The enabled mutation moves and their weights."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = {k: m.weight for k, m in MOVES.items()}
        for k, v in (weights or {}).items():
            if k not in MOVES:
                raise ValueError(f"unknown move {k!r}")
            self.weights[k] = float(v)

    @staticmethod
    def from_flags(off: list[str] | None = None, weights: list[str] | None = None,
                   kinds: list[str] | None = None) -> "MoveSet":
        w = {}
        for spec in weights or []:
            k, _, v = spec.partition("=")
            w[k.strip()] = float(v)
        for k in off or []:
            if k not in MOVES:
                raise ValueError(f"unknown move {k!r}")
            w[k] = 0.0
        if kinds:
            for k, m in MOVES.items():
                if m.kind not in kinds:
                    w.setdefault(k, 0.0)
        return MoveSet(w)

    def enabled(self) -> list[str]:
        return [k for k in MOVES if self.weights.get(k, 0.0) > 0]

    def pick(self, rng) -> str:
        ks = self.enabled()
        if not ks:
            raise ValueError("every mutation move is switched off")
        w = np.array([self.weights[k] for k in ks], dtype=float)
        return str(rng.choice(ks, p=w / w.sum()))


class NetEvolver:
    """Evolutionary search over the geometry of one net, with local topology moves."""

    def __init__(self, n: int, seed_sketch: Bd.Sketch, *, ruleset: R.RuleSet | None = None,
                 hset: H.HeuristicSet | None = None, moves: MoveSet | None = None,
                 cfg: O.OptConfig | None = None, ctx: dict | None = None, device: str = "cpu",
                 rng=None, pool=None, pop: int = 8, elite: int = 4, log=None):
        self.n = n
        self.seed_sketch = seed_sketch
        self.rules = ruleset or R.RuleSet()
        self.hset = hset or H.HeuristicSet()
        self.moves = moves or MoveSet()
        self.cfg = cfg or O.OptConfig()
        self.ctx = dict(ctx or {})
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.pool = pool
        self.pop_size = pop
        self.elite_size = elite
        self.log = log or (lambda rec: None)
        self.ledger = R.Ledger()
        self.population: list[Individual] = []
        self.best: Individual | None = None
        self.history: list[dict] = []
        self.move_stats: dict[str, dict] = {}
        self.evaluations = 0

    # -- helpers -------------------------------------------------------------
    def _candidate(self, sk: Bd.Sketch, origin: str) -> R.NetCandidate | None:
        try:
            topo = sk.topology()
        except (G.InvalidTopology, Bd.BuildError) as exc:
            self.ledger.note_error(exc, origin)
            return None
        cand = R.NetCandidate(n=self.n, topo=topo, coords=sk.coords(), family="evolved",
                              origin=origin)
        rej = self.rules.check(cand, self.ctx)
        self.ledger.note(rej, origin)
        if rej is not None:
            self.log({"event": "rejected", "n": self.n, "origin": origin, "rule": rej.rule,
                      "number": rej.number, "reason": rej.reason})
            return None
        return cand

    def _signature(self, topo: G.Topology) -> str | None:
        try:
            return Mn.canonical_form(topo).signature
        except Exception:  # noqa: BLE001 -- a signature is a convenience, not a requirement
            return None

    def evaluate(self, ind: Individual) -> Individual:
        """Local optimizer + referee on one individual. Sets area/config in place."""
        cand = self._candidate(ind.sketch, ind.origin)
        if cand is None:
            ind.area = None
            return ind
        ind.signature = self._signature(cand.topo)
        seed = int(self.rng.integers(2 ** 31))
        t0 = time.time()
        try:
            res = O.optimize_topology(cand.topo, cand.coords, self.n, self.cfg,
                                      device=self.device, seed=seed, pool=self.pool)
        except Exception as exc:  # noqa: BLE001 -- a failed optimization is just an unfit individual
            self.ledger.note_error(exc, f"optimize:{ind.origin}")
            ind.area = None
            return ind
        self.evaluations += 1
        ind.evals += 1
        ind.opt = {"adam_areas": res.get("adam_areas"), "polished": res.get("polished_areas"),
                   "seconds": round(time.time() - t0, 2)}
        best = res.get("best")
        if best is None:
            ind.area = None
            self.log({"event": "no_verified", "n": self.n, "origin": ind.origin,
                      "rejected": res.get("rejected", [])[:2]})
            return ind
        ind.area = float(best["area"])
        ind.config = best["config"]
        # keep the verified geometry: the next generation mutates the optimized layout
        try:
            topo2, X2 = G.config_from_json(best["config"])
            ind.sketch = Bd.Sketch.from_config(topo2, X2)
            ind.heuristic = self.hset.score(
                R.NetCandidate(n=self.n, topo=topo2, coords=X2, family="evolved",
                               origin=ind.origin, exact=True), self.ctx).total
        except Exception:  # noqa: BLE001 -- fall back to the pre-optimization sketch
            pass
        if self.best is None or ind.area > (self.best.area or -1):
            self.best = ind
            self.log({"event": "net_best", "n": self.n, "area": ind.area, "origin": ind.origin})
        return ind

    # -- the loop ------------------------------------------------------------
    def apply_move(self, sk: Bd.Sketch, name: str) -> str | None:
        """Apply one move in place and restore the fence count to n.

        Several topology moves (add_face, remove_face, face_fill) deliberately
        change the number of fences; the layout is only a candidate again once
        `fit_count` has grown or pruned it back to exactly n, so that happens here
        rather than leaving rule 1 to reject the result.
        """
        snap = sk.copy()
        st = self.move_stats.setdefault(name, {"tried": 0, "built": 0, "verified": 0, "wins": 0})
        st["tried"] += 1
        try:
            detail = MOVES[name].fn(sk, self.rng, self.ctx)
        except (Bd.BuildError, G.InvalidTopology, ValueError) as exc:
            self.ledger.note_error(exc, f"move:{name}")
            detail = None
        if detail is None:
            _restore(sk, snap)
            return None
        if sk.n != self.n:
            grow = {**Bd.DEFAULT_GROW, **(self.ctx.get("grow") or {})}
            try:
                Bd.fit_count(sk, self.n, self.rng, grow, 1)
            except (Bd.BuildError, G.InvalidTopology) as exc:
                self.ledger.note_error(exc, f"refit:{name}")
                _restore(sk, snap)
                return None
        st["built"] += 1
        return detail

    def seed_population(self) -> None:
        self.population = [Individual(self.seed_sketch.copy(), "seed", 0)]
        for k in range(self.pop_size - 1):
            sk = self.seed_sketch.copy()
            name = self.moves.pick(self.rng) if self.moves.enabled() else None
            detail = self.apply_move(sk, name) if name else None
            self.population.append(
                Individual(sk, f"seed+{name}({detail})#{k}" if detail else f"seed#{k}", 0))

    def mutate(self, parent: Individual, generation: int) -> Individual | None:
        sk = parent.sketch.copy()
        name = self.moves.pick(self.rng)
        detail = self.apply_move(sk, name)
        if detail is None:
            return None
        return Individual(sk, f"{name}({detail})", generation)

    def run(self, max_evals: int = 40, generations: int = 6,
            deadline: float | None = None) -> EvolveResult:
        t0 = time.time()
        if not self.population:
            self.seed_population()
        gen = 0
        for ind in self.population:
            if self.evaluations >= max_evals or (deadline and time.time() > deadline):
                break
            self.evaluate(ind)
        self._record(gen)
        while gen < generations and self.evaluations < max_evals:
            if deadline and time.time() > deadline:
                break
            gen += 1
            parents = self._select()
            if not parents:
                break
            children = []
            # a move is allowed to decline (it returns None when it does not apply to this
            # layout), so the attempts are counted, not just the successes -- otherwise a
            # net where every enabled move declines would spin here forever
            attempts = 0
            max_attempts = max(8, 6 * self.pop_size)
            while (len(children) < self.pop_size
                   and self.evaluations + len(children) < max_evals
                   and attempts < max_attempts):
                attempts += 1
                if deadline and time.time() > deadline:
                    break
                p = parents[int(self.rng.integers(len(parents)))]
                child = self.mutate(p, gen)
                if child is not None:
                    children.append(child)
            if not children:
                self.log({"event": "no_children", "n": self.n, "generation": gen,
                          "attempts": attempts})
                break
            for c in children:
                if self.evaluations >= max_evals or (deadline and time.time() > deadline):
                    break
                before = self.best.area if self.best else None
                self.evaluate(c)
                nm = c.origin.split("(")[0]
                st = self.move_stats.setdefault(nm, {"tried": 0, "built": 0, "verified": 0, "wins": 0})
                if c.area is not None:
                    st["verified"] += 1
                    if before is None or c.area > before:
                        st["wins"] += 1
            self.population = parents + [c for c in children if c.area is not None]
            self._record(gen)
        return EvolveResult(n=self.n, best=self.best, ledger=self.ledger,
                            evaluations=self.evaluations, generations=gen, history=self.history,
                            seconds=time.time() - t0, move_stats=self.move_stats)

    def _select(self) -> list[Individual]:
        """Best individuals, at most one per topology signature (diversity).

        Verified area decides. Exact ties are common (every grid-of-unit-squares answer
        at an n is the same number), and there the stage-1 heuristic score breaks the
        tie, so the survivor is the layout the heuristics think has somewhere to go.
        """
        alive = [i for i in self.population if i.area is not None]
        alive.sort(key=lambda i: (-i.area, -i.heuristic))
        out, seen = [], set()
        for i in alive:
            if i.signature is not None and i.signature in seen:
                continue
            if i.signature is not None:
                seen.add(i.signature)
            out.append(i)
            if len(out) >= self.elite_size:
                break
        return out or alive[: self.elite_size]

    def _record(self, gen: int) -> None:
        alive = [i for i in self.population if i.area is not None]
        self.history.append({
            "generation": gen, "evaluations": self.evaluations,
            "alive": len(alive), "best": self.best.area if self.best else None,
            "mean": round(float(np.mean([i.area for i in alive])), 6) if alive else None,
            "signatures": len({i.signature for i in alive if i.signature}),
        })


def describe_moves() -> str:
    out = []
    for kind in ("geometry", "topology"):
        out.append(f"  {kind}:")
        for m in MOVES.values():
            if m.kind == kind:
                out.append(f"    {m.id:<16} w={m.weight:<4} {m.summary}")
    return "\n".join(out)

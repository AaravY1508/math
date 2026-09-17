"""Stage 2: the mutation moves, the rule gate in front of the optimizer, and the loop."""
import numpy as np
import pytest

from fence import builder as Bd
from fence import geometry as G
from fence import nets as N
from fence import netevolve as NE
from fence import optimize as O
from fence import rules as R

RNG = lambda s=0: np.random.default_rng(s)      # noqa: E731


def square_cluster(c=4):
    cells = N.polyomino_shapes(c, limit=1)[0]
    return N.polyomino_sketch(cells)


# -- the move registry -------------------------------------------------------

def test_moves_cover_both_kinds_and_the_edits_the_brief_asks_for():
    kinds = {m.kind for m in NE.MOVES.values()}
    assert kinds == {"geometry", "topology"}
    for needed in ("join_to_t", "t_to_join", "slide_t", "slide_t_past", "add_face", "remove_face"):
        assert needed in NE.MOVES


def test_moves_can_be_switched_off_one_at_a_time():
    for mid in NE.MOVES:
        ms = NE.MoveSet.from_flags([mid])
        assert mid not in ms.enabled()
    with pytest.raises(ValueError):
        NE.MoveSet.from_flags(["not_a_move"])


def test_move_kinds_filter():
    ms = NE.MoveSet.from_flags(kinds=["geometry"])
    assert all(NE.MOVES[k].kind == "geometry" for k in ms.enabled())
    assert ms.enabled()


def test_pick_respects_zero_weights():
    ms = NE.MoveSet.from_flags(weights=[f"{k}=0" for k in NE.MOVES if k != "jitter"])
    rng = RNG(1)
    assert {ms.pick(rng) for _ in range(20)} == {"jitter"}
    with pytest.raises(ValueError):
        NE.MoveSet.from_flags(list(NE.MOVES)).pick(rng)


# -- individual moves --------------------------------------------------------

@pytest.mark.parametrize("name", sorted(NE.MOVES))
def test_every_move_either_declines_or_leaves_a_legal_layout(name):
    """A move may fail (returning None), but it must never leave an illegal sketch behind."""
    rng = RNG(7)
    tried = applied = 0
    for c in (2, 4, 6):
        sk = square_cluster(c)
        for _ in range(6):
            trial = sk.copy()
            tried += 1
            try:
                detail = NE.MOVES[name].fn(trial, rng, {"min_faces": c + 1})
            except (Bd.BuildError, G.InvalidTopology, ValueError):
                continue
            if detail is None:
                continue
            applied += 1
            trial.topology()          # raises InvalidTopology if the move broke the embedding
    assert tried > 0


def test_geometry_moves_keep_the_topology():
    sk = square_cluster(4)
    before = sk.topology()
    rng = RNG(3)
    for name in [k for k, m in NE.MOVES.items() if m.kind == "geometry" and k != "slide_t"]:
        trial = sk.copy()
        if NE.MOVES[name].fn(trial, rng, {}) is None:
            continue
        after = trial.topology()
        assert after.n == before.n
        assert len(after.faces) == len(before.faces), name
        assert set(map(tuple, after.fences.tolist())) == set(map(tuple, before.fences.tolist())), name


def test_join_to_t_changes_the_t_junction_count():
    sk = square_cluster(4)
    rng = RNG(11)
    for _ in range(20):
        trial = sk.copy()
        before = len(trial.pins)
        if NE.MOVES["join_to_t"].fn(trial, rng, {}) is None:
            continue
        assert len(trial.pins) != before
        trial.topology()
        return
    pytest.skip("join_to_t did not apply on this cluster")


def test_t_to_join_removes_a_pin():
    sk = square_cluster(4)
    rng = RNG(5)
    for _ in range(40):
        trial = sk.copy()
        if NE.MOVES["join_to_t"].fn(trial, rng, {}) is None or not trial.pins:
            continue
        before = len(trial.pins)
        if NE.MOVES["t_to_join"].fn(trial, rng, {}) is None:
            continue
        assert len(trial.pins) == before - 1
        trial.topology()
        return
    pytest.skip("no T-junction to merge back on this cluster")


def test_slide_t_keeps_the_pin_on_its_host():
    sk = square_cluster(4)
    rng = RNG(9)
    for _ in range(40):
        trial = sk.copy()
        if NE.MOVES["join_to_t"].fn(trial, rng, {}) is None or not trial.pins:
            continue
        if NE.MOVES["slide_t"].fn(trial, rng, {}) is None:
            continue
        X = trial.coords()
        for v, h in trial.pins.items():
            a, b = trial.fences[h]
            d = G.point_segment_distance(X[v], X[a], X[b])
            assert d < 1e-9, "a slid T-junction left its host"
        return
    pytest.skip("slide_t did not apply on this cluster")


def test_slide_t_past_reorders_a_host_chain():
    """Two pins on one host: exchanging their order is a different net, not just a move."""
    sk = square_cluster(6)
    rng = RNG(13)
    for _ in range(80):
        trial = sk.copy()
        for _ in range(3):
            NE.MOVES["join_to_t"].fn(trial, rng, {})
        try:
            chains = trial.topology().chains
        except G.InvalidTopology:
            continue
        if not any(len(ch) >= 4 for ch in chains):
            continue
        before = [list(ch) for ch in chains]
        if NE.MOVES["slide_t_past"].fn(trial, rng, {}) is None:
            continue
        after = [list(ch) for ch in trial.topology().chains]
        assert after != before
        return
    pytest.skip("no host carried two T-junctions in this sample")


def test_reproject_pins_puts_pins_back_on_their_hosts():
    sk = square_cluster(4)
    rng = RNG(2)
    for _ in range(40):
        if NE.MOVES["join_to_t"].fn(sk, rng, {}) is not None and sk.pins:
            break
    X = sk.coords()
    X[:, 0] += 0.01 * np.arange(len(X))
    sk.P = X.tolist()
    sk._dirty()
    NE._reproject_pins(sk)
    X = sk.coords()
    for v, h in sk.pins.items():
        a, b = sk.fences[h]
        assert G.point_segment_distance(X[v], X[a], X[b]) < 1e-9


# -- the evolver -------------------------------------------------------------

def _evolver(n, sk, **kw):
    cfg = O.OptConfig(batch=32, steps=40, topk=1, polish_iters=80, verify_top=1)
    kw.setdefault("cfg", cfg)
    kw.setdefault("device", "cpu")
    kw.setdefault("rng", RNG(4))
    kw.setdefault("pop", 3)
    kw.setdefault("elite", 2)
    return NE.NetEvolver(n, sk, **kw)


def test_apply_move_restores_the_fence_count():
    sk = square_cluster(4)
    ev = _evolver(sk.n, sk)
    rng = RNG(6)
    for _ in range(30):
        trial = sk.copy()
        name = str(rng.choice(sorted(NE.MOVES)))
        if ev.apply_move(trial, name) is None:
            continue
        assert trial.n == sk.n, f"{name} left {trial.n} fences"


def test_the_rule_gate_runs_before_the_optimizer():
    """A candidate that breaks a rule is rejected and logged without being optimized."""
    sk = square_cluster(4)
    ev = _evolver(sk.n, sk, ruleset=R.RuleSet(ctx={"min_faces": 99}))
    ind = NE.Individual(sk.copy(), "seed")
    ev.evaluate(ind)
    assert ind.area is None
    assert ev.evaluations == 0
    assert ev.ledger.counts["record_faces"] == 1


def test_rejections_are_logged_with_the_rule_id():
    seen = []
    sk = square_cluster(4)
    ev = _evolver(sk.n, sk, ruleset=R.RuleSet(ctx={"min_faces": 99}), log=seen.append)
    ev.evaluate(NE.Individual(sk.copy(), "seed"))
    assert seen and seen[0]["rule"] == "record_faces" and seen[0]["number"] == 14


@pytest.mark.slow
def test_run_produces_a_verified_result_and_a_history():
    sk = square_cluster(4)            # 12 fences, 4 unit cells: area 4 is reachable
    ev = _evolver(12, sk, ctx={"min_faces": 4})
    res = ev.run(max_evals=4, generations=2)
    assert res.evaluations > 0
    assert res.best is not None and res.best.area is not None
    assert res.best.area > 3.0
    assert res.history and res.history[-1]["evaluations"] == res.evaluations
    assert res.to_json()["best_area"] == res.best.area
    # the verified geometry is carried forward, so the next generation mutates the optimum
    L = np.linalg.norm(
        res.best.sketch.coords()[np.array(res.best.sketch.fences)[:, 1]]
        - res.best.sketch.coords()[np.array(res.best.sketch.fences)[:, 0]], axis=1)
    assert np.allclose(L, 1.0, atol=1e-6)


@pytest.mark.slow
def test_run_respects_the_evaluation_budget():
    sk = square_cluster(4)
    ev = _evolver(12, sk, ctx={"min_faces": 4}, pop=4)
    res = ev.run(max_evals=3, generations=5)
    assert res.evaluations <= 3


def test_select_keeps_one_individual_per_topology():
    sk = square_cluster(4)
    ev = _evolver(12, sk)
    a = NE.Individual(sk.copy(), "a"); a.area, a.signature = 4.0, "sigA"
    b = NE.Individual(sk.copy(), "b"); b.area, b.signature = 3.9, "sigA"
    c = NE.Individual(sk.copy(), "c"); c.area, c.signature = 3.5, "sigB"
    ev.population = [a, b, c]
    sel = ev._select()
    assert [i.origin for i in sel] == ["a", "c"]


def test_select_breaks_an_exact_area_tie_on_the_heuristic_score():
    sk = square_cluster(4)
    ev = _evolver(12, sk, elite=1)
    a = NE.Individual(sk.copy(), "plain"); a.area, a.signature, a.heuristic = 4.0, "sigA", 0.60
    b = NE.Individual(sk.copy(), "promising"); b.area, b.signature, b.heuristic = 4.0, "sigB", 0.85
    ev.population = [a, b]
    assert ev._select()[0].origin == "promising"

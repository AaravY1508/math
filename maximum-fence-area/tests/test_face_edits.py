"""Face-aware edits (fence.builder.face_fill) and the face-aware mutation operator.

Fixtures in data/search_records.json are verified configurations the search found (all-time
records of the minima library on 2026-09-12) chosen for their edge cases:
    13  a dead face (0.048) that is the eye of a 3-fence pinwheel: no move applies
    18  four under-filled faces (0.78-0.84) that only touch through pin-hosting fences
    20  7 faces where beating the record needs 8, no dead face: dissolve
    23  a degenerate face (~1e-9): relocate
"""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from fence import builder as Bd
from fence import evolve as E
from fence import geometry as G
from fence import minima as Mn
from tests import configs as C
from tests.test_minima import verified

RECORDS = json.loads((Path(__file__).parent / "data" / "search_records.json").read_text())
BASE = E.load_baselines()


def sketch_of(n: int) -> Bd.Sketch:
    return Bd.Sketch.from_json(E._config_to_sketch(RECORDS[str(n)]))


def signature(sk: Bd.Sketch) -> str:
    return Mn.canonical_form(sk.topology()).signature


def joints(sk: Bd.Sketch) -> int:
    return sk.topology().n_vertices - len(sk.pins)


def not_parent(sk0: Bd.Sketch):
    sig0 = signature(sk0)
    return lambda t: signature(t) == sig0


def test_face_count_identity_on_fixtures():
    for rec in RECORDS.values():
        sk = sketch_of(rec["n"])
        assert len(sk.topology().faces) == sk.n + 1 - joints(sk)


def test_face_sides_and_removal_cost():
    cfg, topo, X = verified(C.SPLIT_SQUARE, 5)
    sk = Bd.Sketch.from_config(topo, X)
    faces, areas, left = Bd.face_sides(sk)
    assert sorted(np.round(areas, 12)) == [0.5, 0.5]
    a, b = sk.fences[4]                                   # the chord, pinned on bottom and top
    assert {left[(a, b)], left[(b, a)]} == {0, 1}
    assert Bd.removal_cost((0, 1), areas) == 0.0          # two halves merge into one full face
    assert Bd.removal_cost((0, -1), areas) == 0.5         # a face opened to the outside is lost
    assert Bd.removal_cost((1, 1), areas) is None         # same face both sides: would disconnect


@pytest.mark.parametrize("n,move,dF", [(23, "relocate", 0), (18, "relocate", 0), (20, "dissolve", 1)])
def test_face_fill_moves_keep_the_layout_legal(n, move, dF):
    sk0 = sketch_of(n)
    F0, J0 = len(sk0.topology().faces), joints(sk0)
    for seed in range(5):
        sk = sk0.copy()
        info = Bd.face_fill(sk, np.random.default_rng(seed), E.target_faces(BASE[n]), reject=not_parent(sk0))
        assert info is not None and info["move"] == move
        topo = sk.topology()                              # legal embedding (raises otherwise)
        assert sk.n == n
        assert len(topo.faces) == F0 + dF and joints(sk) == J0 - dF   # F = n + 1 - J
        X = sk.coords()
        assert G.face_areas(X, topo.faces).min() >= Bd.MIN_FACE
        L = np.linalg.norm(X[topo.fences[:, 1]] - X[topo.fences[:, 0]], axis=1)
        assert L.min() >= Bd.FENCE_LEN[0] and L.max() <= Bd.FENCE_LEN[1]
        assert signature(sk) != signature(sk0)


def test_degenerate_face_is_relocated():
    sk0 = sketch_of(23)
    assert min(float(a) for a in RECORDS["23"]["face_areas"]) < 1e-6
    sk = sk0.copy()
    Bd.face_fill(sk, np.random.default_rng(0), E.target_faces(BASE[23]), reject=not_parent(sk0))
    assert G.face_areas(sk.coords(), sk.topology().faces).min() >= Bd.MIN_FACE


def test_shared_fence_between_underfilled_faces_goes_first():
    # both halves of a split square are under-filled and the chord between them is the only
    # removable fence: merging them costs nothing, and it is considered once, not per face
    cfg, topo, X = verified(C.SPLIT_SQUARE, 5)
    sk = Bd.Sketch.from_config(topo, X)
    info = Bd.face_fill(sk, np.random.default_rng(0))
    assert info == {**info, "move": "relocate", "cost": 0.0, "shared": True}
    assert sk.n == 5 and len(sk.topology().faces) == 2


def test_underfilled_faces_touching_through_pin_hosts_are_not_shared():
    sk = sketch_of(18)
    info = Bd.face_fill(sk, np.random.default_rng(1), E.target_faces(BASE[18]))
    assert info["move"] == "relocate" and info["shared"] is False and info["face_area"] < 0.8


def test_pinwheel_eye_takes_no_edit_and_is_left_untouched():
    sk = sketch_of(13)
    before = json.dumps(sk.to_json())
    assert Bd.face_fill(sk, np.random.default_rng(0), E.target_faces(BASE[13])) is None
    assert json.dumps(sk.to_json()) == before


def test_full_layout_at_target_takes_no_edit():
    cfg, topo, X = verified(C.GRID_2x2, 12)
    sk = Bd.Sketch.from_config(topo, X)
    assert Bd.face_fill(sk, np.random.default_rng(0), target_faces=4) is None
    info = Bd.face_fill(sk, np.random.default_rng(0), target_faces=5)
    assert info["move"] == "dissolve" and len(sk.topology().faces) == 5 and sk.n == 12


# -- the mutation operator ------------------------------------------------------------------

OLD_OPS = [("jitter", 0.25), ("add_edit", 0.12), ("remove_edit", 0.05), ("seed", 0.08), ("symmetry", 0.08),
           ("elite", 0.36), ("fresh", 0.06)]


def test_operator_weights_are_unchanged_without_the_face_op():
    no_elite = [(k, 0.0 if k == "elite" else w) for k, w in OLD_OPS]
    for fw in (0.0, 0.25):
        m = E.RandomMutator(random.Random(0), face_weight=fw)
        assert m.op_weights(True, False) == OLD_OPS
        assert m.op_weights(False, False) == no_elite
    assert E.RandomMutator(random.Random(0)).op_weights(True, True) == OLD_OPS   # default: off


def test_face_op_takes_its_share_from_the_regrowing_operators():
    w = dict(E.RandomMutator(random.Random(0), face_weight=0.25).op_weights(True, True))
    assert w["face"] == 0.25 and w["elite"] == 0.36
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["jitter"] == pytest.approx(0.25 * (0.64 - 0.25) / 0.64)


def _elite_entry(n):
    rec = RECORDS[str(n)]
    return {"sketch": E._config_to_sketch(rec), "note": f"n{n}:{rec['area']:.4f}", "proposer": f"lineage{n}",
            "n": n, "area": rec["area"]}


def test_face_children_edit_an_elite_of_the_same_n():
    n = 23
    mut = E.RandomMutator(random.Random(4), face_weight=0.9)
    parent = E.Program.from_genome(E.seed_genomes(n)[2])
    ctx = {"n": n, "elite_sketches": [_elite_entry(20), _elite_entry(n)], "face_sources": [_elite_entry(n)],
           "target_faces": E.target_faces(BASE[n])}
    sig_elite = signature(sketch_of(n))
    faces = 0
    for i in range(30):                                 # face op picked with probability 0.9 / 1.26
        child = mut.mutate(parent, ctx)
        if not child.origin.startswith("random:face:"):
            continue
        faces += 1
        assert child.proposer == f"lineage{n}" and child.genome["face_edit"]["move"] == "relocate"
        sk = child.propose(n, i)
        assert sk.topology().n == n and signature(sk) != sig_elite
    assert faces >= 12


def test_face_op_falls_back_to_the_elite_op_when_no_elite_takes_an_edit():
    n = 13
    mut = E.RandomMutator(random.Random(1), face_weight=0.9)
    parent = E.Program.from_genome(E.seed_genomes(n)[2])
    ctx = {"n": n, "elite_sketches": [_elite_entry(n)], "face_sources": [_elite_entry(n)],
           "target_faces": E.target_faces(BASE[n])}
    tags = {mut.mutate(parent, ctx).origin.split(":")[1] for _ in range(20)}
    assert "elite-fb" in tags and "face" not in tags


def test_worker_passes_the_face_weight(tmp_path):
    from tests.test_evolve import make_worker
    w = make_worker(tmp_path, "20", minima=False, extra=["--face-weight", "0.3"])
    assert w.mutator.face_weight == 0.3
    job_ctx_target = E.target_faces(w.state[20].baseline)
    assert job_ctx_target == 8


def test_face_children_of_a_blocked_lineage_start_their_own():
    # n=20's grid-trivial answer: its lineage is blocked at n for the other operators, but a
    # dissolve adds a face, so the face op may edit it; the child must not inherit the block
    n = 20
    mut = E.RandomMutator(random.Random(2), face_weight=0.9)
    parent = E.Program.from_genome(E.seed_genomes(n)[2])
    src = {**_elite_entry(n), "blocked": True}
    ctx = {"n": n, "elite_sketches": [], "face_sources": [src], "target_faces": E.target_faces(BASE[n])}
    children = [mut.mutate(parent, ctx) for _ in range(10)]
    faces = [c for c in children if c.origin.startswith("random:face:dissolve")]
    assert faces and all(c.proposer == E.proposer_id(c.source) != src["proposer"] for c in faces)
    assert not any(c.origin.startswith("random:elite") for c in children)   # the elite op has no source


def test_worker_offers_blocked_elites_to_the_face_op_only(tmp_path):
    from tests.test_evolve import make_worker
    w = make_worker(tmp_path, "20", extra=["--face-weight", "0.3"])
    grid = E.Program.from_genome(E.seed_genomes(20)[0])
    w.trivial[20] = {grid.proposer}
    w.records[20] = 7.0                                   # the grid answer: its lineage is blocked
    e = E.Elite(grid, 0.98, 7.0, {"faces": 7, "tjunctions": 0, "sym": 1},
                {**RECORDS["20"], "verification": {"total_area": "7.0"}})
    w.state[20].islands[0].try_insert(e)
    allsk = w._elite_sketches(20, with_blocked=True)
    assert [s["blocked"] for s in allsk] == [True] and w._elite_sketches(20) == []


def test_face_op_without_any_usable_elite_falls_back_to_another_operator():
    # only a blocked elite on which no face edit applies, and no elite for the elite op
    n = 13
    mut = E.RandomMutator(random.Random(5), face_weight=0.25)
    parent = E.Program.from_genome(E.seed_genomes(n)[2])
    ctx = {"n": n, "elite_sketches": [], "face_sources": [{**_elite_entry(n), "blocked": True}],
           "target_faces": E.target_faces(BASE[n])}
    tags = {mut.mutate(parent, ctx).origin.split(":")[1] for _ in range(40)}
    assert not {"face", "elite", "elite-fb"} & tags

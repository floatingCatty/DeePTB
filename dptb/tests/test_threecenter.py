"""Unit tests for the three-center factorized term
``H^(3)_{AB} = sum_{C != A,B} P_{AC} D_C P_{CB}`` (``dptb/nn/threecenter.py``).

Two cutoffs are exercised: target blocks ``(A,B)`` come from the *basis* bond list (``r_max``,
built via ``AtomicData.from_points``), while the projector reach ``A-C``/``C-B`` comes from the
*environment* list (``er_max``). Orbital/projector indexing comes from ``OrbitalMapper``. The tests
prove the batched assembly matches a naive per-triangle reference, is Hermitian and rotation
covariant, honours both cutoffs, and handles heterogeneous species.
"""

import torch
import pytest

from dptb.data import AtomicData, AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.tensor_product import wigner_D
from dptb.nn.threecenter import ThreeCenterFactorized, EDGE_THREECENTER_KEY, NODE_THREECENTER_KEY

DTYPE = torch.float64
BASIS = {"C": ["2s", "2p"]}          # norb = 4
PROJ = {"C": ["1s", "2s", "1p"]}     # nproj = 5


def _data(pos, r_max, er_max, idp, Z=None):
    if Z is None:
        Z = torch.full((pos.shape[0],), 6, dtype=torch.long)  # all carbon
    d = AtomicData.from_points(pos=pos, r_max=r_max, er_max=er_max, pbc=False, atomic_numbers=Z)
    dd = AtomicData.to_AtomicDataDict(d)
    if dd[AtomicDataDict.EDGE_INDEX_KEY].shape[1] > 0:
        idp(dd)
    else:  # DeePTB's bond mapper can't map an edgeless graph; set node types directly
        dd[AtomicDataDict.ATOM_TYPE_KEY] = idp.transform(dd[AtomicDataDict.ATOMIC_NUMBERS_KEY])
    return dd


def _module(er_max=100.0, coupling_mode="block_diag", seed=0, basis=BASIS, proj=PROJ):
    torch.manual_seed(seed)
    idp = OrbitalMapper(basis, method="e3tb")
    m = ThreeCenterFactorized(basis, proj, idp=idp, er_max=er_max, coupling_mode=coupling_mode,
                              dtype=DTYPE).to(DTYPE)
    return m, idp


def _orb_rotation(R, l_list):
    from e3nn.o3 import matrix_to_angles
    P = torch.zeros(3, 3, dtype=R.dtype)
    P[0, 1] = P[1, 2] = P[2, 0] = 1.0
    a, b, c = matrix_to_angles((P @ R @ P.t()).unsqueeze(0))
    U = torch.zeros(sum(2 * l + 1 for l in l_list), sum(2 * l + 1 for l in l_list), dtype=R.dtype)
    off = 0
    for l in l_list:
        U[off:off + 2 * l + 1, off:off + 2 * l + 1] = wigner_D(int(l), a, b, c)[0].to(R.dtype)
        off += 2 * l + 1
    return U


def test_one_atom_onsite_sanity():
    m, idp = _module()
    dd = _data(torch.zeros(1, 3, dtype=DTYPE), 5.0, 5.0, idp)
    assert torch.count_nonzero(m(dd)[EDGE_THREECENTER_KEY]) == 0


def test_two_atoms_no_third_center():
    m, idp = _module()
    dd = _data(torch.tensor([[0., 0, 0], [1.4, 0, 0]], dtype=DTYPE), 5.0, 5.0, idp)
    assert m(dd)[EDGE_THREECENTER_KEY].abs().max().item() == 0.0


def test_three_atoms_correction_and_reference():
    m, idp = _module(seed=1)
    pos = torch.tensor([[0., 0, 0], [1.4, 0, 0], [0.6, 1.1, 0]], dtype=DTYPE)
    dd = _data(pos, 5.0, 5.0, idp)
    out = m(dd)[EDGE_THREECENTER_KEY]
    assert out.abs().max().item() > 1e-6
    assert torch.allclose(out, m.reference(dd), atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("n,seed", [(4, 2), (6, 3), (9, 4)])
def test_batched_equals_reference(n, seed):
    m, idp = _module(seed=seed)
    torch.manual_seed(seed)
    dd = _data(torch.randn(n, 3, dtype=DTYPE) * 2.0, 100.0, 100.0, idp)
    assert torch.allclose(m(dd)[EDGE_THREECENTER_KEY], m.reference(dd), atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("mode", ["diag", "block_diag", "dense"])
def test_hermiticity(mode):
    m, idp = _module(coupling_mode=mode, seed=5)
    torch.manual_seed(5)
    dd = _data(torch.randn(6, 3, dtype=DTYPE) * 2.0, 100.0, 100.0, idp)
    out = m(dd)[EDGE_THREECENTER_KEY]
    ei = dd[AtomicDataDict.EDGE_INDEX_KEY]
    slot = {(int(ei[0, e]), int(ei[1, e])): e for e in range(ei.shape[1])}
    err = max((out[e] - out[slot[(b, a)]].transpose(-1, -2)).abs().max().item()
              for (a, b), e in slot.items())
    assert err < 1e-10


@pytest.mark.parametrize("mode", ["diag", "block_diag"])
def test_rotation_covariance(mode):
    from e3nn.o3 import rand_matrix
    m, idp = _module(coupling_mode=mode, seed=7)
    torch.manual_seed(7)
    pos = torch.randn(6, 3, dtype=DTYPE) * 2.0
    dd = _data(pos, 100.0, 100.0, idp)
    out = m(dd)[EDGE_THREECENTER_KEY]
    R = rand_matrix(1)[0].to(DTYPE)
    U = _orb_rotation(R, m.orb_l)
    dd_rot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dd.items()}
    dd_rot[AtomicDataDict.POSITIONS_KEY] = pos @ R.t()
    for k in (AtomicDataDict.EDGE_VECTORS_KEY, AtomicDataDict.EDGE_LENGTH_KEY,
              AtomicDataDict.ENV_VECTORS_KEY, AtomicDataDict.ENV_LENGTH_KEY):
        dd_rot.pop(k, None)
    out_rot = m(dd_rot)[EDGE_THREECENTER_KEY]
    err = max((out_rot[e] - U @ out[e] @ U.t()).abs().max().item() for e in range(out.shape[0]))
    assert err < 1e-5


def test_target_blocks_follow_basis_cutoff():
    # A and B share a common third center C within er_max. Whether (A,B) receives a three-center
    # block must be decided by the *basis* cutoff r_max, not by er_max.
    m, idp = _module(er_max=5.0, seed=3)
    A, B, C = [0., 0, 0], [2.4, 0, 0], [1.2, 0.6, 0]
    pos = torch.tensor([A, B, C], dtype=DTYPE)

    # r_max large enough that A-B (dist 2.4) is a basis bond -> (A,B) gets a correction.
    dd_in = _data(pos, 3.0, 5.0, idp)
    out_in = m(dd_in)[EDGE_THREECENTER_KEY]
    ei = dd_in[AtomicDataDict.EDGE_INDEX_KEY]
    ab = [e for e in range(ei.shape[1]) if {int(ei[0, e]), int(ei[1, e])} == {0, 1}]
    assert ab and max(out_in[e].abs().max().item() for e in ab) > 1e-6

    # r_max too small for A-B (2.4 > 2.0) -> (A,B) is not a basis bond, so it must not appear at all,
    # even though C is within er_max of both. Only A-C / B-C blocks (basis bonds) may be nonzero.
    dd_out = _data(pos, 2.0, 5.0, idp)
    ei2 = dd_out[AtomicDataDict.EDGE_INDEX_KEY]
    assert not any({int(ei2[0, e]), int(ei2[1, e])} == {0, 1} for e in range(ei2.shape[1]))


def test_env_cutoff_controls_projector_reach():
    # Moving C beyond er_max removes its three-center contribution to the A-B bond.
    m, idp = _module(er_max=3.0, seed=11)
    pos = torch.tensor([[0., 0, 0], [1.0, 0, 0], [0.4, 0.3, 0]], dtype=DTYPE)
    assert m(_data(pos, 3.0, 3.0, idp))[EDGE_THREECENTER_KEY].abs().max().item() > 1e-6
    pos_far = pos.clone(); pos_far[2] = torch.tensor([50., 50., 0], dtype=DTYPE)
    assert m(_data(pos_far, 3.0, 3.0, idp))[EDGE_THREECENTER_KEY].abs().max().item() == 0.0


def test_heterogeneous_species():
    # B has {s,p} (norb=4), N has {s,p,d} (norb=9); projectors differ too. Everything lives in the
    # full_basis (norb=9) and mask_to_basis zeros the channels a species lacks.
    hbasis = {"B": ["2s", "2p"], "N": ["2s", "2p", "3d"]}
    hproj = {"B": ["1s", "1p"], "N": ["1s", "2s", "1p"]}
    m, idp = _module(seed=2, basis=hbasis, proj=hproj)
    assert m.norb == idp.full_basis_norb == 9
    torch.manual_seed(2)
    pos = torch.randn(6, 3, dtype=DTYPE) * 2.2
    Z = torch.tensor([5, 7, 5, 7, 7, 5])  # B=5, N=7
    dd = _data(pos, 100.0, 100.0, idp, Z=Z)
    out = m(dd)[EDGE_THREECENTER_KEY]
    assert torch.allclose(out, m.reference(dd), atol=1e-10, rtol=0.0)

    ei, at = dd[AtomicDataDict.EDGE_INDEX_KEY], dd[AtomicDataDict.ATOM_TYPE_KEY].flatten()
    # Boron (type 0) has no d orbitals. out[e] rows are the center A=ei[0]'s orbitals, cols the
    # neighbour B=ei[1]'s, so a Boron endpoint must zero the corresponding d block (indices 4:9).
    for e in range(ei.shape[1]):
        if at[ei[0, e]] == 0:
            assert out[e, 4:9, :].abs().max() < 1e-12
        if at[ei[1, e]] == 0:
            assert out[e, :, 4:9].abs().max() < 1e-12
    slot = {(int(ei[0, e]), int(ei[1, e])): e for e in range(ei.shape[1])}
    err = max((out[e] - out[slot[(b, a)]].transpose(-1, -2)).abs().max().item() for (a, b), e in slot.items())
    assert err < 1e-10


def test_gradients_flow():
    m, idp = _module(seed=9)
    torch.manual_seed(9)
    dd = _data(torch.randn(5, 3, dtype=DTYPE) * 2.0, 100.0, 100.0, idp)
    m(dd)[EDGE_THREECENTER_KEY].pow(2).sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_onsite_batched_equals_reference():
    m, idp = _module(seed=3)
    torch.manual_seed(3)
    dd = _data(torch.randn(7, 3, dtype=DTYPE) * 2.0, 100.0, 100.0, idp)
    node = m(dd)[NODE_THREECENTER_KEY]
    assert node.abs().max().item() > 1e-6
    assert torch.allclose(node, m.reference_node(dd), atol=1e-10, rtol=0.0)


def test_onsite_hermiticity():
    # each onsite block H^(3)_AA = sum_C P_AC D_C P_CA is symmetric (D_C symmetric).
    m, idp = _module(seed=4)
    torch.manual_seed(4)
    dd = _data(torch.randn(6, 3, dtype=DTYPE) * 2.0, 100.0, 100.0, idp)
    node = m(dd)[NODE_THREECENTER_KEY]
    assert (node - node.transpose(-1, -2)).abs().max().item() < 1e-12


def test_to_feature_matches_official_converter():
    # to_feature must be the exact inverse of DeePTB's feature_to_block, for edge (Hermitian across
    # reflective bonds) and node (symmetric) blocks alike.
    from dptb.data.interfaces.ham_to_feature import feature_to_block
    m, idp = _module(seed=0)
    torch.manual_seed(0)
    dd = _data(torch.randn(5, 3, dtype=DTYPE) * 2.0, 100.0, 100.0, idp)
    out = m(dd)
    D, Dn = out[EDGE_THREECENTER_KEY], out[NODE_THREECENTER_KEY]
    dd2 = dict(dd)
    dd2[AtomicDataDict.EDGE_FEATURES_KEY] = m.to_feature(D)
    dd2[AtomicDataDict.NODE_FEATURES_KEY] = m.to_feature(Dn)
    dd2[AtomicDataDict.ATOMIC_NUMBERS_KEY] = torch.full((5,), 6, dtype=torch.long)
    blocks = feature_to_block(dd2, idp)
    ei, ecs = dd[AtomicDataDict.EDGE_INDEX_KEY], dd[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
    err = 0.0
    for e in range(ei.shape[1]):
        i, j = int(ei[0, e]), int(ei[1, e])
        key = f"{i}_{j}_" + "_".join(map(str, ecs[e].int().tolist()))
        if key in blocks:
            err = max(err, (blocks[key].to(DTYPE) - D[e]).abs().max().item())
    for a in range(5):  # onsite blocks
        err = max(err, (blocks[f"{a}_{a}_0_0_0"].to(DTYPE) - Dn[a]).abs().max().item())
    assert err < 1e-10


def test_to_reduced_inverts_e3hamiltonian():
    # to_reduced must be the exact inverse of E3Hamiltonian(decompose=False), so adding it pre-transform
    # and letting NNENV's transform run reproduces the block correction (to_feature).
    from dptb.nn.hamiltonian import E3Hamiltonian
    m, idp = _module(seed=0)
    torch.manual_seed(0)
    dd = _data(torch.randn(5, 3, dtype=DTYPE) * 2.0, 100.0, 100.0, idp)
    out = m(dd)
    e3 = E3Hamiltonian(idp=idp, dtype=DTYPE)
    tmp = dict(dd)
    tmp[AtomicDataDict.EDGE_FEATURES_KEY] = m.to_reduced(out[EDGE_THREECENTER_KEY])
    tmp[AtomicDataDict.NODE_FEATURES_KEY] = m.to_reduced(out[NODE_THREECENTER_KEY])
    tmp = e3(tmp)
    assert (tmp[AtomicDataDict.EDGE_FEATURES_KEY] - m.to_feature(out[EDGE_THREECENTER_KEY])).abs().max() < 1e-10
    assert (tmp[AtomicDataDict.NODE_FEATURES_KEY] - m.to_feature(out[NODE_THREECENTER_KEY])).abs().max() < 1e-10


def test_mode_channels():
    # a single trinity `mode` (replacing only2b/exclusive) toggles the channels on top of the
    # always-present two-body base; "full" also freezes the 2b+3b params.
    import pytest as _pytest
    from dptb.nn.build import build_model
    common = {"basis": {"C": ["2s", "2p"]}, "device": "cpu", "dtype": "float32", "overlap": False}

    def build(mode):
        mo = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                            "n_layers": 2, "avg_num_neighbors": 10, "mode": mode,
                            "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
              "prediction": {"method": "e3tb", "neurons": [16, 16]}}
        return build_model(None, mo, common)

    m2b = build("2b")
    assert (not m2b.embedding.use_3b) and (not m2b.embedding.use_env) and m2b.embedding.freeze == set()
    # 3b params are ALWAYS built when a three_center config is given (even in 2b mode), so the
    # checkpoint carries them for progressive 2b -> 2b+3b -> full training; the mode only gates use.
    assert m2b.embedding.three_center is not None
    assert all(p.requires_grad for p in m2b.embedding.three_center.parameters())  # built, trainable, unused

    m23 = build("2b+3b")
    assert m23.embedding.use_3b and (not m23.embedding.use_env) and m23.embedding.freeze == set()
    assert all(p.requires_grad for p in m23.embedding.two_ness.parameters())   # 2b+3b not frozen

    mfull = build("full")
    assert mfull.embedding.use_3b and mfull.embedding.use_env and mfull.embedding.freeze == {"2b", "3b"}
    assert not any(p.requires_grad for p in mfull.embedding.two_ness.parameters())        # 2b frozen
    assert not any(p.requires_grad for p in mfull.embedding.three_center.parameters())    # 3b frozen

    # progressive training relies on the state_dict being identical across modes (same param set)
    assert set(m2b.state_dict().keys()) == set(m23.state_dict().keys()) == set(mfull.state_dict().keys())

    # a pure 2b model without any three_center config is still allowed (three_center simply absent)
    common2b = dict(common)
    mo_plain = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                              "n_layers": 2, "avg_num_neighbors": 10, "mode": "2b"},
                "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    assert build_model(None, mo_plain, common2b).embedding.three_center is None

    # 3b/2b+3b/full require a three_center config
    bad = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                         "n_layers": 2, "avg_num_neighbors": 10, "mode": "full"},
           "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    with _pytest.raises(AssertionError, match="three_center"):
        build_model(None, bad, common)


def test_selective_freeze():
    # `freeze` selectively holds trainable blocks fixed, orthogonal to `mode`, for progressive training.
    import pytest as _pytest
    from dptb.nn.build import build_model
    common = {"basis": {"C": ["2s", "2p"]}, "device": "cpu", "dtype": "float32", "overlap": False}

    def build(mode, freeze):
        mo = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                            "n_layers": 2, "avg_num_neighbors": 10, "mode": mode, "freeze": freeze,
                            "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
              "prediction": {"method": "e3tb", "neurons": [16, 16]}}
        return build_model(None, mo, common).embedding

    def grad(emb, block):
        if block == "2b":
            return [p.requires_grad for p in emb.two_ness.parameters()]
        if block == "3b":
            return [p.requires_grad for p in emb.three_center.parameters()]
        mods = [emb.init_layer, emb.out_edge, emb.out_node, *emb.layers]
        return [p.requires_grad for m in mods for p in m.parameters()]

    # train 3b only: apply 2b+3b but freeze the 2b base -> only 3b learns
    e = build("2b+3b", ["2b"])
    assert e.freeze == {"2b"}
    assert not any(grad(e, "2b")) and all(grad(e, "3b"))

    # freeze 2b+3b in a non-full mode too (explicit, independent of mode's default)
    e = build("2b+3b", ["2b", "3b"])
    assert not any(grad(e, "2b")) and not any(grad(e, "3b"))

    # override the "full" default: freeze=[] trains everything incl. env
    e = build("full", [])
    assert e.freeze == set()
    assert all(grad(e, "2b")) and all(grad(e, "3b")) and all(grad(e, "env"))

    # freeze env explicitly (train only the additive 2b/3b terms on top of a fixed message-passing net)
    e = build("full", ["env"])
    assert not any(grad(e, "env")) and all(grad(e, "2b")) and all(grad(e, "3b"))

    # a string is accepted as a single-block shorthand; unknown blocks are rejected
    assert build("2b", "2b").freeze == {"2b"}
    with _pytest.raises(AssertionError, match="freeze"):
        build("2b", ["nope"])


def test_trinity_scale_shift_heads():
    # trinity's e3tb prediction heads are real E3PerSpecies/E3PerEdgeSpeciesScaleShift modules (not
    # the old identity lambdas), so E3statistics can whiten the multi-scale H targets at train start.
    # Defaults (scales=1, shifts=0) must reproduce the identity behaviour exactly.
    from dptb.nn.build import build_model
    from dptb.nn.rescale import E3PerSpeciesScaleShift, E3PerEdgeSpeciesScaleShift
    common = {"basis": {"C": ["2s", "2p"]}, "device": "cpu", "dtype": "float32",
              "overlap": False, "seed": 42}
    mo = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                        "n_layers": 2, "avg_num_neighbors": 10, "mode": "full",
                        "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
          "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    model = build_model(None, mo, common)
    assert isinstance(model.node_prediction_h, E3PerSpeciesScaleShift)
    assert isinstance(model.edge_prediction_h, E3PerEdgeSpeciesScaleShift)

    torch.manual_seed(0)
    pos = torch.randn(5, 3) * 2.0
    Z = torch.full((5,), 6, dtype=torch.long)
    def mkdd():
        return AtomicData.to_AtomicDataDict(
            AtomicData.from_points(pos=pos, r_max=6.0, er_max=6.0, pbc=False, atomic_numbers=Z))

    with torch.no_grad():
        d_id = model(mkdd())                          # scales=1, shifts=0 -> identity heads
    e_id = d_id[AtomicDataDict.EDGE_FEATURES_KEY].clone()
    n_id = d_id[AtomicDataDict.NODE_FEATURES_KEY].clone()

    # set nontrivial statistics: shift must move node scalar channels, scale must rescale outputs
    ni = model.idp.orbpair_irreps.num_irreps
    ns = model.node_prediction_h.num_scalar
    model.node_prediction_h.set_scale_shift(scales=torch.full((1, ni), 2.0), shifts=torch.full((1, ns), 3.0))
    model.edge_prediction_h.set_scale_shift(scales=torch.full((1, ni), 2.0), shifts=torch.full((1, ns), 0.0))
    with torch.no_grad():
        d_ss = model(mkdd())
    assert not torch.allclose(d_ss[AtomicDataDict.NODE_FEATURES_KEY], n_id)   # shift+scale took effect
    assert not torch.allclose(d_ss[AtomicDataDict.EDGE_FEATURES_KEY], e_id)   # scale took effect
    assert torch.isfinite(d_ss[AtomicDataDict.EDGE_FEATURES_KEY]).all()


def test_decay_envelope_and_trace_loss():
    # (a) the radial decay envelope in E3PerEdgeSpeciesScaleShift: kappa=0 is exactly the identity;
    #     kappa>0 multiplies each channel by exp(-kappa (r - r0)) per bond type.
    # (b) HamilLossAbs trace_weight: T = sum(coeff^2) per irrep slice; the auxiliary loss is zero for
    #     a perfect prediction and positive otherwise, and the total stays finite.
    from dptb.nn.rescale import E3PerEdgeSpeciesScaleShift
    from dptb.nnops.loss import HamilLossAbs
    from dptb.data.transforms import OrbitalMapper

    idp = OrbitalMapper({"C": ["2s", "2p"]}, method="e3tb")
    idp.get_irreps(no_parity=False)
    irr = idp.orbpair_irreps
    head = E3PerEdgeSpeciesScaleShift(field=AtomicDataDict.EDGE_FEATURES_KEY, num_types=1, irreps_in=irr,
                                      out_field=AtomicDataDict.EDGE_FEATURES_KEY, shifts=None, scales=1.)
    torch.manual_seed(0)
    n_edge = 7
    feats = torch.randn(n_edge, irr.dim) + 0.5
    r = torch.linspace(1.0, 5.0, n_edge)
    def run():
        d = {AtomicDataDict.EDGE_FEATURES_KEY: feats.clone(),
             AtomicDataDict.EDGE_TYPE_KEY: torch.zeros(n_edge, 1, dtype=torch.long),
             AtomicDataDict.EDGE_INDEX_KEY: torch.zeros(2, n_edge, dtype=torch.long),
             AtomicDataDict.EDGE_LENGTH_KEY: r.clone()}
        return head(d)[AtomicDataDict.EDGE_FEATURES_KEY]
    out0 = run()                                            # kappa = 0 -> identity
    assert torch.allclose(out0, feats)
    kappa = torch.full((1, irr.num_irreps), 0.7); r0 = torch.full((1, irr.num_irreps), 1.0)
    head.set_decay(kappa=kappa, r0=r0)
    out1 = run()
    env = torch.exp(-0.7 * (r - 1.0)).unsqueeze(-1)         # same kappa for all channels here
    assert torch.allclose(out1, feats * env, atol=1e-6)

    # (b) trace loss
    lossfn = HamilLossAbs(idp=idp, overlap=False, trace_weight=1.0)
    n_at = 3
    dd = {AtomicDataDict.NODE_FEATURES_KEY: torch.randn(n_at, idp.reduced_matrix_element),
          AtomicDataDict.EDGE_FEATURES_KEY: torch.randn(n_edge, idp.reduced_matrix_element),
          AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(n_at, 1, dtype=torch.long),
          AtomicDataDict.EDGE_TYPE_KEY: torch.zeros(n_edge, 1, dtype=torch.long)}
    ref = {k: v.clone() for k, v in dd.items()}
    assert lossfn(dd, ref).item() == 0.0                     # perfect prediction -> 0 (incl. trace)
    dd2 = {k: (v + 0.1 if v.dtype is torch.float32 else v) for k, v in ref.items()}
    L_with = lossfn(dd2, {k: v.clone() for k, v in ref.items()})
    L_wo = HamilLossAbs(idp=idp, overlap=False, trace_weight=0.0)(
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dd2.items()},
        {k: v.clone() for k, v in ref.items()})
    assert torch.isfinite(L_with) and L_with > L_wo > 0      # aux term adds a positive, finite amount


def test_so2_gate_wigner_cache_and_hermitian():
    # (a) SO2_Linear with wigner precomputed == without (bit-level same math);
    # (b) SO2_Linear with so2_gate stays SO(3)-equivariant;
    # (c) trinity full-mode hermitian symmetrization: env edge channel satisfies r_ji = (-1)^J r_ij.
    from e3nn import o3
    from dptb.nn.tensor_product import SO2_Linear, batch_wigner_D, _Jd, wigner_D
    from e3nn.o3 import xyz_to_angles

    torch.manual_seed(0)
    irr_in = o3.Irreps("4x0e+3x1o+2x2e")
    irr_out = o3.Irreps("3x0e+2x1o+2x2e")
    n = 11
    x = torch.randn(n, irr_in.dim)
    R = torch.randn(n, 3)

    # (a) wigner cache exactness (incl. a larger-lmax matrix being sliced)
    m = SO2_Linear(irr_in, irr_out)
    ang = xyz_to_angles(R[:, [1, 2, 0]])
    W_big = batch_wigner_D(4, ang[0], ang[1], torch.zeros_like(ang[0]), _Jd)  # lmax 4 > needed 2
    y0 = m(x, R)
    y1 = m(x, R, wigner=W_big)
    assert torch.allclose(y0, y1, atol=1e-6)

    # (b) equivariance with the SO(2) gate on. Convention: the module reads R via the xyz->yzx
    # permutation, i.e. R[:,[1,2,0]] lives in the e3nn "1o" space, so the rotated vector is
    # perm^-1(D_1o @ perm(R)) (verified to 4e-6 on the ungated module).
    mg = SO2_Linear(irr_in, irr_out, so2_gate=True)
    rot = o3.rand_matrix()
    Din = irr_in.D_from_matrix(rot); Dout = irr_out.D_from_matrix(rot)
    D1 = o3.Irreps("1o").D_from_matrix(rot)
    R_rot = (R[:, [1, 2, 0]] @ D1.t())[:, [2, 0, 1]]
    y = mg(x, R)
    y_rot = mg(x @ Din.t(), R_rot)
    assert torch.allclose(y_rot, y @ Dout.t(), atol=1e-4), (y_rot - y @ Dout.t()).abs().max()

    # (c) full-mode trinity: hermitian env channel
    from dptb.nn.build import build_model
    from dptb.utils.constants import anglrMId
    common = {"basis": {"C": ["2s", "2p"]}, "device": "cpu", "dtype": "float32", "overlap": False, "seed": 42}
    mo = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                        "n_layers": 2, "avg_num_neighbors": 10, "mode": "full", "so2_gate": True,
                        "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
          "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    model = build_model(None, mo, common)
    model.transform = False           # keep reduced features to check the constraint directly
    torch.manual_seed(1)
    pos = torch.randn(5, 3) * 2.0
    dd = AtomicData.to_AtomicDataDict(AtomicData.from_points(
        pos=pos, r_max=6.0, er_max=6.0, pbc=False, atomic_numbers=torch.full((5,), 6, dtype=torch.long)))
    with torch.no_grad():
        out = model(dd)
    idp = model.idp
    ef = out[AtomicDataDict.EDGE_FEATURES_KEY]
    rev = model.embedding._reversed_edges(out)
    assert (rev != torch.arange(len(rev))).all()          # every edge found its reversed partner
    hm, hs = model.embedding.herm_mask, model.embedding.herm_sign
    assert torch.allclose(ef[:, hm], hs * ef[rev][:, hm], atol=1e-5)   # r_ij == (-1)^J r_ji exactly
    assert ef.abs().max() > 0                                          # nontrivial output


def test_boundary_envelope_dtype_stability():
    # Edges within the last ~5% of r_max used to be numerically unstable: the p=6 polynomial cutoff
    # evaluated at r/r_max ~ 1 suffers catastrophic cancellation in float32 (absolute noise ~3e-6 vs
    # true values 1e-6..1e-11), flipping active-set membership, and the env pathway was discontinuous
    # at activation (downstream MLP biases + O(1) spherical harmonics) -> O(0.1-1 eV) dtype-dependent
    # output noise on boundary edges. The boundary_envelope makes the env output continuous in r.
    from dptb.nn.embedding.trinity import boundary_envelope
    from dptb.nn.build import build_model

    # (a) envelope properties: exactly 1 below onset, 0 at r_max, monotone, stable at the tip
    rmax = torch.tensor(6.0)
    r = torch.tensor([0.5, 5.69, 5.7, 5.85, 5.9999999, 6.0])
    env = boundary_envelope(r, rmax, onset=0.95)
    assert env[0] == 1.0 and env[1] == 1.0 and env[2] == 1.0     # untouched below onset (0.95*6=5.7)
    assert 0.0 < env[3] < 1.0
    assert env[4] >= 0.0 and env[4] < 1e-18 and env[5] == 0.0    # smooth, non-negative tip
    assert (env[:-1] >= env[1:]).all()                            # monotone decreasing

    # (b) continuity at the cutoff: the env-channel prediction for a bond at r -> r_max^- must
    # vanish (beyond r_max the edge leaves the graph, so the prediction there is structurally
    # zero). Without the envelope, the downstream MLP biases leave an O(0.01-1) env prediction on
    # such edges (the discontinuity whose float32 corollary was 0.7 eV noise on boundary edges).
    common = {"basis": {"C": ["2s", "2p"]}, "device": "cpu", "dtype": "float32", "overlap": False, "seed": 42}
    mo = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                        "n_layers": 2, "avg_num_neighbors": 10, "mode": "full",
                        "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
          "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    from dptb.utils.tools import setup_seed
    setup_seed(7)
    model = build_model(None, {k: (dict(v) if isinstance(v, dict) else v) for k, v in mo.items()}, dict(common))
    model.transform = False        # inspect reduced features (the env channel) directly
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [2.1, 0.3, -0.4],
                        [-1.8, 1.2, 0.9],
                        [0.0, 0.0, 5.9994]])           # bond 0-3 at r ~ 0.9999*r_max (envelope ~ 8e-7)
    Z = torch.full((4,), 6, dtype=torch.long)
    dd = AtomicData.to_AtomicDataDict(AtomicData.from_points(pos=pos, r_max=6.0, er_max=6.0, pbc=False, atomic_numbers=Z))
    with torch.no_grad():
        out_full = model(dict(dd))
        model.embedding.use_env = False                # 2b+3b only -> env channel by difference
        out_23 = model(AtomicData.to_AtomicDataDict(AtomicData.from_points(
            pos=pos, r_max=6.0, er_max=6.0, pbc=False, atomic_numbers=Z)))
        model.embedding.use_env = True
    ei = out_full[AtomicDataDict.EDGE_INDEX_KEY]
    r = AtomicDataDict.with_edge_vectors({k: (v.clone() if torch.is_tensor(v) else v) for k, v in out_full.items()})[
        AtomicDataDict.EDGE_LENGTH_KEY]
    env_ch = (out_full[AtomicDataDict.EDGE_FEATURES_KEY] - out_23[AtomicDataDict.EDGE_FEATURES_KEY]).abs().max(dim=1).values
    boundary = r > 0.999 * 6.0
    interior = r < 0.9 * 6.0
    assert boundary.any() and interior.any()
    # env channel must be strongly suppressed on the boundary bond but alive in the interior
    assert env_ch[interior].max() > 1e-3, "test setup: env channel unexpectedly inactive"
    assert env_ch[boundary].max() < 1e-4 * env_ch[interior].max(), \
        f"env prediction not continuous at cutoff: boundary {env_ch[boundary].max():.3e} vs interior {env_ch[interior].max():.3e}"

    # (c) THE root mechanism of the measured 0.7 eV dtype noise: E3PerEdgeSpeciesScaleShift skips
    # exactly-zero rows, so without the shift envelope an inactive boundary edge gets NO shift while
    # an epsilon-active one gets the FULL per-channel shift. With r_max set, the shift is enveloped
    # to ~0 at the boundary, so the zero-row flip changes the output only negligibly.
    from dptb.nn.rescale import E3PerEdgeSpeciesScaleShift
    from dptb.data.transforms import OrbitalMapper
    idp2 = OrbitalMapper({"C": ["2s", "2p"]}, method="e3tb")
    idp2.get_irreps(no_parity=False)
    irr = idp2.orbpair_irreps
    rmax = 6.0
    r_edge = torch.tensor([5.999, 5.999, 3.0])          # two boundary edges (one zero-row), one interior
    feats = torch.zeros(3, irr.dim)
    feats[1, 0] = 1e-7                                   # epsilon-active boundary row
    feats[2, :] = 0.5                                    # interior row
    shifts = torch.full((1, sum(1 for ir in irr if ir.ir.l == 0)), 0.7)

    def head(with_rmax):
        h = E3PerEdgeSpeciesScaleShift(field=AtomicDataDict.EDGE_FEATURES_KEY, num_types=1, irreps_in=irr,
                                       out_field=AtomicDataDict.EDGE_FEATURES_KEY, shifts=0., scales=None,
                                       r_max=rmax if with_rmax else None)
        h.set_scale_shift(shifts=shifts)
        d = {AtomicDataDict.EDGE_FEATURES_KEY: feats.clone(),
             AtomicDataDict.EDGE_TYPE_KEY: torch.zeros(3, 1, dtype=torch.long),
             AtomicDataDict.EDGE_INDEX_KEY: torch.zeros(2, 3, dtype=torch.long),
             AtomicDataDict.EDGE_LENGTH_KEY: r_edge.clone()}
        return h(d)[AtomicDataDict.EDGE_FEATURES_KEY]

    out_naked = head(with_rmax=False)
    out_env = head(with_rmax=True)
    # zero-row (edge 0) is always skipped; the flip discontinuity is |out[1] - out[0]| at the boundary
    jump_naked = (out_naked[1] - out_naked[0]).abs().max()
    jump_env = (out_env[1] - out_env[0]).abs().max()
    assert jump_naked > 0.5, f"expected the naked head to jump by ~the shift, got {jump_naked:.3e}"
    assert jump_env < 1e-5, f"shift envelope failed to restore continuity: jump {jump_env:.3e}"
    assert torch.allclose(out_env[2], out_naked[2])      # interior edges: full shift, unchanged


def test_trinity_integration():
    # end-to-end: a real Trinity + e3tb NNENV with the additive three-center term.
    from dptb.nn.build import build_model
    common = {"basis": {"C": ["2s", "2p"]}, "device": "cpu", "dtype": "float32",
              "overlap": False, "seed": 42}
    model_options = {
        "embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                      "n_layers": 2, "avg_num_neighbors": 10, "tp_radial_emb": True, "mode": "2b+3b",
                      "three_center": {"projectors": {"C": ["1s", "2s", "1p"]}, "er_max": 6.0,
                                       "coupling": "block_diag"}},
        "prediction": {"method": "e3tb", "neurons": [16, 16]},
    }
    model = build_model(None, model_options, common)
    assert model.embedding.three_center is not None                  # owned by the Trinity embedding
    assert model.model_options["embedding"]["three_center"]["projectors"] == {"C": ["1s", "2s", "1p"]}

    torch.manual_seed(0)
    pos = torch.randn(6, 3) * 2.0
    Z = torch.full((6,), 6, dtype=torch.long)

    def mkdd():
        return AtomicData.to_AtomicDataDict(
            AtomicData.from_points(pos=pos, r_max=6.0, er_max=6.0, pbc=False, atomic_numbers=Z))

    d1 = model(mkdd())     # 2b + 3b
    edge_with, node_with = d1[AtomicDataDict.EDGE_FEATURES_KEY].clone(), d1[AtomicDataDict.NODE_FEATURES_KEY].clone()
    tc = model.embedding.three_center
    dd_c = mkdd(); model.idp(dd_c); out_c = tc(dd_c)
    edge_contrib = tc.to_feature(out_c[EDGE_THREECENTER_KEY])
    node_contrib = tc.to_feature(out_c[NODE_THREECENTER_KEY])
    model.embedding.use_3b = False                                   # drop the 3b channel -> 2b only
    d0 = model(mkdd())

    assert torch.isfinite(edge_with).all()
    assert edge_contrib.abs().max().item() > 1e-6 and node_contrib.abs().max().item() > 1e-6
    assert torch.allclose(edge_with, d0[AtomicDataDict.EDGE_FEATURES_KEY] + edge_contrib, atol=1e-5)
    assert torch.allclose(node_with, d0[AtomicDataDict.NODE_FEATURES_KEY] + node_contrib, atol=1e-5)


def test_three_center_is_trinity_only():
    # three_center lives under the trinity embedding config; argcheck accepts it there and rejects it
    # under any other embedding (e.g. slem), so it is a Trinity-only feature by construction.
    from dptb.utils.argcheck import model_options
    arg = model_options()
    ok = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                        "n_layers": 2, "avg_num_neighbors": 10,
                        "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
          "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    arg.check_value(arg.normalize_value(ok, trim_pattern="_*"), strict=True)  # no raise

    bad = {"embedding": {"method": "slem", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                         "n_layers": 2, "avg_num_neighbors": 10,
                         "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
           "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    with pytest.raises(Exception):
        arg.check_value(arg.normalize_value(bad, trim_pattern="_*"), strict=True)

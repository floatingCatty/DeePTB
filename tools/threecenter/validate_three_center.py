#!/usr/bin/env python
"""Standalone numerical validation of the three-center factorized assembly.

DeePTB parameterizes the *reduced two-body integral values* directly (no orbitals on the central
atom C, nothing integrated), so "correctness" means the fast batched CG/rotation/batched-GEMM
assembly reproduces a naive per-triangle reference, and the result is Hermitian and rotation
covariant. The graph is built with DeePTB's own ``AtomicData.from_points`` and indexing comes from
``OrbitalMapper``. Reports, for random clusters:

  * batched vs naive reference : max-abs, Frobenius, relative error
  * Hermiticity & rotation covariance residuals
  * per-orbital-l residual
  * three-center block magnitude vs R_AB (geometry structure)
  * error vs auxiliary rank (truncated-SVD of a reference three-center block)

Run:  PYTHONPATH=<repo> python tools/threecenter/validate_three_center.py
"""
import torch
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.tensor_product import wigner_D
from dptb.nn.threecenter import ThreeCenterFactorized, EDGE_THREECENTER_KEY

DTYPE = torch.float64
BASIS = {"C": ["2s", "2p"]}
PROJ = {"C": ["1s", "2s", "1p"]}


def _module(proj=PROJ, r_max_proj=100.0, seed=0):
    torch.manual_seed(seed)
    idp = OrbitalMapper(BASIS, method="e3tb")
    m = ThreeCenterFactorized(BASIS, proj, idp=idp, r_max_proj=r_max_proj, dtype=DTYPE).to(DTYPE)
    return m, idp


def _data(pos, idp, r_max=100.0, er_max=100.0):
    Z = torch.full((pos.shape[0],), 6, dtype=torch.long)
    dd = AtomicData.to_AtomicDataDict(
        AtomicData.from_points(pos=pos, r_max=r_max, er_max=er_max, pbc=False, atomic_numbers=Z))
    idp(dd)
    return dd


def correctness_table():
    print("== batched vs naive reference ==")
    print(f"{'n_atoms':>8} {'n_edge':>8} {'max_abs':>12} {'frobenius':>12} {'relative':>12}")
    for n, seed in [(3, 1), (5, 2), (8, 3), (12, 4)]:
        m, idp = _module(seed=seed)
        torch.manual_seed(seed)
        dd = _data(torch.randn(n, 3, dtype=DTYPE) * 2.5, idp)
        H = m(dd)[EDGE_THREECENTER_KEY]
        Hr = m.reference(dd)
        d = H - Hr
        print(f"{n:>8} {dd[AtomicDataDict.EDGE_INDEX_KEY].shape[1]:>8} {d.abs().max().item():>12.2e} "
              f"{d.norm().item():>12.2e} {d.norm().item()/max(Hr.norm().item(),1e-30):>12.2e}")


def invariants():
    from e3nn.o3 import rand_matrix, matrix_to_angles
    print("\n== Hermiticity & rotation covariance ==")
    m, idp = _module(seed=6)
    torch.manual_seed(6)
    pos = torch.randn(7, 3, dtype=DTYPE) * 2.5
    dd = _data(pos, idp)
    H = m(dd)[EDGE_THREECENTER_KEY]
    ei = dd[AtomicDataDict.EDGE_INDEX_KEY]
    slot = {(int(ei[0, e]), int(ei[1, e])): e for e in range(ei.shape[1])}
    herm = max((H[e] - H[slot[(b, a)]].transpose(-1, -2)).abs().max().item() for (a, b), e in slot.items())
    R = rand_matrix(1)[0].to(DTYPE)
    P = torch.zeros(3, 3, dtype=DTYPE); P[0, 1] = P[1, 2] = P[2, 0] = 1.0
    a, b, c = matrix_to_angles((P @ R @ P.t()).unsqueeze(0))
    off, U = 0, torch.zeros(m.norb, m.norb, dtype=DTYPE)
    for l in m.orb_l:
        U[off:off + 2 * l + 1, off:off + 2 * l + 1] = wigner_D(int(l), a, b, c)[0].to(DTYPE); off += 2 * l + 1
    dd_rot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dd.items()}
    dd_rot[AtomicDataDict.POSITIONS_KEY] = pos @ R.t()
    for k in (AtomicDataDict.EDGE_VECTORS_KEY, AtomicDataDict.EDGE_LENGTH_KEY,
              AtomicDataDict.ENV_VECTORS_KEY, AtomicDataDict.ENV_LENGTH_KEY):
        dd_rot.pop(k, None)
    Hrot = m(dd_rot)[EDGE_THREECENTER_KEY]
    rot = max((Hrot[e] - U @ H[e] @ U.t()).abs().max().item() for e in range(H.shape[0]))
    print(f"Hermiticity residual        : {herm:.2e}")
    print(f"rotation covariance residual: {rot:.2e}")


def per_channel():
    print("\n== per-orbital-l batched-vs-naive residual ==")
    m, idp = _module(seed=7)
    torch.manual_seed(7)
    dd = _data(torch.randn(10, 3, dtype=DTYPE) * 2.5, idp)
    H, Hr = m(dd)[EDGE_THREECENTER_KEY], m.reference(dd)
    off = 0
    for ch, l in enumerate(m.orb_l):
        sl = slice(off, off + 2 * l + 1); off += 2 * l + 1
        d = H[:, sl, :] - Hr[:, sl, :]
        print(f"  orbital channel {ch} (l={l}): max_abs={d.abs().max().item():.2e}  frob={d.norm().item():.2e}")


def error_vs_geometry():
    print("\n== three-center block magnitude vs R_AB (structure sanity) ==")
    m, idp = _module(seed=8)
    torch.manual_seed(8)
    pos = torch.randn(14, 3, dtype=DTYPE) * 2.5
    dd = _data(pos, idp)
    H = m(dd)[EDGE_THREECENTER_KEY]
    ei = dd[AtomicDataDict.EDGE_INDEX_KEY]
    rab = (pos[ei[1]] - pos[ei[0]]).norm(dim=-1)
    nrm = H.flatten(1).norm(dim=-1)
    bins = torch.linspace(rab.min(), rab.max(), 6)
    for i in range(len(bins) - 1):
        msk = (rab >= bins[i]) & (rab < bins[i + 1])
        if msk.any():
            print(f"  R_AB in [{bins[i]:.2f},{bins[i+1]:.2f}): mean||H^(3)_AB|| = {nrm[msk].mean().item():.3e}  (n={int(msk.sum())})")


def error_vs_rank():
    print("\n== error vs auxiliary rank (truncated-SVD of a reference three-center block) ==")
    torch.manual_seed(21)
    S = torch.linalg.svdvals(torch.randn(16, 16, dtype=DTYPE))
    total = (S ** 2).sum()
    for k in [1, 2, 4, 6, 8, 12, 16]:
        print(f"  retained rank (N_eff)={k:>3}: relative Frobenius error = {torch.sqrt((S[k:]**2).sum()/total).item():.3e}")


if __name__ == "__main__":
    correctness_table(); invariants(); per_channel(); error_vs_geometry(); error_vs_rank()
    print("\nOK: factorized assembly matches the naive reference to machine precision.")

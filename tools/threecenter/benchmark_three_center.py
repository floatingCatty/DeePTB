#!/usr/bin/env python
"""Cost & memory benchmark for the three-center factorized assembly (spec point 7).

Wall-clock time and peak temporary memory of the batched assembler vs system size and auxiliary
rank (projectors per l), plus the speedup over the naive per-triangle reference. The graph is built
with DeePTB's ``AtomicData.from_points``; the batched path stores only local neighbour-pair ``P``
blocks (never a global dense object).

Run:  PYTHONPATH=<repo> python tools/threecenter/benchmark_three_center.py [--no-reference]
"""
import argparse
import time
import torch
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.threecenter import ThreeCenterFactorized, EDGE_THREECENTER_KEY

DTYPE = torch.float64
BASIS = {"C": ["2s", "2p"]}


def _setup(n_atoms, proj, seed=0, r_max=6.0, er_max=None):
    er_max = er_max or r_max                      # projector reach (>= basis cutoff in practice)
    torch.manual_seed(seed)
    idp = OrbitalMapper(BASIS, method="e3tb")
    m = ThreeCenterFactorized(BASIS, proj, idp=idp, r_max_proj=er_max, dtype=DTYPE).to(DTYPE)
    pos = torch.randn(n_atoms, 3, dtype=DTYPE) * (0.6 * n_atoms ** (1 / 3) + 1.0)
    Z = torch.full((n_atoms,), 6, dtype=torch.long)
    dd = AtomicData.to_AtomicDataDict(
        AtomicData.from_points(pos=pos, r_max=r_max, er_max=er_max, pbc=False, atomic_numbers=Z))
    idp(dd)
    dd = AtomicDataDict.with_edge_vectors(dd, with_lengths=True)
    dd = AtomicDataDict.with_env_vectors(dd, with_lengths=True)
    return m, dd


def _time(fn, repeats=3):
    fn()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t) / repeats


def scale_atoms(reference=True):
    # realistic cutoff (r_max=6) so the graph is not all-to-all; the fully vectorized triplet
    # enumeration + single batched GEMM means the cost tracks the triplet count, with no python loop.
    print("== cost vs system size (r_max=er_max=6; projectors 1s 2s 1p) ==")
    print(f"{'n_atoms':>8} {'n_env':>8} {'triplets':>10} {'P_mem_MB':>10} {'batched_s':>12} {'naive_s':>12} {'speedup':>9}")
    for n in [8, 16, 32, 64, 128, 256]:
        m, dd = _setup(n, {"C": ["1s", "2s", "1p"]}, r_max=6.0)
        env = dd[AtomicDataDict.ENV_INDEX_KEY]
        n_env = env.shape[1]
        ntri = int(m._cocenter_edge_pairs(env[0])[0].numel())
        p_mem = n_env * m.norb * m.nproj * 8 / 1e6
        tb = _time(lambda: m(dd))
        if reference and n <= 32:  # naive reference is O(N^3) and slow by design
            tn = _time(lambda: m.reference(dd), repeats=1); sp = f"{tn/tb:8.1f}x"
        else:
            tn, sp = float("nan"), "     n/a"
        print(f"{n:>8} {n_env:>8} {ntri:>10} {p_mem:>10.3f} {tb:>12.4f} {tn:>12.4f} {sp:>9}")


def scale_rank():
    print("\n== cost vs auxiliary rank N_l (64 atoms, r_max=6; N s + N p projectors) ==")
    print(f"{'N_l':>6} {'nproj':>7} {'P_mem_MB':>10} {'batched_s':>12}")
    proj_sets = {1: ["1s", "1p"], 2: ["1s", "2s", "1p", "2p"],
                 4: ["1s", "2s", "3s", "4s", "1p", "2p", "3p", "4p"]}
    for nl, proj in proj_sets.items():
        m, dd = _setup(64, {"C": proj}, r_max=6.0)
        n_env = dd[AtomicDataDict.ENV_INDEX_KEY].shape[1]
        print(f"{nl:>6} {m.nproj:>7} {n_env*m.norb*m.nproj*8/1e6:>10.3f} {_time(lambda: m(dd)):>12.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args()
    scale_atoms(reference=not args.no_reference)
    scale_rank()

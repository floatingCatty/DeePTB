"""Tests for the P2/P3 trainer-hardening features:

  P2  per-group learning rate scaled by each block's weight RMS (build_wrms_param_groups)
  P3  weight EMA (ExponentialMovingAverage), gradient clipping, warmup->cosine schedule,
      and the TF32-off / EMA-checkpoint wiring in the Trainer.

They are CPU-only and self-contained (small Trinity model, no dataset needed for the unit
checks). See scratchpad bench_p2p3.py for the on-data A/B/C convergence benchmark.
"""
import os
import torch
import pytest

from dptb.nnops.ema import ExponentialMovingAverage
from dptb.utils.tools import build_wrms_param_groups, get_lr_scheduler


def _small_trinity(overlap=True, mode="2b+3b"):
    from dptb.nn.build import build_model
    common = {"basis": {"C": ["2s", "2p"]}, "device": "cpu", "dtype": "float32", "overlap": overlap}
    mo = {"embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "16x0e+16x1o+8x2e",
                        "n_layers": 3, "avg_num_neighbors": 10, "mode": mode,
                        "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
          "prediction": {"method": "e3tb", "neurons": [16, 16]}}
    torch.manual_seed(0)
    return build_model(None, mo, common)


# ----------------------------- P2: per-group lr -----------------------------

def test_wrms_param_groups_cover_all_trainable_once():
    model = _small_trinity()
    groups = build_wrms_param_groups(model, lr=1e-2)
    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)), "a parameter landed in two groups"
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(grouped) == trainable, "grouping must cover exactly the trainable params"


def test_wrms_param_groups_downscale_small_init_heads():
    # The AtomicResNet prediction heads (edge_prediction_h2 / edge_prediction_s) are init'd at
    # std 1e-3, so their weight RMS is ~20x smaller than the O(1) embedding stack. P2 must give
    # them a correspondingly smaller lr so their *relative* Adam step matches.
    model = _small_trinity()
    groups = {g["name"]: g for g in build_wrms_param_groups(model, lr=1e-2)}
    head_names = [n for n in groups if "prediction" in n]
    assert head_names, "expected at least one prediction head group"
    emb_scale = max(g["lr_scale"] for n, g in groups.items() if n.startswith("embedding.layers"))
    for hn in head_names:
        assert groups[hn]["lr_scale"] < emb_scale, f"{hn} not down-scaled vs embedding"
        assert groups[hn]["lr"] == pytest.approx(1e-2 * groups[hn]["lr_scale"])
    # scales are bounded to [min_scale, max_scale]
    for g in groups.values():
        assert 0.02 - 1e-9 <= g["lr_scale"] <= 1.0 + 1e-9


def test_wrms_scale_composes_with_scheduler():
    # a scheduler must scale each group relative to that group's own base lr
    model = _small_trinity()
    opt = torch.optim.Adam(build_wrms_param_groups(model, lr=1e-2))
    base = [g["lr"] for g in opt.param_groups]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    opt.step(); sch.step()
    after = [g["lr"] for g in opt.param_groups]
    # all groups decayed by the same *ratio* (cosine multiplies each base_lr)
    ratios = [a / b for a, b in zip(after, base)]
    assert max(ratios) - min(ratios) < 1e-6


# ------------------------------- P3: EMA -----------------------------------

def test_ema_tracks_and_restores():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.5, use_num_updates=False)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight += 2.0
    ema.update()  # shadow <- 0.5*w0 + 0.5*(w0+2) = w0 + 1
    assert torch.allclose(ema.shadow[0], w0 + 1.0, atol=1e-6)
    # context manager swaps in EMA weights then restores raw
    with ema.average_parameters():
        assert torch.allclose(m.weight, w0 + 1.0, atol=1e-6)
    assert torch.allclose(m.weight, w0 + 2.0, atol=1e-6)


def test_ema_warmup_decay_ramps():
    # with use_num_updates the effective decay ramps up, so early updates track fast
    torch.manual_seed(0)
    m = torch.nn.Linear(2, 2)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.999, use_num_updates=True)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight += 1.0
    ema.update()  # effective decay = min(0.999, 2/11) = 0.1818 -> shadow moves ~0.818
    moved = (ema.shadow[0] - w0).abs().mean().item()
    assert moved > 0.5, f"warmup step should track fast, moved {moved}"


def test_ema_state_dict_roundtrip_and_mismatch():
    torch.manual_seed(0)
    m = torch.nn.Linear(3, 3)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.9)
    ema.update()
    sd = ema.state_dict()
    ema2 = ExponentialMovingAverage(m.parameters(), decay=0.9)
    ema2.load_state_dict(sd)
    for a, b in zip(ema.shadow, ema2.shadow):
        assert torch.allclose(a, b)
    # loading into a model with a different #params must error, not silently corrupt
    other = torch.nn.Linear(5, 5)
    bad = ExponentialMovingAverage(other.parameters(), decay=0.9)
    with pytest.raises(ValueError):
        bad.load_state_dict(sd)


# ------------------------- P3: warmup-cosine schedule -----------------------

def test_warmup_cosine_shape():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=1e-2)
    sch = get_lr_scheduler(type="warmup_cos", optimizer=opt,
                           warmup_steps=50, start_factor=1e-2, T_max=500, eta_min=1e-5)
    lrs = []
    for _ in range(500):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    assert lrs[0] == pytest.approx(1e-4, rel=1e-3)          # start_factor * base
    peak = max(range(len(lrs)), key=lambda i: lrs[i])
    assert 45 <= peak <= 55                                  # peak right after warmup
    assert lrs[peak] == pytest.approx(1e-2, rel=1e-2)
    assert lrs[-1] == pytest.approx(1e-5, abs=5e-4)          # decays toward eta_min
    assert lrs[-1] < lrs[peak]


# --------------------------- P3: grad clip ---------------------------------

def test_grad_clip_caps_global_norm():
    model = _small_trinity()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p) * 50.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
    assert total <= 1.0 + 1e-4


# --------------- Trainer integration: options wire through -----------------

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "e3_band", "data")


def _si_trainer(tmp_path, train_options):
    """A real Trainer on the tiny Si64 e3_band dataset (basis 1s1p, H+overlap)."""
    from dptb.data import build_dataset
    from dptb.nn.build import build_model
    from dptb.nnops.trainer import Trainer

    if not os.path.isdir(_ROOT):
        pytest.skip("e3_band Si64 fixture not available")

    common = {"basis": {"Si": ["1s", "1p"]}, "device": "cpu", "dtype": "float32",
              "overlap": True, "seed": 1}
    ds = build_dataset(root=_ROOT, type="DefaultDataset", prefix="Si64",
                       get_Hamiltonian=True, get_overlap=True,
                       basis=common["basis"], r_max=5.0, er_max=4.0, oer_max=None, pbc=True)
    mo = {"embedding": {"method": "trinity", "r_max": 5.0, "irreps_hidden": "8x0e+8x1o+4x2e",
                        "n_layers": 2, "avg_num_neighbors": 10, "mode": "2b"},
          "prediction": {"method": "e3tb", "neurons": [8, 8]}}
    torch.manual_seed(0)
    model = build_model(None, mo, common)

    base_train = {
        "num_epoch": 1, "batch_size": 1, "ref_batch_size": 1, "val_batch_size": 1,
        "optimizer": {"type": "Adam", "lr": 1e-3},
        "lr_scheduler": {"type": "exp", "gamma": 0.999},
        "save_freq": 1, "validation_freq": 1, "display_freq": 1, "use_tensorboard": False,
        "update_lr_per_iter": False, "sliding_win_size": 50, "max_ckpt": 2, "valid_fast": True,
        "loss_options": {"train": {"method": "hamil_abs"}},
    }
    base_train.update(train_options)
    return Trainer(train_options=base_train, common_options=common, model=model, train_datasets=ds)


def test_trainer_engages_hardening_and_checkpoints_ema(tmp_path):
    # end-to-end: the Trainer accepts the new options, runs iterations with per-group lr +
    # grad clip + EMA, and the Saver checkpoint carries EMA weights (model_state_dict) plus the
    # raw training weights for exact restart.
    from dptb.plugins.saver import Saver

    trainer = _si_trainer(tmp_path, {
        "grad_clip_norm": 1.0, "ema_decay": 0.99, "per_group_lr": True,
    })
    # options wired through
    assert trainer.ema is not None
    assert trainer.grad_clip_norm == 1.0
    assert len(trainer.optimizer.param_groups) > 1  # per-group lr made multiple groups

    # run two iterations
    trainer.model.train()
    for batch in trainer.train_loader:
        trainer.iteration(batch)
        break
    for batch in trainer.train_loader:
        trainer.iteration(batch)
        break
    assert trainer.ema.num_updates == 2  # EMA updated once per step

    # checkpoint carries raw + ema weights, and model_state_dict == EMA weights
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    saver = Saver()
    saver.register(trainer, str(ckpt_dir))
    saver._save("test", trainer.model, trainer.model.model_options,
                trainer.common_options, trainer.train_options)
    obj = torch.load(str(ckpt_dir / "test.pth"), weights_only=False)
    assert "raw_model_state_dict" in obj and "ema_state_dict" in obj
    # model_state_dict should equal the EMA (averaged) weights, not the raw ones
    with trainer.ema.average_parameters():
        avg = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    for k in avg:
        assert torch.allclose(obj["model_state_dict"][k], avg[k]), k


def test_trainer_default_options_backward_compatible(tmp_path):
    # with none of the new options set, Trainer behaves exactly as before: no EMA, single group,
    # no clipping.
    trainer = _si_trainer(tmp_path, {})
    assert trainer.ema is None
    assert trainer.grad_clip_norm == 0.0
    assert len(trainer.optimizer.param_groups) == 1
    trainer.model.train()
    for batch in trainer.train_loader:
        loss = trainer.iteration(batch)
        break
    assert torch.isfinite(loss)


def test_full_workflow_fresh_restart_initmodel(tmp_path):
    # The features must be consistent across ALL training entry paths with the full stack on
    # (spectral_balance + per_group_lr + ema + grad_clip), mirroring dptb/entrypoints/train.py:
    #   fresh    : build_model(None) -> E3statistics(model) -> Trainer -> train -> save
    #   restart  : Trainer.restart(ckpt) -> exact resume (iter, EMA shadow, optimizer/scheduler)
    #   init_model: build_model(ckpt) -> E3statistics(model) -> fresh Trainer -> train
    from dptb.data import build_dataset
    from dptb.nn.build import build_model
    from dptb.nnops.trainer import Trainer
    from dptb.plugins.saver import Saver

    if not os.path.isdir(_ROOT):
        pytest.skip("e3_band Si64 fixture not available")

    common = {"basis": {"Si": ["1s", "1p"]}, "device": "cpu", "dtype": "float32", "overlap": True, "seed": 1}
    mo = {"embedding": {"method": "trinity", "r_max": 5.0, "irreps_hidden": "8x0e+8x1o+4x2e",
                        "n_layers": 2, "avg_num_neighbors": 10, "mode": "full", "spectral_balance": True,
                        "three_center": {"projectors": {"Si": ["1s", "1p"]}, "er_max": 4.0}},
          "prediction": {"method": "e3tb", "neurons": [8, 8]}}
    train_opts = {
        "num_epoch": 1, "batch_size": 1, "ref_batch_size": 1, "val_batch_size": 1,
        "optimizer": {"type": "Adam", "lr": 1e-3}, "lr_scheduler": {"type": "exp", "gamma": 0.999},
        "save_freq": 1, "validation_freq": 1, "display_freq": 1, "use_tensorboard": False,
        "update_lr_per_iter": False, "sliding_win_size": 50, "max_ckpt": 4, "valid_fast": True,
        "grad_clip_norm": 1.0, "ema_decay": 0.99, "per_group_lr": True,
        "loss_options": {"train": {"method": "hamil_abs"}},
    }

    def make_ds():
        return build_dataset(root=_ROOT, type="DefaultDataset", prefix="Si64",
                             get_Hamiltonian=True, get_overlap=True, basis=common["basis"],
                             r_max=5.0, er_max=4.0, oer_max=None, pbc=True)

    # ---- fresh start ----
    ds = make_ds()
    torch.manual_seed(0)
    model = build_model(None, {k: (dict(v) if isinstance(v, dict) else v) for k, v in mo.items()}, dict(common))
    ds.E3statistics(model=model)                       # data-based head calibration (gains stay 1.0)
    trainer = Trainer(train_options=dict(train_opts), common_options=dict(common), model=model, train_datasets=ds)
    assert trainer.ema is not None and len(trainer.optimizer.param_groups) > 1
    for _ in range(2):
        for batch in trainer.train_loader:
            trainer.iteration(batch)
            break
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    saver = Saver(); saver.register(trainer, str(ckpt_dir))
    saver._save("trinity.iter", trainer.model, trainer.model.model_options,
                trainer.common_options, trainer.train_options)
    ckpt = str(ckpt_dir / "trinity.iter.pth")
    iter_at_save = trainer.iter
    ema_shadow_at_save = [s.clone() for s in trainer.ema.shadow]

    # ---- restart: exact resume ---- (train.py always passes common_options)
    r = Trainer.restart(checkpoint=ckpt, train_datasets=make_ds(), common_options=dict(common))
    assert r.iter == iter_at_save + 1                  # resumes at the next iteration
    assert r.ema is not None
    for a, b in zip(r.ema.shadow, ema_shadow_at_save):
        assert torch.allclose(a, b), "EMA shadow not restored on restart"
    assert len(r.optimizer.param_groups) == len(trainer.optimizer.param_groups)
    # continue training a step
    for batch in r.train_loader:
        assert torch.isfinite(r.iteration(batch))
        break

    # ---- init_model: warm-start weights, fresh training + fresh stats ----
    ds2 = make_ds()
    model2 = build_model(ckpt, {k: (dict(v) if isinstance(v, dict) else v) for k, v in mo.items()}, dict(common))
    ds2.E3statistics(model=model2)                     # re-fit heads from data (as train.py does)
    trainer2 = Trainer(train_options=dict(train_opts), common_options=dict(common), model=model2, train_datasets=ds2)
    trainer2.model.train()
    for batch in trainer2.train_loader:
        assert torch.isfinite(trainer2.iteration(batch))
        break


# ------------------------- hamil_huber loss ----------------------------------

def test_hamil_huber_registered_and_reports_energy_units():
    # hamil_huber is a registered Loss method, a subclass of HamilLossAbs (inherits masking /
    # onsite_shift / trace). Its REPORTED value is the energy-unit metric (identical to hamil_abs,
    # 0.5*(MAE+RMSE), eV); only the gradient is Huber (checked separately).
    from dptb.nnops.loss import Loss, HamilLossAbs, HamilLossHuber
    from dptb.data.transforms import OrbitalMapper

    idp = OrbitalMapper({"Si": ["1s", "1p"]}, method="e3tb")
    lf = Loss(method="hamil_huber", idp=idp, overlap=True, huber_delta=1e-2, dtype=torch.float32, device="cpu")
    absl = HamilLossAbs(idp=idp, overlap=True, dtype=torch.float32, device="cpu")
    assert isinstance(lf, HamilLossHuber) and isinstance(lf, HamilLossAbs)
    assert lf.huber_delta == 1e-2

    # the reported value equals the hamil_abs energy metric (eV) at BOTH large and small residuals,
    # regardless of delta -> the curve reads in eV and is comparable to a 1e-3 target line.
    torch.manual_seed(0)
    for scale in (1.0, 1e-2, 1e-3):
        pre = torch.zeros(200)
        tgt = torch.randn(200) * scale
        assert torch.allclose(lf._elem_loss(pre, tgt), absl._elem_loss(pre, tgt)), scale
    # and for a constant residual the value IS the error magnitude (energy units): |r|=1e-3 -> 1e-3
    assert lf._elem_loss(torch.zeros(50), torch.full((50,), 1e-3)).item() == pytest.approx(1e-3, rel=1e-4)


def test_hamil_huber_value_energy_but_gradient_is_huber():
    # straight-through check: value == energy metric, gradient == Huber gradient (NOT the energy one).
    from dptb.nnops.loss import HamilLossHuber, HamilLossAbs
    from dptb.data.transforms import OrbitalMapper
    idp = OrbitalMapper({"Si": ["1s", "1p"]}, method="e3tb")
    lf = HamilLossHuber(idp=idp, overlap=False, huber_delta=1e-2, dtype=torch.float32, device="cpu")
    absl = HamilLossAbs(idp=idp, overlap=False, dtype=torch.float32, device="cpu")
    tgt = torch.full((64,), 1e-4)                        # deep in the L2 regime

    x = torch.zeros(64, requires_grad=True)
    out = lf._elem_loss(x, tgt)
    assert out.item() == pytest.approx(1e-4, rel=1e-3)   # reported value = energy error (eV)
    out.backward()
    g_huber = x.grad.norm().item()

    x2 = torch.zeros(64, requires_grad=True)
    absl._elem_loss(x2, tgt).backward()
    g_energy = x2.grad.norm().item()
    assert g_huber < 0.1 * g_energy                      # gradient is the vanishing Huber one


def test_hamil_huber_gradient_vanishes_at_minimum():
    # THE point of the Huber loss: near the minimum its gradient -> 0 (so training converges at a
    # fixed lr), unlike hamil_abs whose L1 half keeps a magnitude-independent gradient (lr-limited
    # floor -> the slow power-law tail).
    from dptb.nnops.loss import HamilLossAbs, HamilLossHuber
    from dptb.data.transforms import OrbitalMapper
    idp = OrbitalMapper({"Si": ["1s", "1p"]}, method="e3tb")

    def grad_norm(loss, resid):
        x = torch.zeros(64, requires_grad=True)
        loss._elem_loss(x, torch.full((64,), resid)).backward()
        return x.grad.norm().item()

    huber = HamilLossHuber(idp=idp, overlap=False, huber_delta=1e-2, dtype=torch.float32, device="cpu")
    absl = HamilLossAbs(idp=idp, overlap=False, dtype=torch.float32, device="cpu")
    # far from the minimum both push comparably (Huber in its L1 regime)
    assert grad_norm(huber, 1.0) == pytest.approx(grad_norm(absl, 1.0), rel=0.5)
    # near the minimum the Huber gradient collapses while abs stays O(1)
    g_huber = grad_norm(huber, 1e-4)
    g_abs = grad_norm(absl, 1e-4)
    assert g_huber < 0.05 * g_abs, (g_huber, g_abs)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

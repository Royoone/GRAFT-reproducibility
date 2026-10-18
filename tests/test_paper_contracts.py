"""Check scientific axis, split and scoring contracts on the packaged inputs."""
import numpy as np
from tools.verify import verify_contracts
from experiments.backbone.predict import score


def test_fixed_paper_contracts():
    assert verify_contracts()["internal_run_records"] == 255


def test_perfect_and_constant_predictions():
    x = np.random.default_rng(13).normal(size=(4, 976))
    metrics, _ = score(x, x.copy(), ["a", "b", "c", "d"])
    for k in (20, 50, 100):
        assert abs(metrics[f"r{k}"] - 1) < 1e-12
        assert metrics[f"mse{k}"] == 0
        assert metrics[f"recall{k}"] == 1
    constant, _ = score(x, np.zeros_like(x), ["a", "b", "c", "d"])
    assert constant["r20"] == 0 and constant["rho20"] == 0


def test_fixed_exclusion_does_not_remove_other_metrics():
    x = np.random.default_rng(19).normal(size=(4, 976))
    names = ["BRD-K43389675", "BRD-K49371609", "BRD-K17516382", "other"]
    metrics, per = score(x, x.copy(), names)
    assert metrics["n_r20"] == 1 and metrics["n_r50"] == 4
    assert np.isnan(per["r20"][:3]).all() and np.isnan(per["rho20"][:3]).all()
    assert np.isfinite(per["mse20"]).all()

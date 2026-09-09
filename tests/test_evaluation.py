"""Tests de scripts/evaluate_model.py — le garde-fou d'évaluation continue.

Couvre : calcul des métriques, comparaison aux seuils (plancher absolu +
baisse vs golden run), chargement/gel du golden run, et validation du jeu
de référence. Le scénario `--degrade` (bug de désalignement X/y) est
vérifié de bout en bout via `check_thresholds`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_model as em  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return joblib.load(em.MODELS_DIR / "pyrenex_risk_v2.joblib")


@pytest.fixture(scope="module")
def meta():
    return json.loads((em.MODELS_DIR / "pyrenex_risk_v2.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference_df():
    return em.load_reference_set()


@pytest.fixture(scope="module")
def baseline():
    return json.loads(em.REFERENCE_BASELINE.read_text(encoding="utf-8"))


# --- compute_metrics ---------------------------------------------------------

def test_compute_metrics_returns_the_four_tracked_metrics(model, meta, reference_df):
    metrics = em.compute_metrics(model, reference_df, meta)

    assert set(metrics) == {"f1_macro", "f1_default", "roc_auc", "recall_default"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_compute_metrics_matches_frozen_golden_run(model, meta, reference_df, baseline):
    metrics = em.compute_metrics(model, reference_df, meta)

    for name in ("f1_macro", "f1_default", "roc_auc", "recall_default"):
        assert metrics[name] == pytest.approx(baseline[name], abs=1e-9)


def test_compute_metrics_degrades_when_target_is_misaligned(model, meta, reference_df):
    """Reproduit le scénario `--degrade` : la cible désalignée doit faire
    chuter les métriques vers le niveau du hasard."""
    degraded = reference_df.copy()
    target_column = meta["target_column"]
    degraded[target_column] = degraded[target_column].sample(frac=1, random_state=0).to_numpy()

    metrics = em.compute_metrics(model, degraded, meta)

    assert metrics["roc_auc"] < 0.6


# --- check_thresholds ---------------------------------------------------------

def test_check_thresholds_passes_on_the_golden_run_itself(baseline):
    violations = em.check_thresholds(baseline, baseline)
    assert violations == []


def test_check_thresholds_flags_absolute_floor_violation(baseline):
    metrics = dict(baseline)
    metrics["roc_auc"] = em.THRESHOLDS["roc_auc"]["absolute_min"] - 0.01

    violations = em.check_thresholds(metrics, baseline)

    assert any("plancher absolu" in v and "roc_auc" in v for v in violations)


def test_check_thresholds_flags_excessive_drop_vs_baseline(baseline):
    metrics = dict(baseline)
    tolerance = em.THRESHOLDS["f1_macro"]["max_drop_vs_baseline"]
    metrics["f1_macro"] = baseline["f1_macro"] - tolerance - 0.01

    violations = em.check_thresholds(metrics, baseline)

    assert any("tolérance" in v and "f1_macro" in v for v in violations)


def test_check_thresholds_tolerates_drop_within_tolerance(baseline):
    metrics = dict(baseline)
    tolerance = em.THRESHOLDS["f1_macro"]["max_drop_vs_baseline"]
    metrics["f1_macro"] = baseline["f1_macro"] - tolerance / 2

    violations = em.check_thresholds(metrics, baseline)

    assert violations == []


# --- load_baseline / freeze_baseline ------------------------------------------

def test_load_baseline_raises_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "REFERENCE_BASELINE", tmp_path / "missing_baseline.json")

    with pytest.raises(SystemExit, match="--freeze-baseline"):
        em.load_baseline()


def test_load_baseline_reads_the_frozen_golden_run(tmp_path, monkeypatch):
    baseline_path = tmp_path / "reference_baseline.json"
    baseline_path.write_text(json.dumps({"f1_macro": 0.6}), encoding="utf-8")
    monkeypatch.setattr(em, "REFERENCE_BASELINE", baseline_path)

    assert em.load_baseline() == {"f1_macro": 0.6}


def test_freeze_baseline_writes_metrics_and_metadata(model, meta, reference_df, tmp_path, monkeypatch):
    baseline_path = tmp_path / "reference_baseline.json"
    monkeypatch.setattr(em, "REFERENCE_BASELINE", baseline_path)

    result = em.freeze_baseline(model, reference_df, meta)

    assert baseline_path.exists()
    written = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert written == result
    assert result["model_version"] == meta["model_version"]
    assert result["n_reference"] == len(reference_df)
    assert set(em.THRESHOLDS) <= set(result)


# --- load_reference_set --------------------------------------------------------

def test_load_reference_set_raises_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "REFERENCE_SET", tmp_path / "missing_reference_set.csv")

    with pytest.raises(SystemExit, match="data/README.md"):
        em.load_reference_set()


def test_load_reference_set_rejects_the_20_line_template(tmp_path, monkeypatch):
    template_like = pd.DataFrame({"loan_status": ["Fully Paid"] * 15 + ["Charged Off"] * 5})
    csv_path = tmp_path / "reference_set.csv"
    template_like.to_csv(csv_path, index=False)
    monkeypatch.setattr(em, "REFERENCE_SET", csv_path)

    with pytest.raises(SystemExit, match="reference_set_TEMPLATE"):
        em.load_reference_set()


def test_load_reference_set_rejects_a_single_class(tmp_path, monkeypatch):
    single_class = pd.DataFrame({"loan_status": ["Fully Paid"] * 200})
    csv_path = tmp_path / "reference_set.csv"
    single_class.to_csv(csv_path, index=False)
    monkeypatch.setattr(em, "REFERENCE_SET", csv_path)

    with pytest.raises(SystemExit):
        em.load_reference_set()


def test_load_reference_set_accepts_the_real_reference_set(reference_df):
    assert len(reference_df) >= 100
    assert reference_df.iloc[:, -1].nunique() == 2

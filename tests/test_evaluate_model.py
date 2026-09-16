"""Tests de l'évaluation continue (M5-B2, tâche 5).

Isolés de data/reference_set.csv et du modèle réel (tmp_path + monkeypatch) :
rapides, et sans risque de corrompre le golden run réel en s'exécutant.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# scripts/ n'est pas un package : on l'ajoute au sys.path pour l'import direct.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate_model as em  # noqa: E402


# ---------------------------------------------------------------------------
# check_thresholds — logique pure, aucun modèle nécessaire
# ---------------------------------------------------------------------------
def test_check_thresholds_no_violation_on_unchanged_model():
    baseline = {"f1_macro": 0.67, "f1_defaut": 0.63, "roc_auc": 0.77, "recall_defaut": 0.55}
    metrics = dict(baseline)  # écart nul, comme un modèle inchangé
    assert em.check_thresholds(metrics, baseline) == []


def test_check_thresholds_detects_absolute_floor_violation():
    baseline = {"f1_macro": 0.67, "f1_defaut": 0.63, "roc_auc": 0.77, "recall_defaut": 0.55}
    metrics = dict(baseline)
    metrics["roc_auc"] = 0.40  # < absolute_min (0.65), et loin en dessous du golden run
    violations = em.check_thresholds(metrics, baseline)
    assert any("plancher absolu" in v and "roc_auc" in v for v in violations)


def test_check_thresholds_detects_relative_drop_violation():
    baseline = {"f1_macro": 0.67, "f1_defaut": 0.63, "roc_auc": 0.77, "recall_defaut": 0.55}
    metrics = dict(baseline)
    # baisse de 0.10 sur recall_defaut : > 0.065 (tolérance), mais reste > 0.45 (plancher)
    metrics["recall_defaut"] = baseline["recall_defaut"] - 0.10
    violations = em.check_thresholds(metrics, baseline)
    assert any("chuté" in v and "recall_defaut" in v for v in violations)
    assert not any("plancher absolu" in v and "recall_defaut" in v for v in violations)


# ---------------------------------------------------------------------------
# degrade_dataframe — désalignement X/y reproductible
# ---------------------------------------------------------------------------
def test_degrade_dataframe_desyncs_target_reproductibly():
    meta = {"target_column": "target"}
    df = pd.DataFrame({
        "feature": range(20),
        "target": [0, 1] * 10,
    })

    degraded = em.degrade_dataframe(df, meta, random_state=0)

    # même multi-ensemble de valeurs cible (une permutation, pas une invention)
    assert sorted(degraded["target"]) == sorted(df["target"])
    # les features ne bougent pas
    assert degraded["feature"].tolist() == df["feature"].tolist()
    # la permutation est EXACTEMENT celle attendue pour random_state=0 (idempotence)
    expected_perm = np.random.RandomState(0).permutation(len(df))
    expected_target = df["target"].to_numpy()[expected_perm]
    assert degraded["target"].tolist() == expected_target.tolist()


# ---------------------------------------------------------------------------
# _scalarize — défense contre les valeurs non-scalaires pour MLflow
# ---------------------------------------------------------------------------
def test_scalarize_passes_through_scalars():
    assert em._scalarize(5) == 5
    assert em._scalarize("balanced") == "balanced"
    assert em._scalarize(None) is None
    assert em._scalarize(True) is True


def test_scalarize_stringifies_non_scalars():
    result = em._scalarize([1, 2, 3])
    assert isinstance(result, str)
    assert result == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# freeze_baseline / load_baseline — round-trip avec un modèle factice
# ---------------------------------------------------------------------------
class _DummyModel:
    """Modèle factice déterministe : évite de dépendre du vrai .joblib."""

    def predict_proba(self, X):
        x = X["x1"].to_numpy()
        p = np.where(x > x.mean(), 0.8, 0.2)
        return np.column_stack([1 - p, p])


@pytest.fixture
def fake_meta():
    return {
        "model_version": "vtest",
        "dataset_sha256": "deadbeef",
        "hyperparameters": {"n_estimators": 10, "n_jobs": -1},
        "feature_columns_numeric": ["x1"],
        "feature_columns_categorical": [],
        "target_column": "label",
        "target_mapping": {"neg": 0, "pos": 1},
    }


@pytest.fixture
def fake_df():
    rng = np.random.RandomState(1)
    x1 = rng.normal(size=40)
    labels = ["pos" if v > 0 else "neg" for v in x1]
    return pd.DataFrame({"x1": x1, "label": labels})


def test_compute_metrics_is_idempotent(fake_meta, fake_df):
    model = _DummyModel()
    m1 = em.compute_metrics(model, fake_df, fake_meta)
    m2 = em.compute_metrics(model, fake_df, fake_meta)
    assert m1 == m2  # même modèle, même jeu, même mesure : aucune tolérance


def test_freeze_and_load_baseline_roundtrip(tmp_path, monkeypatch, fake_meta, fake_df):
    monkeypatch.setattr(em, "REFERENCE_SET", tmp_path / "reference_set.csv")
    monkeypatch.setattr(em, "REFERENCE_BASELINE", tmp_path / "reference_baseline.json")

    model = _DummyModel()
    frozen = em.freeze_baseline(model, fake_df, fake_meta)

    assert set(frozen["metrics"]) == em._EXPECTED_METRICS
    assert (tmp_path / "reference_baseline.json").exists()

    loaded = em.load_baseline()
    assert loaded == frozen["metrics"]


def test_load_baseline_fails_clearly_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "REFERENCE_BASELINE", tmp_path / "does_not_exist.json")
    with pytest.raises(SystemExit, match="freeze-baseline"):
        em.load_baseline()


# ---------------------------------------------------------------------------
# load_reference_set — garde-fou sur la validité du jeu (<100 lignes / mono-classe)
# ---------------------------------------------------------------------------
def test_load_reference_set_rejects_too_few_rows(tmp_path, monkeypatch):
    small = tmp_path / "reference_set.csv"
    pd.DataFrame({"f1": range(10), "target": [0, 1] * 5}).to_csv(small, index=False)
    monkeypatch.setattr(em, "REFERENCE_SET", small)

    with pytest.raises(SystemExit):
        em.load_reference_set()


def test_load_reference_set_rejects_mono_class(tmp_path, monkeypatch):
    mono = tmp_path / "reference_set.csv"
    pd.DataFrame({"f1": range(150), "target": [1] * 150}).to_csv(mono, index=False)
    monkeypatch.setattr(em, "REFERENCE_SET", mono)

    with pytest.raises(SystemExit):
        em.load_reference_set()


def test_load_reference_set_accepts_valid_set(tmp_path, monkeypatch):
    ok = tmp_path / "reference_set.csv"
    pd.DataFrame({"f1": range(150), "target": [0, 1] * 75}).to_csv(ok, index=False)
    monkeypatch.setattr(em, "REFERENCE_SET", ok)

    loaded = em.load_reference_set()
    assert len(loaded) == 150
"""Évaluation continue + tracking MLflow.

À chaque release : recalcule les métriques cibles sur un jeu de référence
figé, **trace le run dans MLflow**, compare aux seuils, et **sort un code
retour non-zéro** si dégradation (→ bloque la release en CI).

Renommez ce fichier en `scripts/evaluate_model.py` une fois complété.
Mini-cours : `07_MLflow_tracking_essentiel.md` + `08_Evaluation_continue_seuils`.

Usage cible::

    python scripts/evaluate_model.py --freeze-baseline             # une fois, au gel du jeu
    python scripts/evaluate_model.py --release-tag v2.0.0
    python scripts/evaluate_model.py --release-tag bad --degrade   # test du rouge
    mlflow ui    # comparer les runs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, recall_score

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "services" / "model" / "models"
REFERENCE_SET = ROOT / "data" / "reference_set.csv"
REFERENCE_BASELINE = ROOT / "data" / "reference_baseline.json"

DECISION_THRESHOLD = 0.5 

THRESHOLDS: dict[str, dict[str, float]] = {
    "f1_macro":       {"absolute_min": 0.50, "max_drop_vs_baseline": 0.06},
    "f1_defaut":      {"absolute_min": 0.35, "max_drop_vs_baseline": 0.08},
    "roc_auc":        {"absolute_min": 0.65, "max_drop_vs_baseline": 0.05},
    "recall_defaut":  {"absolute_min": 0.45, "max_drop_vs_baseline": 0.08},
}

_EXPECTED_METRICS = {"f1_macro", "f1_defaut", "roc_auc", "recall_defaut"}
assert set(THRESHOLDS) == _EXPECTED_METRICS, (
    f"THRESHOLDS ({sorted(THRESHOLDS)}) doit couvrir exactement les 4 "
    f"métriques cibles ({sorted(_EXPECTED_METRICS)})."
)
 
def load_meta() -> dict:
    with open(MODELS_DIR / "pyrenex_risk_v2.json", encoding="utf-8") as f:
        return json.load(f)


def build_X(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    cols = meta["feature_columns_numeric"] + meta["feature_columns_categorical"]
    return df[cols]
 
 
def build_y(df: pd.DataFrame, meta: dict) -> np.ndarray:
    target_col = meta["target_column"]
    mapping = meta["target_mapping"]
    return df[target_col].map(mapping).to_numpy()

def compute_metrics(model, df: pd.DataFrame, meta: dict) -> dict[str, float]:
    """Calcule les métriques cibles sur le jeu de référence."""
    X = build_X(df, meta)
    y_true = build_y(df, meta)
 
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
 
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_defaut": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "recall_defaut": float(recall_score(y_true, y_pred, pos_label=1)),
    }


def check_thresholds(metrics: dict[str, float], baseline: dict) -> list[str]:
    """Retourne la liste des violations de seuil (vide = release OK)."""
    violations = []
    for name, rule in THRESHOLDS.items():
        v = metrics[name]
 
        if v < rule["absolute_min"]:
            violations.append(
                f"{name}={v:.4f} < plancher absolu {rule['absolute_min']}"
            )
 
        base_v = baseline.get(name)
        if base_v is not None:
            drop = base_v - v
            if drop > rule["max_drop_vs_baseline"]:
                violations.append(
                    f"{name}={v:.4f} a chuté de {drop:.4f} vs golden run "
                    f"{base_v:.4f} (max toléré {rule['max_drop_vs_baseline']})"
                )
    return violations


def load_baseline() -> dict:
    """Charge le golden run (baseline mesurée sur le jeu de référence)."""
    if not REFERENCE_BASELINE.exists():
        raise SystemExit(
            f"{REFERENCE_BASELINE} est absent.\n"
            "Lancez d'abord : python scripts/evaluate_model.py --freeze-baseline\n"
            "Rappel : la baseline du garde-fou est le golden run mesuré sur "
            "VOTRE jeu de référence, pas metrics_holdout du meta json."
        )
    with open(REFERENCE_BASELINE, encoding="utf-8") as f:
        frozen = json.load(f)
    return frozen["metrics"]


def freeze_baseline(model, df: pd.DataFrame, meta: dict) -> dict:
    """Mesure et gèle le golden run sur le jeu de référence."""
    metrics = compute_metrics(model, df, meta)
    frozen = {
        "model_version": meta["model_version"],
        "reference_set": REFERENCE_SET.name,
        "n_reference": len(df),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    REFERENCE_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_BASELINE, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    return frozen


def load_reference_set() -> pd.DataFrame:
    """Charge le jeu de référence, avec un garde-fou sur sa validité."""
    if not REFERENCE_SET.exists():
        raise SystemExit(
            f"{REFERENCE_SET} est absent.\n"
        )
    df = pd.read_csv(REFERENCE_SET)
    if len(df) < 100 or df.iloc[:, -1].nunique() < 2:
        raise SystemExit(
            f"{REFERENCE_SET} contient {len(df)} ligne(s) et "
            f"{df.iloc[:, -1].nunique()} classe(s) de cible.\n"
        )
    return df

def _scalarize(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def degrade_dataframe(df: pd.DataFrame, meta: dict, random_state: int = 0) -> pd.DataFrame:
    """Simule une régression réaliste : désaligne X et y."""
    degraded = df.copy()
    target_col = meta["target_column"]
    rng = np.random.RandomState(random_state)
    degraded[target_col] = rng.permutation(degraded[target_col].to_numpy())
    return degraded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-tag", default="dev")
    parser.add_argument("--degrade", action="store_true")
    parser.add_argument("--freeze-baseline", action="store_true")
    args = parser.parse_args()

    model = joblib.load(MODELS_DIR / "pyrenex_risk_v2.joblib")
    meta = json.loads((MODELS_DIR / "pyrenex_risk_v2.json").read_text(encoding="utf-8"))
    df = load_reference_set()

    if args.freeze_baseline:
        print(json.dumps(freeze_baseline(model, df, meta), indent=2))
        return 0

    if args.degrade:
        df = degrade_dataframe(df, meta)

    metrics = compute_metrics(model, df, meta)
    baseline = load_baseline()  # ← le golden run, PAS metrics_holdout
    violations = check_thresholds(metrics, baseline)

    # --- Bloc MLflow ------------------
    mlflow.set_experiment("pyrenex-eval-continue")
    with mlflow.start_run(run_name=args.release_tag):
        hp = {
            f"hp_{k}": _scalarize(v)
            for k, v in meta["hyperparameters"].items()
            if k != "n_jobs"
        }

        mlflow.log_params(
            {
                "model_version": meta["model_version"],
                "release_tag": args.release_tag,
                "reference_set": REFERENCE_SET.name,
                "n_reference": len(df),
                "dataset_sha256": meta["dataset_sha256"],
                **hp,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tag("release_blocked", str(bool(violations)))
    # ------------------------------------------------------------------------

    print(json.dumps({"metrics": metrics, "violations": violations}, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

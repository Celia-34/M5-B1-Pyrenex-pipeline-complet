"""
Tâche 1 — M5-B2 : construction du jeu de référence + estimation du bruit.

Usage :
    python prepare_reference_set.py --build
    python prepare_reference_set.py --build --composition representative
    python prepare_reference_set.py --bootstrap

"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score

ROOT = "."
HOLDOUT_PATH = f"{ROOT}/data/lending_club_holdout.csv"
REFERENCE_PATH = f"{ROOT}/data/reference_set.csv"
NOISE_PATH = f"{ROOT}/data/reference_noise.json"
MODEL_PATH = f"{ROOT}/services/model/models/pyrenex_risk_v2.joblib"
META_PATH = f"{ROOT}/services/model/models/pyrenex_risk_v2.json"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5  # cf. note en bas de fichier


def load_meta() -> dict:
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_X(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Colonnes brutes attendues par le pipeline, dans l'ordre du meta json."""
    cols = meta["feature_columns_numeric"] + meta["feature_columns_categorical"]
    return df[cols]


def build_y(df: pd.DataFrame, meta: dict) -> np.ndarray:
    target_col = meta["target_column"]
    mapping = meta["target_mapping"]
    return df[target_col].map(mapping).to_numpy()


# ---------------------------------------------------------------------------
# Construction du jeu de référence
# ---------------------------------------------------------------------------
def build_reference_set(composition: str):
    meta = load_meta()
    target_col = meta["target_column"]
    mapping = meta["target_mapping"]
    positive_label = [k for k, v in mapping.items() if v == 1][0]

    df = pd.read_csv(HOLDOUT_PATH)
    pos = df[df[target_col] == positive_label]
    neg = df[df[target_col] != positive_label]

    if composition == "balanced":
        n_each = 250
        pos_sample = pos.sample(n=n_each, random_state=RANDOM_STATE)
        neg_sample = neg.sample(n=n_each, random_state=RANDOM_STATE)

    elif composition == "representative":
        n_total = 500
        frac = n_total / len(df)
        pos_sample = pos.sample(frac=frac, random_state=RANDOM_STATE)
        neg_sample = neg.sample(frac=frac, random_state=RANDOM_STATE)

    else:
        raise ValueError(f"composition inconnue : {composition}")

    ref = pd.concat([pos_sample, neg_sample]).sample(
        frac=1, random_state=RANDOM_STATE
    )

    ref.to_csv(REFERENCE_PATH, index=False)

    n = len(ref)
    n_pos = int((ref[target_col] == positive_label).sum())
    print(f"[build] {REFERENCE_PATH} écrit : {n} lignes, {n_pos} positifs "
          f"({n_pos/n:.1%}) — composition = {composition}")
    print("        -> à coller dans evaluation_thresholds.md : "
          f"n={n}, n_pos={n_pos}, %pos={n_pos/n:.1%}")

    if n < 100 or ref[target_col].nunique() < 2:
        raise SystemExit("[build] jeu construit invalide (<100 lignes ou mono-classe)")


# ---------------------------------------------------------------------------
# Bootstrap du bruit sur les 4 métriques cibles
# ---------------------------------------------------------------------------
def bootstrap_all_metrics(y_true, y_pred, y_proba, n_boot=500):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)

    scores = {"f1_macro": [], "f1_defaut": [], "roc_auc": [], "recall_defaut": []}
    skipped = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]

        if len(np.unique(yt)) < 2:
            skipped += 1
            continue  # tirage mono-classe (rare sur un jeu balanced, on l'écarte)

        scores["f1_macro"].append(f1_score(yt, yp, average="macro"))
        scores["f1_defaut"].append(f1_score(yt, yp, pos_label=1))
        scores["roc_auc"].append(roc_auc_score(yt, ypr))
        scores["recall_defaut"].append(recall_score(yt, yp, pos_label=1))

    if skipped:
        print(f"[bootstrap] {skipped}/{n_boot} tirages mono-classe écartés")

    sigmas = {k: float(np.std(v)) for k, v in scores.items()}
    for k, s in sigmas.items():
        print(f"[bootstrap] sigma({k}) = {s:.4f}  ->  tolérance mini (2σ) = {2*s:.4f}")
    return sigmas


def run_bootstrap():
    meta = load_meta()
    model = joblib.load(MODEL_PATH)

    ref = pd.read_csv(REFERENCE_PATH)
    X_ref = build_X(ref, meta)
    y_true = build_y(ref, meta)

    y_proba = model.predict_proba(X_ref)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    sigmas = bootstrap_all_metrics(y_true, y_pred, y_proba)

    with open(NOISE_PATH, "w", encoding="utf-8") as f:
        json.dump(sigmas, f, indent=2)
    print(f"[bootstrap] sigmas écrits dans {NOISE_PATH}")
    print("        -> à coller dans evaluation_thresholds.md, tableau bruit mesuré")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument(
        "--composition",
        choices=["balanced", "representative"],
        default="balanced",
        help="balanced = 250/250 (recommandé, réduit le bruit sur recall_défaut) "
             "| representative = ~18%% de défauts comme la production",
    )
    args = parser.parse_args()

    if args.build:
        build_reference_set(args.composition)
    if args.bootstrap:
        run_bootstrap()
    if not (args.build or args.bootstrap):
        parser.print_help()


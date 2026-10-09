from __future__ import annotations

"""Baseline V2 : gradient boosting multiclasses, sans utilisation du Test.

Entrées : splits chronologiques naturels Train / Validation.
Cibles : -1=SL, 0=NO_TOUCH, 1=TP pour LONG et SHORT séparément.
NaN=ambigu ou indisponible : exclusion avec décompte.
Comparateurs : prior du Train et régression logistique V1.
Résultats descriptifs uniquement ; aucun backtest ni choix de seuil sur Test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

# Les scripts V1 et V2 doivent être dans le même dossier scripts/.
from evaluate_tabular_baseline_v1 import (
    CLASSES, FEATURES, LABEL_NAMES, TP_R, SL_R,
    TRAIN_FILE, VAL_FILE, load, multiclass_brier,
)

OUTPUT_DIR = Path("data/evaluation/tabular-baseline-v2")
V1_METRICS = Path("data/evaluation/tabular-baseline-v1/metrics.json")
V1_PREDICTIONS = Path(
    "data/evaluation/tabular-baseline-v1/validation_probabilities.parquet"
)

# Configuration unique, volontairement conservatrice, fixée avant Validation.
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 80,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": 3407,
}


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    predicted = np.asarray(CLASSES)[np.argmax(p, axis=1)]
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, p, labels=CLASSES)),
        "multiclass_brier": multiclass_brier(y, p),
    }


def get_v1_comparison(side: str, timestamps: pd.Series):
    if not V1_METRICS.exists() or not V1_PREDICTIONS.exists():
        raise FileNotFoundError(
            "Exécuter d'abord evaluate_tabular_baseline_v1.py "
            "pour garantir une comparaison sur les mêmes exemples."
        )
    metrics = json.loads(V1_METRICS.read_text(encoding="utf-8"))
    previous = next((x for x in metrics if x["side"] == side), None)
    if previous is None:
        raise ValueError(f"Absence de métriques V1 pour {side}")
    stored = pd.read_parquet(V1_PREDICTIONS)
    stored = stored.loc[stored["side"] == side].copy()
    stored["timestamp"] = pd.to_datetime(stored["timestamp"], utc=True)
    if stored["timestamp"].duplicated().any():
        raise ValueError(f"Horodatages V1 dupliqués pour {side}")
    if len(stored) != len(timestamps) or not np.array_equal(
        stored["timestamp"].to_numpy(), timestamps.to_numpy()
    ):
        raise ValueError(
            f"{side}: univers / ordre Validation différent de la V1"
        )
    return previous, stored


def heuristic_eligible(df: pd.DataFrame, side: str) -> np.ndarray:
    score = (df["trend_1h"] + df["4h_trend"] + df["1d_trend"]).astype(int)
    score += (df["close"] > df["ema20"]).astype(int) - (
        df["close"] < df["ema20"]
    ).astype(int)
    score += (df["ema20"] > df["ema50"]).astype(int) - (
        df["ema20"] < df["ema50"]
    ).astype(int)
    score += df["rsi14"].between(52, 70, inclusive="both").astype(int)
    score -= df["rsi14"].between(30, 48, inclusive="both").astype(int)
    return ((score >= 2) if side == "LONG" else (score <= -2)).to_numpy()


def signal_summary(
    y: np.ndarray, side: str, model_name: str,
    eligible: np.ndarray, proxy: np.ndarray,
) -> list[dict]:
    rows = []
    for group, mask in (
        ("ALL_VALID", np.ones(len(y), dtype=bool)),
        ("HEURISTIC_ELIGIBLE", eligible),
        ("HEURISTIC_AND_PROXY_POSITIVE", eligible & (proxy > 0)),
    ):
        values = y[mask]
        n = int(mask.sum())
        wins = int((values == 1).sum())
        losses = int((values == -1).sum())
        no_touch = int((values == 0).sum())
        resolved = wins + losses
        rows.append({
            "side": side,
            "model": model_name,
            "group": group,
            "signals": n,
            "wins": wins,
            "losses": losses,
            "no_touch": no_touch,
            "win_rate_resolved": wins / resolved if resolved else np.nan,
            "r_per_signal_zero_no_touch": (
                (TP_R * wins - SL_R * losses) / n if n else np.nan
            ),
        })
    return rows


def main() -> None:
    print("=" * 72)
    print("TABULAR BASELINE V2 - HISTGRADIENTBOOSTING - TRAIN / VALIDATION")
    print("=" * 72)
    print("Paramètres fixes :", MODEL_PARAMS)
    train = load(TRAIN_FILE)
    validation = load(VAL_FILE)
    if (
        train["available_at"].max() + pd.Timedelta(hours=24)
        >= validation["available_at"].min()
    ):
        raise ValueError("Purge chronologique Train / Validation insuffisante")
    print(f"TRAIN: {len(train)} | VALIDATION: {len(validation)}")

    reports, predicted_rows, summary_rows = [], [], []

    for side in ("LONG", "SHORT"):
        outcome_col = f"{side.lower()}_outcome_12h"
        tr = train.loc[train[outcome_col].isin(CLASSES)].copy()
        va = validation.loc[validation[outcome_col].isin(CLASSES)].copy()
        y_train = tr[outcome_col].astype(int).to_numpy()
        y_val = va[outcome_col].astype(int).to_numpy()
        if not len(va) or set(np.unique(y_train)) != set(CLASSES):
            raise ValueError(f"{side}: données ou classes insuffisantes")

        original_metrics, v1_probabilities = get_v1_comparison(
            side, va["timestamp"]
        )
        if (
            original_metrics["train_valid"] != len(tr)
            or original_metrics["validation_valid"] != len(va)
        ):
            raise ValueError(f"{side}: effectifs différents de la V1")
        if not np.array_equal(
            v1_probabilities["outcome"].to_numpy(), y_val
        ):
            raise ValueError(f"{side}: issues Validation différentes de la V1")

        model = Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", keep_empty_features=True
            )),
            ("classifier", HistGradientBoostingClassifier(**MODEL_PARAMS)),
        ])
        model.fit(tr[FEATURES], y_train)
        raw_prob = model.predict_proba(va[FEATURES])
        classes = model.named_steps["classifier"].classes_.tolist()
        probabilities = np.column_stack([
            raw_prob[:, classes.index(c)] for c in CLASSES
        ])
        prior = np.array([(y_train == c).mean() for c in CLASSES])
        prior_prob = np.tile(prior, (len(va), 1))
        v1_prob = v1_probabilities[
            ["p_sl", "p_no_touch", "p_tp"]
        ].to_numpy()

        comparisons = {
            "train_prior": scores(y_val, prior_prob),
            "logistic_v1": scores(y_val, v1_prob),
            "hist_gradient_boosting": scores(y_val, probabilities),
        }
        # Vérifie également les résultats enregistrés de V1.
        for key, saved_key in (
            ("train_prior", "train_prior_baseline"),
            ("logistic_v1", "logistic"),
        ):
            for metric in ("log_loss", "multiclass_brier"):
                if not np.isclose(
                    comparisons[key][metric],
                    original_metrics[saved_key][metric],
                    atol=1e-9,
                ):
                    raise ValueError(
                        f"{side}: {key}/{metric} diffère du rapport V1"
                    )

        print(
            f"\n{side}: Train valide={len(tr)} / {len(train)}"
            f", Validation valide={len(va)} / {len(validation)}"
        )
        print(
            f"  Exclus (ambigus/manquants) : "
            f"Train={len(train)-len(tr)}, "
            f"Validation={len(validation)-len(va)}"
        )
        for name, result in comparisons.items():
            print(
                f"  {name:24s} | "
                f"logloss={result['log_loss']:.4f} "
                f"brier={result['multiclass_brier']:.4f} "
                f"accuracy={result['accuracy']:.4f}"
            )
        delta = (
            comparisons["hist_gradient_boosting"]["log_loss"]
            - comparisons["train_prior"]["log_loss"]
        )
        print(f"  Delta logloss (boosting - prior): {delta:+.4f}")

        report = {
            "side": side,
            "train_valid": len(tr),
            "validation_valid": len(va),
            "train_excluded": len(train) - len(tr),
            "validation_excluded": len(validation) - len(va),
            "train_class_counts": {
                LABEL_NAMES[c]: int((y_train == c).sum()) for c in CLASSES
            },
            "validation_class_counts": {
                LABEL_NAMES[c]: int((y_val == c).sum()) for c in CLASSES
            },
            "metrics": comparisons,
            "fixed_model_params": MODEL_PARAMS,
        }
        reports.append(report)

        eligible = heuristic_eligible(va, side)
        for name, prob in (
            ("logistic_v1", v1_prob),
            ("hist_gradient_boosting", probabilities),
        ):
            proxy = TP_R * prob[:, 2] - SL_R * prob[:, 0]
            summary_rows.extend(
                signal_summary(y_val, side, name, eligible, proxy)
            )

        output = va[["timestamp", "available_at", outcome_col]].copy()
        output = output.rename(columns={outcome_col: "outcome"})
        output["side"] = side
        output["heuristic_eligible"] = eligible
        for idx, c in enumerate(CLASSES):
            output[f"p_{LABEL_NAMES[c].lower()}"] = probabilities[:, idx]
        output["proxy_expected_r"] = (
            TP_R * output["p_tp"] - SL_R * output["p_sl"]
        )
        predicted_rows.append(output)

    summaries = pd.DataFrame(summary_rows)
    print("\nDIAGNOSTIC PAR SIGNAUX - NON BACKTEST (NO_TOUCH=0R PROVISOIRE)")
    print(summaries.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.concat(predicted_rows, ignore_index=True).to_parquet(
        OUTPUT_DIR / "validation_probabilities.parquet", index=False
    )
    summaries.to_csv(OUTPUT_DIR / "validation_signal_summary.csv", index=False)
    print(f"\nRésultats : {OUTPUT_DIR.resolve()}")
    print("Aucun Test consulté. Aucun hyperparamètre ajusté sur Validation.")
    print("Rappel : les NO_TOUCH ne sont pas des trades à 0R réels.")


if __name__ == "__main__":
    main()

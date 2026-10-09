from __future__ import annotations

"""Audit descriptif des trades exécutés en Validation (aucun réentraînement).

Entrée : trades.csv du backtest événementiel V1 et features causales du marché.
Aucune sélection de seuil ou simulation de stratégie alternative.
La contribution logarithmique est additive pour les trades séquentiels à
allocation=100 %, mais n'est pas un P&L indépendant / attribuable causalement.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_TRADES = Path("data/evaluation/backtest-event-driven-v1/validation_only/trades.csv")
MARKET = Path("data/processed/btc_usdc_1h_labeled.parquet")
OUT = Path("data/evaluation/backtest-concentration-v1/validation_only")
FEATURES = ["rsi14", "atr14_pct", "volume_ratio", "trend_alignment", "distance_ema200_pct"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--output", type=Path, default=OUT)
    return parser.parse_args()


def bin_values(df):
    x = df.copy()
    x["atr_band"] = pd.cut(x["atr14_pct"], [-np.inf, 0.5, 1.0, 1.5, np.inf],
                           labels=["ATR_LE_0_5", "ATR_0_5_1", "ATR_1_1_5", "ATR_GT_1_5"]).astype("string").fillna("MISSING")
    x["rsi_band"] = pd.cut(x["rsi14"], [-np.inf, 30, 45, 55, 70, np.inf],
                           labels=["RSI_LE_30", "RSI_30_45", "RSI_45_55", "RSI_55_70", "RSI_GT_70"]).astype("string").fillna("MISSING")
    x["trend_band"] = np.select([x["trend_alignment"] >= 2, x["trend_alignment"] <= -2],
                                 ["BULLISH_ALIGNMENT", "BEARISH_ALIGNMENT"], default="MIXED")
    x.loc[x["trend_alignment"].isna(), "trend_band"] = "MISSING"
    x["volume_band"] = pd.cut(x["volume_ratio"], [-np.inf, 0.7, 1.5, np.inf],
                              labels=["LOW_VOLUME", "NORMAL_VOLUME", "HIGH_VOLUME"]).astype("string").fillna("MISSING")
    x["ema200_band"] = pd.cut(x["distance_ema200_pct"], [-np.inf, -2, 0, 2, np.inf],
                              labels=["BELOW_FAR", "BELOW_NEAR", "ABOVE_NEAR", "ABOVE_FAR"]).astype("string").fillna("MISSING")
    return x


def summarize(df, dims):
    rows = []
    for key, g in df.groupby(dims, observed=True, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(dims, key))
        net = g["return_net_pct"]
        gross = g["return_gross_pct"]
        record.update({
            "trades": len(g),
            "win_rate_net_pct": round(100 * (net > 0).mean(), 2),
            "mean_gross_pct": round(gross.mean(), 4),
            "mean_net_pct": round(net.mean(), 4),
            "median_net_pct": round(net.median(), 4),
            "sum_log_return_pct": round(100 * g["log_equity_return"].sum(), 4),
            "compounded_return_pct_if_sequential": round(100 * np.expm1(g["log_equity_return"].sum()), 4),
            "mean_r_proxy": round(g["r_net_proxy"].mean(), 4),
        })
        rows.append(record)
    return pd.DataFrame(rows)


def main():
    a = arguments()
    required = ["split", "strategy", "signal_time", "exit_time", "side", "reason",
                "pnl", "equity_before", "equity_after", "return_net_pct",
                "return_gross_pct", "r_net_proxy"]
    t = pd.read_csv(a.trades)
    missing = [c for c in required if c not in t.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    if t.empty or set(t["split"]) != {"VALIDATION"}:
        raise ValueError("Fichier attendu : trades.csv de VALIDATION uniquement")
    if set(t["strategy"]) != {"QWEN", "HEURISTIC"}:
        raise ValueError("Stratégies attendues : QWEN, HEURISTIC")
    for c in ["signal_time", "exit_time"]:
        t[c] = pd.to_datetime(t[c], utc=True, errors="raise")
    if t[required].isna().any().any():
        raise ValueError("Valeurs manquantes dans les colonnes essentielles")
    t["timestamp"] = t["signal_time"] - pd.Timedelta(hours=1)
    m = pd.read_parquet(MARKET, columns=["timestamp", *FEATURES])
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    if m["timestamp"].duplicated().any():
        raise ValueError("Timestamps marché dupliqués")
    t = t.merge(m, on="timestamp", how="left", validate="many_to_one", indicator=True)
    if not (t["_merge"] == "both").all():
        raise ValueError("Trade(s) sans feature au timestamp du signal")
    t = t.drop(columns="_merge")
    if t[FEATURES].isna().any().any():
        raise ValueError("Features manquantes sur les trades")
    if (t["equity_before"] <= 0).any() or (t["equity_after"] <= 0).any():
        raise ValueError("Equity non positive, logarithmes impossibles")
    if not np.allclose(t["equity_after"], t["equity_before"] + t["pnl"], atol=1e-5):
        raise ValueError("Incohérence equity / PnL")
    if not np.allclose(100 * t["pnl"] / t["equity_before"], t["return_net_pct"], atol=1e-5):
        raise ValueError("Incohérence rendement / PnL (allocation supposée à 100 %)")
    for strategy, g in t.groupby("strategy"):
        g = g.sort_values("exit_time")
        if not np.allclose(g["equity_after"].to_numpy()[:-1],
                           g["equity_before"].to_numpy()[1:], atol=1e-4):
            raise ValueError(f"{strategy} : chaîne des equities non continue")
    t["log_equity_return"] = np.log(t["equity_after"] / t["equity_before"])
    t["week_start_utc"] = (t["exit_time"].dt.tz_localize(None)
                           .dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d"))
    t = bin_values(t)
    weekly = summarize(t, ["strategy", "week_start_utc"])
    direction = summarize(t, ["strategy", "side"])
    regimes = []
    for feature in ["atr_band", "rsi_band", "trend_band", "volume_band", "ema200_band"]:
        s = summarize(t, ["strategy", "side", feature])
        s = s.rename(columns={feature: "regime_value"})
        s.insert(2, "regime_type", feature)
        regimes.append(s)
    regimes = pd.concat(regimes, ignore_index=True)
    concentration = []
    for strategy, g in t.groupby("strategy"):
        contributions = g["log_equity_return"].to_numpy()
        total = contributions.sum()
        negative = np.sort(contributions[contributions < 0])
        positive = np.sort(contributions[contributions > 0])[::-1]
        item = {"strategy": strategy, "trades": len(g),
                "compounded_total_pct": round(100 * np.expm1(total), 4),
                "sum_log_pct": round(100 * total, 4),
                "positive_trades": len(positive), "negative_trades": len(negative)}
        for k in (1, 5, 10, 20):
            item[f"best_{k}_sum_log_pct"] = round(100 * positive[:k].sum(), 4)
            item[f"worst_{k}_sum_log_pct"] = round(100 * negative[:k].sum(), 4)
        concentration.append(item)
    concentration = pd.DataFrame(concentration)
    # Mise en évidence des épisodes, sans retrait ni sélection rétrospective de trades.
    ranked = weekly.sort_values(["strategy", "sum_log_return_pct"])
    print("=" * 78)
    print("QWEN V2 - AUDIT DE CONCENTRATION - VALIDATION UNIQUEMENT")
    print("=" * 78)
    print(f"Trades vérifiés : {len(t)} ; jointure features sur la bougie de signal : OK")
    print("\nPERFORMANCE PAR DIRECTION")
    print(direction.to_string(index=False))
    print("\nPERFORMANCE HEBDOMADAIRE (selon semaine de sortie UTC)")
    print(weekly.to_string(index=False))
    print("\nCONCENTRATION DES GAINS / PERTES (contributions logarithmiques)")
    print(concentration.to_string(index=False))
    print("\n3 PIRES SEMAINES DE CHAQUE STRATEGIE")
    print(ranked.groupby("strategy", sort=True).head(3).to_string(index=False))
    print("\nREGIMES : effectifs >= 15 (descriptif, non significatif)")
    sizeable = regimes[regimes["trades"] >= 15]
    print(sizeable.to_string(index=False))
    a.output.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.output / "trades_enriched.csv", index=False)
    weekly.to_csv(a.output / "weekly_summary.csv", index=False)
    direction.to_csv(a.output / "direction_summary.csv", index=False)
    regimes.to_csv(a.output / "regime_summary.csv", index=False)
    concentration.to_csv(a.output / "concentration_summary.csv", index=False)
    print(f"\nCSV : {a.output.resolve()}")
    print("Aucun Test consulté. Pas de nouvel entraînement ni de tuning.")
    print("Regimes/semaines observés a posteriori : NE PAS en déduire des règles de trading.")
    print("PnL log additif uniquement dans l'ordre de la stratégie, drawdown intra-position absent.")


if __name__ == "__main__":
    main()

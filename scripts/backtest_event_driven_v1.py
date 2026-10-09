from __future__ import annotations

"""
Backtest événementiel de Qwen V2 versus heuristique.
Hypothèses :
- timestamp = OUVERTURE de la bougie 1H; available_at = timestamp + 1h.
- signal émis à la clôture (available_at); ordre exécuté à l'ouverture
  de la bougie suivante (timestamp == available_at).
- Un seul trade simultané par stratégie; signaux pendant position ignorés.
- TP=1.5*ATR, SL=1.0*ATR, niveaux ancrés sur le prix d'ouverture d'entrée.
- Si TP et SL sont dans la même bougie : SL prioritaire (pessimiste).
- Gap à l'ouverture après entrée : stop/target exécuté à l'open, avant
  l'examen du high/low (pas de fill artificiel au prix du stop).
- Fin d'horizon : clôture de la 12e bougie après signal.
- Fin du split : ne pas ouvrir un trade qui ne peut être suivi jusqu'à H
  dans le split; ne consulte pas les bougies d'un split suivant.
- Frais et slippage en bps PAR CÔTÉ. PnL calculé sur les prix exécutés;
  notionnel fixe = fraction de l'equity avant chaque position.
- Equity/drawdown aux sorties uniquement, pas de MTM intra-position.
- Modèle de short synthétique sans financement, borrow ni liquidation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_FILE = Path("data/processed/btc_usdc_1h_labeled.parquet")
RESULT_FILES = {
    "VALIDATION": Path(
        "data/evaluation/qwen3.5-9b-trading-v2-validation/results.parquet"
    ),
    "TEST": Path(
        "data/evaluation/qwen3.5-9b-trading-v2-fast/results_fast.parquet"
    ),
}
OUTPUT_ROOT = Path("data/evaluation/backtest-event-driven-v1")
VALID = {"NO_TRADE", "LONG_BIAS", "SHORT_BIAS"}
HOUR = pd.Timedelta(hours=1)


def options():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true",
                    help="Évaluation descriptive du Test; ne pas tuner avec ces résultats")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--fee-bps", type=float, default=5.0,
                    help="Frais par côté, points de base")
    ap.add_argument("--slippage-bps", type=float, default=2.0,
                    help="Slippage adverse par côté, points de base")
    ap.add_argument("--initial-equity", type=float, default=10000.0)
    ap.add_argument("--allocation", type=float, default=1.0,
                    help="Fraction d'equity utilisée comme notionnel [0,1]")
    a = ap.parse_args()
    if (a.horizon < 1 or a.tp_atr <= 0 or a.sl_atr <= 0
            or a.fee_bps < 0 or a.slippage_bps < 0
            or a.initial_equity <= 0 or not 0 < a.allocation <= 1):
        ap.error("Paramètres invalides")
    return a


def load_market():
    columns = ["timestamp", "available_at", "open", "high",
               "low", "close", "atr14"]
    df = pd.read_parquet(MARKET_FILE, columns=columns)
    for col in ("timestamp", "available_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].isna().any() or df["timestamp"].duplicated().any():
        raise ValueError("Timestamps du marché invalides / dupliqués")
    if not (df["available_at"] == df["timestamp"] + HOUR).all():
        raise ValueError("available_at n'est pas timestamp + 1H")
    if not (df["timestamp"].diff().dropna() == HOUR).all():
        raise ValueError("Trous détectés dans les bougies 1H")
    if not ((df["high"] >= df[["open", "close", "low"]].max(axis=1))
            & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
            & (df[["open", "high", "low", "close"]] > 0).all(axis=1)).all():
        raise ValueError("Bougies OHLC invalides")
    return df


def load_signals(split, market):
    df = pd.read_parquet(RESULT_FILES[split])
    for col in ("timestamp", "qwen_prediction", "heuristic_prediction"):
        if col not in df:
            raise ValueError(f"{split}: champ absent: {col}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        raise ValueError(f"{split}: timestamps de prédiction dupliqués")
    lookup = pd.Series(np.arange(len(market)), index=market["timestamp"])
    df["market_i"] = df["timestamp"].map(lookup)
    if df["market_i"].isna().any():
        raise ValueError(f"{split}: prédictions non jointes au marché")
    for c in ("qwen_prediction", "heuristic_prediction"):
        if not df[c].isin(VALID | {"INVALID"}).all():
            raise ValueError(f"{split}: prédictions inconnues dans {c}")
    return df


def exit_fill(price, side, slip):
    # Slippage adverse à la sortie : LONG vend moins cher, SHORT rachète plus cher.
    return price * (1 - slip if side == "LONG" else 1 + slip)


def simulate_one(signal_index, side, market, horizon, tp_atr, sl_atr,
                 fee_bps, slippage_bps, equity, allocation):
    """Ne lance l'opération que si toutes les H bougies futures existent."""
    entry_i = signal_index + 1
    end_i = signal_index + horizon
    if end_i >= len(market):
        return None
    atr = float(market.at[signal_index, "atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    raw_entry = float(market.at[entry_i, "open"])
    slip = slippage_bps / 10000.0
    fee = fee_bps / 10000.0
    entry = raw_entry * (1 + slip if side == "LONG" else 1 - slip)
    if side == "LONG":
        target, stop = raw_entry + tp_atr * atr, raw_entry - sl_atr * atr
    else:
        target, stop = raw_entry - tp_atr * atr, raw_entry + sl_atr * atr
    if stop <= 0 or target <= 0:
        return None

    for i in range(entry_i, end_i + 1):
        bar = market.iloc[i]
        o, high, low = map(float, (bar["open"], bar["high"], bar["low"]))
        if side == "LONG":
            if i > entry_i and o <= stop:
                raw_exit, reason = o, "SL_GAP"
            elif i > entry_i and o >= target:
                raw_exit, reason = o, "TP_GAP"
            elif low <= stop:
                raw_exit, reason = stop, "SL"  # inclut bougie ambiguë
            elif high >= target:
                raw_exit, reason = target, "TP"
            else:
                continue
        else:
            if i > entry_i and o >= stop:
                raw_exit, reason = o, "SL_GAP"
            elif i > entry_i and o <= target:
                raw_exit, reason = o, "TP_GAP"
            elif high >= stop:
                raw_exit, reason = stop, "SL"  # inclut bougie ambiguë
            elif low <= target:
                raw_exit, reason = target, "TP"
            else:
                continue
        exit_i = i
        break
    else:
        exit_i = end_i
        raw_exit = float(market.at[end_i, "close"])
        reason = "TIME"

    exit_price = exit_fill(raw_exit, side, slip)
    direction = 1 if side == "LONG" else -1
    gross_return = direction * (exit_price - entry) / entry
    # Frais proportionnels au notionnel à l'entrée et à la sortie.
    net_return = gross_return - fee * (1 + exit_price / entry)
    notional = equity * allocation
    pnl = notional * net_return
    new_equity = equity + pnl
    r_proxy = (direction * (exit_price - entry)
               - fee * (entry + exit_price)) / (sl_atr * atr)
    return {
        "entry_i": entry_i,
        "exit_i": exit_i,
        "signal_time": market.at[signal_index, "available_at"],
        "entry_time": market.at[entry_i, "timestamp"],
        "exit_time": market.at[exit_i, "available_at"],
        "side": side,
        "reason": reason,
        "entry_raw": raw_entry,
        "entry_fill": entry,
        "exit_raw": raw_exit,
        "exit_fill": exit_price,
        "atr_at_signal": atr,
        "tp_price": target,
        "sl_price": stop,
        "duration_bars": exit_i - entry_i + 1,
        "return_gross_pct": 100 * gross_return,
        "return_net_pct": 100 * net_return,
        "r_net_proxy": r_proxy,
        "notional": notional,
        "pnl": pnl,
        "equity_before": equity,
        "equity_after": new_equity,
    }


def run(split, strategy, signals, market, a):
    signal_col = ("qwen_prediction" if strategy == "QWEN"
                  else "heuristic_prediction")
    first_i = int(signals["market_i"].min())
    last_i = int(signals["market_i"].max())
    # Autorise un trade uniquement si ses H bougies existent DANS le split
    # (et donc dans la zone de signaux commune), même si TP/SL
    # serait éventuellement atteint plus tôt.
    max_signal_i = last_i - a.horizon
    equity, trades = a.initial_equity, []
    busy_until_i = -1
    rejected_busy = invalid = truncated = invalid_atr = 0
    for row in signals.itertuples(index=False):
        i = int(row.market_i)
        prediction = getattr(row, signal_col)
        if prediction == "NO_TRADE":
            continue
        if prediction == "INVALID":
            invalid += 1
            continue
        if i <= busy_until_i:
            rejected_busy += 1
            continue
        if i > max_signal_i:
            truncated += 1
            continue
        side = "LONG" if prediction == "LONG_BIAS" else "SHORT"
        result = simulate_one(
            i, side, market, a.horizon, a.tp_atr, a.sl_atr,
            a.fee_bps, a.slippage_bps, equity, a.allocation
        )
        if result is None:
            invalid_atr += 1
            continue
        if result["exit_i"] > last_i:
            raise RuntimeError("La position sort du split")
        result.update({"split": split, "strategy": strategy})
        equity = result["equity_after"]
        trades.append(result)
        busy_until_i = result["exit_i"]
        if equity <= 0:
            raise RuntimeError("Capital <= 0 : simulation interrompue")
    result_df = pd.DataFrame(trades)
    if not len(result_df):
        summary = {
            "split": split, "strategy": strategy, "trades": 0,
            "ending_equity": equity, "return_pct": 0.0,
            "max_drawdown_exit_only_pct": 0.0,
            "win_rate_net_pct": None, "mean_net_r": None,
        }
    else:
        path = np.r_[a.initial_equity, result_df["equity_after"].to_numpy()]
        peaks = np.maximum.accumulate(path)
        dd = path / peaks - 1
        summary = {
            "split": split, "strategy": strategy,
            "trades": len(result_df),
            "long_trades": int((result_df["side"] == "LONG").sum()),
            "short_trades": int((result_df["side"] == "SHORT").sum()),
            "ending_equity": round(equity, 2),
            "return_pct": round(100 * (equity / a.initial_equity - 1), 4),
            "max_drawdown_exit_only_pct": round(100 * dd.min(), 4),
            "win_rate_net_pct": round(
                100 * (result_df["pnl"] > 0).mean(), 4
            ),
            "mean_net_r": round(result_df["r_net_proxy"].mean(), 4),
            "median_duration_bars": float(
                result_df["duration_bars"].median()
            ),
            "total_fees_plus_slippage_modeled": True,
        }
    summary.update({
        "ignored_busy_signals": rejected_busy,
        "ignored_invalid_signals": invalid,
        "ignored_end_of_split": truncated,
        "ignored_invalid_atr": invalid_atr,
        "first_signal_timestamp": str(signals["timestamp"].min()),
        "last_signal_timestamp": str(signals["timestamp"].max()),
    })
    return result_df, summary


def main():
    a = options()
    market = load_market()
    splits = ["VALIDATION", "TEST"] if a.include_test else ["VALIDATION"]
    out = OUTPUT_ROOT / (
        "validation_and_test" if a.include_test else "validation_only"
    )
    summaries, trades = [], []
    print("=" * 76)
    print("BACKTEST EVENT-DRIVEN V1 | QWEN V2 vs HEURISTIC")
    print("=" * 76)
    print("Hypothèses :", json.dumps({
        "horizon": a.horizon, "tp_atr": a.tp_atr,
        "sl_atr": a.sl_atr, "fee_bps_per_side": a.fee_bps,
        "slippage_bps_per_side": a.slippage_bps,
        "allocation": a.allocation,
        "initial_equity": a.initial_equity,
    }, ensure_ascii=False))
    for split in splits:
        signals = load_signals(split, market)
        print(f"\n{split}: {len(signals)} signaux")
        for strategy in ("QWEN", "HEURISTIC"):
            df, summary = run(split, strategy, signals, market, a)
            trades.append(df)
            summaries.append(summary)
            print(
                f"  {strategy:10s} trades={summary['trades']:4d}"
                f"  final={summary['ending_equity']:11,.2f}"
                f"  retour={summary['return_pct']:+9.3f}%"
                f"  maxDD(exit)={summary['max_drawdown_exit_only_pct']:+8.3f}%"
                f"  ignorés(busy)={summary['ignored_busy_signals']}"
            )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    all_trades = [t for t in trades if len(t)]
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            out / "trades.csv", index=False
        )
    (out / "settings.json").write_text(
        json.dumps(vars(a), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nSorties : {out.resolve()}")
    print("ATTENTION : backtest simplifié, pas de mark-to-market intra-trade,")
    print("pas de funding/borrow, ni données carnet, ni fill garanti aux seuils.")
    print("Comparaison uniquement ; ne pas optimiser les paramètres sur Test.")


if __name__ == "__main__":
    main()

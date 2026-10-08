from pathlib import Path

import pandas as pd


FILE = Path(
    "data/evaluation/"
    "qwen3.5-9b-trading-v2-fast/"
    "results_fast.parquet"
)

TP_R = 1.5
SL_R = 1.0


def analyze_side(df, prediction, outcome_column):

    trades = df[
        df["qwen_prediction"] == prediction
    ].copy()

    outcomes = trades[
        outcome_column
    ]

    wins = int(
        (outcomes == 1).sum()
    )

    losses = int(
        (outcomes == -1).sum()
    )

    unresolved = int(
        (outcomes == 0).sum()
    )

    resolved = wins + losses

    total_r = (
        wins * TP_R
        - losses * SL_R
    )

    win_rate = (
        wins / resolved
        if resolved
        else 0
    )

    expected_r_all = (
        total_r / len(trades)
        if len(trades)
        else 0
    )

    expected_r_resolved = (
        total_r / resolved
        if resolved
        else 0
    )

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolved": resolved,
        "win_rate": win_rate,
        "total_r": total_r,
        "expected_r_all": expected_r_all,
        "expected_r_resolved": expected_r_resolved,
    }


def print_side(name, stats):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Trades        :",
        stats["trades"]
    )

    print(
        "Wins          :",
        stats["wins"]
    )

    print(
        "Losses        :",
        stats["losses"]
    )

    print(
        "Non résolus   :",
        stats["unresolved"]
    )

    print(
        "Win rate      :",
        f"{stats['win_rate'] * 100:.2f}%"
    )

    print(
        "Expected R    :",
        f"{stats['expected_r_all']:+.4f} R/trade"
    )

    print(
        "Expected R resolved :",
        f"{stats['expected_r_resolved']:+.4f} R/trade"
    )

    print(
        "Total R       :",
        f"{stats['total_r']:+.2f} R"
    )


def main():

    df = pd.read_parquet(
        FILE
    )

    long_stats = analyze_side(
        df,
        "LONG_BIAS",
        "long_outcome_12h",
    )

    short_stats = analyze_side(
        df,
        "SHORT_BIAS",
        "short_outcome_12h",
    )

    print_side(
        "QWEN V2 - LONG ONLY",
        long_stats,
    )

    print_side(
        "QWEN V2 - SHORT ONLY",
        short_stats,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_FILE = Path(
    "data/evaluation/"
    "qwen3.5-9b-trading-v2-validation/"
    "results.parquet"
)

TEST_FILE = Path(
    "data/evaluation/"
    "qwen3.5-9b-trading-v2-fast/"
    "results_fast.parquet"
)

OUTPUT_DIR = Path(
    "data/evaluation/"
    "qwen3.5-9b-trading-v2-analysis"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "value_added_summary.csv"
)

MONTHLY_FILE = (
    OUTPUT_DIR / "monthly_performance.csv"
)


TP_R = 1.5
SL_R = 1.0

BREAKEVEN_WIN_RATE = (
    SL_R / (TP_R + SL_R)
)


# ============================================================
# WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    wins: int,
    losses: int,
    z: float = 1.96,
):

    n = wins + losses

    if n == 0:
        return 0.0, 0.0

    p = wins / n

    denominator = (
        1
        + z**2 / n
    )

    centre = (
        p
        + z**2 / (2 * n)
    )

    margin = (
        z
        * math.sqrt(
            (
                p * (1 - p)
                + z**2 / (4 * n)
            )
            / n
        )
    )

    lower = (
        centre - margin
    ) / denominator

    upper = (
        centre + margin
    ) / denominator

    return lower, upper


# ============================================================
# PERFORMANCE D'UN SOUS-ENSEMBLE
# ============================================================

def calculate_stats(
    df: pd.DataFrame,
    mask,
    outcome_column: str,
):

    selected = df[
        mask
    ].copy()

    if len(selected) == 0:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "resolved": 0,
            "win_rate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "expected_r": 0.0,
            "expected_r_resolved": 0.0,
            "total_r": 0.0,
        }


    outcomes = selected[
        outcome_column
    ]


    wins = int(
        (outcomes == 1).sum()
    )

    losses = int(
        (outcomes == -1).sum()
    )

    unresolved = int(
        (
            (outcomes == 0)
            |
            outcomes.isna()
        ).sum()
    )


    resolved = (
        wins
        + losses
    )


    win_rate = (
        wins / resolved
        if resolved
        else 0.0
    )


    total_r = (
        wins * TP_R
        - losses * SL_R
    )


    expected_r = (
        total_r
        / len(selected)
    )


    expected_r_resolved = (
        total_r
        / resolved
        if resolved
        else 0.0
    )


    ci_low, ci_high = (
        wilson_interval(
            wins,
            losses,
        )
    )


    return {
        "trades":
            int(len(selected)),

        "wins":
            wins,

        "losses":
            losses,

        "unresolved":
            unresolved,

        "resolved":
            resolved,

        "win_rate":
            float(win_rate),

        "ci95_low":
            float(ci_low),

        "ci95_high":
            float(ci_high),

        "expected_r":
            float(expected_r),

        "expected_r_resolved":
            float(expected_r_resolved),

        "total_r":
            float(total_r),
    }


# ============================================================
# AFFICHAGE
# ============================================================

def print_stats(
    name,
    stats,
):

    print()
    print("-" * 72)
    print(name)
    print("-" * 72)

    print(
        f"Trades       : "
        f"{stats['trades']}"
    )

    print(
        f"W / L / NR   : "
        f"{stats['wins']} / "
        f"{stats['losses']} / "
        f"{stats['unresolved']}"
    )

    print(
        f"Win rate     : "
        f"{stats['win_rate'] * 100:.2f}%"
    )

    print(
        f"IC95 winrate : "
        f"["
        f"{stats['ci95_low'] * 100:.2f}% ; "
        f"{stats['ci95_high'] * 100:.2f}%"
        f"]"
    )

    print(
        f"Expected R   : "
        f"{stats['expected_r']:+.4f} R/trade"
    )

    print(
        f"ER résolu    : "
        f"{stats['expected_r_resolved']:+.4f} R/trade"
    )

    print(
        f"Total R      : "
        f"{stats['total_r']:+.2f} R"
    )


# ============================================================
# AJOUT AU SUMMARY
# ============================================================

def add_summary(
    rows,
    split,
    side,
    group,
    stats,
):

    row = {
        "split":
            split,

        "side":
            side,

        "group":
            group,
    }

    row.update(
        stats
    )

    rows.append(
        row
    )


# ============================================================
# ANALYSE D'UN SPLIT
# ============================================================

def analyze_split(
    df: pd.DataFrame,
    split_name: str,
    summary_rows: list,
):

    print()
    print("=" * 72)
    print(
        split_name.upper()
    )
    print("=" * 72)


    # ========================================================
    # LONG
    # ========================================================

    qwen_long = (
        df["qwen_prediction"]
        == "LONG_BIAS"
    )

    heur_long = (
        df["heuristic_prediction"]
        == "LONG_BIAS"
    )


    groups_long = {

        "QWEN_LONG_ALL":
            qwen_long,

        "HEUR_LONG_ALL":
            heur_long,

        "BOTH_LONG":
            qwen_long
            & heur_long,

        "QWEN_LONG_ONLY":
            qwen_long
            & ~heur_long,

        "HEUR_LONG_ONLY":
            heur_long
            & ~qwen_long,
    }


    print()
    print(
        "### LONG"
    )


    for name, mask in (
        groups_long.items()
    ):

        stats = calculate_stats(
            df,
            mask,
            "long_outcome_12h",
        )

        print_stats(
            name,
            stats,
        )

        add_summary(
            summary_rows,
            split_name,
            "LONG",
            name,
            stats,
        )


    # ========================================================
    # SHORT
    # ========================================================

    qwen_short = (
        df["qwen_prediction"]
        == "SHORT_BIAS"
    )

    heur_short = (
        df["heuristic_prediction"]
        == "SHORT_BIAS"
    )


    groups_short = {

        "QWEN_SHORT_ALL":
            qwen_short,

        "HEUR_SHORT_ALL":
            heur_short,

        "BOTH_SHORT":
            qwen_short
            & heur_short,

        "QWEN_SHORT_ONLY":
            qwen_short
            & ~heur_short,

        "HEUR_SHORT_ONLY":
            heur_short
            & ~qwen_short,
    }


    print()
    print(
        "### SHORT"
    )


    for name, mask in (
        groups_short.items()
    ):

        stats = calculate_stats(
            df,
            mask,
            "short_outcome_12h",
        )

        print_stats(
            name,
            stats,
        )

        add_summary(
            summary_rows,
            split_name,
            "SHORT",
            name,
            stats,
        )


# ============================================================
# PERFORMANCE MENSUELLE
# ============================================================

def monthly_analysis(
    df: pd.DataFrame,
    split_name: str,
):

    timestamp_column = None

    for candidate in [
        "timestamp",
        "open_time",
        "datetime",
        "date",
    ]:

        if candidate in df.columns:

            timestamp_column = (
                candidate
            )

            break


    if timestamp_column is None:

        print()
        print(
            f"{split_name}: "
            "pas de colonne timestamp, "
            "analyse mensuelle ignorée."
        )

        return []


    temp = df.copy()


    temp[
        timestamp_column
    ] = pd.to_datetime(
        temp[
            timestamp_column
        ],
        errors="coerce",
        utc=True,
    )


    temp["month"] = (
        temp[
            timestamp_column
        ]
        .dt.strftime(
            "%Y-%m"
        )
    )


    rows = []


    for month in sorted(
        temp["month"]
        .dropna()
        .unique()
    ):

        month_df = temp[
            temp["month"] == month
        ]


        for (
            side,
            prediction,
            outcome,
        ) in [

            (
                "LONG",
                "LONG_BIAS",
                "long_outcome_12h",
            ),

            (
                "SHORT",
                "SHORT_BIAS",
                "short_outcome_12h",
            ),
        ]:


            mask = (
                month_df[
                    "qwen_prediction"
                ]
                == prediction
            )


            stats = calculate_stats(
                month_df,
                mask,
                outcome,
            )


            row = {
                "split":
                    split_name,

                "month":
                    month,

                "side":
                    side,
            }

            row.update(
                stats
            )

            rows.append(
                row
            )


    return rows


# ============================================================
# COMBINED LONG
# ============================================================

def combined_long_analysis(
    validation,
    test,
):

    validation = (
        validation.copy()
    )

    test = (
        test.copy()
    )


    validation[
        "_split"
    ] = "VALIDATION"

    test[
        "_split"
    ] = "TEST"


    combined = pd.concat(
        [
            validation,
            test,
        ],

        ignore_index=True,
    )


    qwen_long = (
        combined[
            "qwen_prediction"
        ]
        == "LONG_BIAS"
    )


    stats = calculate_stats(

        combined,

        qwen_long,

        "long_outcome_12h",
    )


    print()
    print("=" * 72)
    print(
        "QWEN LONG - VALIDATION + TEST"
    )
    print("=" * 72)


    print_stats(
        "COMBINED LONG",
        stats,
    )


    print()

    print(
        "Break-even théorique : "
        f"{BREAKEVEN_WIN_RATE * 100:.2f}%"
    )


    if (
        stats["ci95_low"]
        >
        BREAKEVEN_WIN_RATE
    ):

        print(
            "IC95 entièrement au-dessus "
            "du break-even."
        )

    else:

        print(
            "ATTENTION : le break-even "
            "est encore contenu dans "
            "l'intervalle de confiance."
        )


    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "QWEN V2 - VALUE ADDED / STABILITY ANALYSIS"
    )
    print("=" * 72)


    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            VALIDATION_FILE
        )


    if not TEST_FILE.exists():

        raise FileNotFoundError(
            TEST_FILE
        )


    validation = pd.read_parquet(
        VALIDATION_FILE
    )


    test = pd.read_parquet(
        TEST_FILE
    )


    print(
        "Validation :",
        len(validation),
        "exemples"
    )

    print(
        "Test       :",
        len(test),
        "exemples"
    )


    required = [
        "qwen_prediction",
        "heuristic_prediction",
        "long_outcome_12h",
        "short_outcome_12h",
    ]


    for name, df in [
        (
            "validation",
            validation,
        ),

        (
            "test",
            test,
        ),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise RuntimeError(
                f"{name}: colonnes "
                f"manquantes : {missing}"
            )


    # ========================================================
    # ANALYSES
    # ========================================================

    summary_rows = []


    analyze_split(
        validation,
        "VALIDATION",
        summary_rows,
    )


    analyze_split(
        test,
        "TEST",
        summary_rows,
    )


    combined_long = (
        combined_long_analysis(
            validation,
            test,
        )
    )


    # ========================================================
    # MONTHLY
    # ========================================================

    monthly_rows = []


    monthly_rows.extend(
        monthly_analysis(
            validation,
            "VALIDATION",
        )
    )


    monthly_rows.extend(
        monthly_analysis(
            test,
            "TEST",
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    if monthly_rows:

        monthly_df = pd.DataFrame(
            monthly_rows
        )

        monthly_df.to_csv(
            MONTHLY_FILE,
            index=False,
        )


        print()
        print("=" * 72)
        print(
            "PERFORMANCE QWEN PAR MOIS"
        )
        print("=" * 72)


        display_columns = [
            "split",
            "month",
            "side",
            "trades",
            "win_rate",
            "expected_r",
            "total_r",
        ]


        print(
            monthly_df[
                display_columns
            ].to_string(
                index=False,
                formatters={
                    "win_rate":
                        lambda x:
                        f"{x * 100:.2f}%",

                    "expected_r":
                        lambda x:
                        f"{x:+.4f}",

                    "total_r":
                        lambda x:
                        f"{x:+.2f}",
                },
            )
        )


    print()
    print("=" * 72)
    print("FICHIERS")
    print("=" * 72)

    print(
        SUMMARY_FILE.resolve()
    )


    if monthly_rows:

        print(
            MONTHLY_FILE.resolve()
        )


if __name__ == "__main__":

    main()
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/sft")

SPLITS = [
    "train",
    "validation",
    "test",
]

SYMBOL = "BTCUSDC"
TIMEFRAME = "1H"


SYSTEM_PROMPT = """
Tu es un analyste spécialisé dans le trading crypto.

Tu analyses uniquement les informations de marché fournies.
Tu ne dois jamais inventer une donnée absente.

Tu privilégies :
- la structure de marché,
- le contexte multi-timeframe,
- le momentum,
- la volatilité,
- le volume,
- la gestion du risque,
- et NO_TRADE lorsqu'aucun avantage clair n'existe.

Tu produis une analyse conditionnelle.
Tu ne passes jamais d'ordre.
""".strip()


# ============================================================
# UTILITAIRES
# ============================================================

def safe_float(value, digits=2):
    if pd.isna(value):
        return None

    return round(
        float(value),
        digits,
    )


def trend_name(value):
    if pd.isna(value):
        return "INCONNU"

    value = int(value)

    if value > 0:
        return "HAUSSIER"

    if value < 0:
        return "BAISSIER"

    return "NEUTRE"


def yes_no(value):
    if pd.isna(value):
        return "NON"

    return (
        "OUI"
        if int(value) == 1
        else "NON"
    )


# ============================================================
# SCORE DE CONTEXTE OBSERVABLE
# ============================================================

def market_bias_score(row):
    """
    Utilise UNIQUEMENT des données connues à l'instant T.

    Aucun label futur ici.
    """

    score = 0

    # --------------------------------------------------------
    # Tendances multi-timeframe
    # --------------------------------------------------------

    for column in [
        "trend_1h",
        "4h_trend",
        "1d_trend",
    ]:

        value = row[column]

        if pd.notna(value):
            score += int(value)


    # --------------------------------------------------------
    # Position par rapport EMA20
    # --------------------------------------------------------

    if row["close"] > row["ema20"]:
        score += 1

    elif row["close"] < row["ema20"]:
        score -= 1


    # --------------------------------------------------------
    # EMA20 vs EMA50
    # --------------------------------------------------------

    if row["ema20"] > row["ema50"]:
        score += 1

    elif row["ema20"] < row["ema50"]:
        score -= 1


    # --------------------------------------------------------
    # RSI léger
    # --------------------------------------------------------

    rsi = row["rsi14"]

    if pd.notna(rsi):

        if 52 <= rsi <= 70:
            score += 1

        elif 30 <= rsi <= 48:
            score -= 1


    return score


# ============================================================
# LABEL FINAL
# ============================================================

def determine_decision(row):
    """
    La direction finale dépend :

    1. du contexte observable à T
    2. du résultat futur 12h

    Le futur sert UNIQUEMENT à construire la cible SFT.
    Il ne sera jamais présent dans le prompt.
    """

    long_outcome = row[
        "long_outcome_12h"
    ]

    short_outcome = row[
        "short_outcome_12h"
    ]


    # Cas OHLC ambigu :
    # TP et SL touchés dans la même bougie.
    if (
        pd.isna(long_outcome)
        or pd.isna(short_outcome)
    ):
        return None


    score = market_bias_score(
        row
    )


    # ========================================================
    # LONG
    # ========================================================

    if (
        score >= 2
        and long_outcome == 1
        and short_outcome != 1
    ):
        return "LONG_BIAS"


    # ========================================================
    # SHORT
    # ========================================================

    if (
        score <= -2
        and short_outcome == 1
        and long_outcome != 1
    ):
        return "SHORT_BIAS"


    # ========================================================
    # SINON
    # ========================================================

    return "NO_TRADE"


# ============================================================
# CONTEXTE TEXTUEL
# ============================================================

def build_context_comments(row):
    comments = []


    # --------------------------------------------------------
    # MULTI TIMEFRAME
    # --------------------------------------------------------

    t1 = trend_name(
        row["trend_1h"]
    )

    t4 = trend_name(
        row["4h_trend"]
    )

    td = trend_name(
        row["1d_trend"]
    )


    if t1 == t4 == td:
        comments.append(
            f"Les tendances 1H, 4H et 1D sont alignées {t1.lower()}."
        )

    else:
        comments.append(
            f"Le contexte multi-timeframe est mixte : "
            f"1H={t1}, 4H={t4}, 1D={td}."
        )


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = row["rsi14"]

    if rsi >= 70:
        comments.append(
            "Le RSI indique une extension haussière importante."
        )

    elif rsi <= 30:
        comments.append(
            "Le RSI indique une extension baissière importante."
        )

    elif rsi >= 55:
        comments.append(
            "Le momentum RSI reste positif."
        )

    elif rsi <= 45:
        comments.append(
            "Le momentum RSI reste négatif."
        )

    else:
        comments.append(
            "Le RSI reste relativement neutre."
        )


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ratio = row[
        "volume_ratio"
    ]

    if volume_ratio >= 1.5:
        comments.append(
            "Le volume est nettement supérieur à sa moyenne récente."
        )

    elif volume_ratio <= 0.7:
        comments.append(
            "Le volume est inférieur à sa moyenne récente."
        )

    else:
        comments.append(
            "Le volume reste proche de sa moyenne récente."
        )


    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    if int(row["breakout_24h"]) == 1:
        comments.append(
            "Le prix clôture au-dessus du plus haut précédent des dernières 24 heures."
        )

    elif int(row["breakdown_24h"]) == 1:
        comments.append(
            "Le prix clôture sous le plus bas précédent des dernières 24 heures."
        )


    # --------------------------------------------------------
    # VOLATILITE / ATR
    # --------------------------------------------------------

    atr = row["atr14_pct"]

    if atr >= 1.5:
        comments.append(
            "La volatilité mesurée par l'ATR est élevée."
        )

    elif atr <= 0.5:
        comments.append(
            "La volatilité mesurée par l'ATR est faible."
        )


    return comments


# ============================================================
# PROMPT UTILISATEUR
# ============================================================

def build_user_prompt(row):
    return f"""
{SYMBOL} | {TIMEFRAME}

PRICE
Close: {safe_float(row["close"], 2)}

TREND
1H: {trend_name(row["trend_1h"])}
4H: {trend_name(row["4h_trend"])}
1D: {trend_name(row["1d_trend"])}

MOVING AVERAGES
EMA20: {safe_float(row["ema20"], 2)}
EMA50: {safe_float(row["ema50"], 2)}
EMA200: {safe_float(row["ema200"], 2)}

Distance EMA20: {safe_float(row["distance_ema20_pct"], 3)}%
Distance EMA50: {safe_float(row["distance_ema50_pct"], 3)}%
Distance EMA200: {safe_float(row["distance_ema200_pct"], 3)}%

MOMENTUM
RSI14: {safe_float(row["rsi14"], 2)}

VOLATILITY
ATR14: {safe_float(row["atr14_pct"], 3)}%
Volatility 24H: {safe_float(row["volatility_24h"], 5)}
Volatility 72H: {safe_float(row["volatility_72h"], 5)}

VOLUME
Volume ratio / SMA20: {safe_float(row["volume_ratio"], 3)}

PRICE STRUCTURE
Previous 24H high: {safe_float(row["prev_high_24h"], 2)}
Previous 24H low: {safe_float(row["prev_low_24h"], 2)}

Distance 24H high: {safe_float(row["distance_high_24h_pct"], 3)}%
Distance 24H low: {safe_float(row["distance_low_24h_pct"], 3)}%

24H breakout: {yes_no(row["breakout_24h"])}
24H breakdown: {yes_no(row["breakdown_24h"])}

RETURNS
1H: {safe_float(row["return_1h"] * 100, 3)}%
3H: {safe_float(row["return_3h"] * 100, 3)}%
6H: {safe_float(row["return_6h"] * 100, 3)}%
24H: {safe_float(row["return_24h"] * 100, 3)}%

Analyse cette situation.

Réponds avec exactement les sections :
BIAS
CONTEXT
DECISION
RISK
""".strip()


# ============================================================
# REPONSE CIBLE
# ============================================================

def build_assistant_response(
    row,
    decision,
):
    comments = build_context_comments(
        row
    )

    score = market_bias_score(
        row
    )


    # ========================================================
    # BIAS OBSERVABLE
    # ========================================================

    if score >= 2:
        bias = "HAUSSIER"

    elif score <= -2:
        bias = "BAISSIER"

    else:
        bias = "NEUTRE"


    context = "\n".join(
        f"- {comment}"
        for comment in comments
    )


    # ========================================================
    # DECISION
    # ========================================================

    if decision == "LONG_BIAS":

        decision_text = (
            "LONG_BIAS\n"
            "Le contexte autorise la recherche d'un scénario long, "
            "mais uniquement après confirmation."
        )

        risk_text = (
            "Référence de risque : stop maximal autour de 1 ATR "
            "et objectif minimal autour de 1.5 ATR. "
            "Ne pas poursuivre le prix si l'entrée devient trop étendue."
        )


    elif decision == "SHORT_BIAS":

        decision_text = (
            "SHORT_BIAS\n"
            "Le contexte autorise la recherche d'un scénario short, "
            "mais uniquement après confirmation."
        )

        risk_text = (
            "Référence de risque : stop maximal autour de 1 ATR "
            "et objectif minimal autour de 1.5 ATR. "
            "Ne pas poursuivre le prix après une extension importante."
        )


    else:

        decision_text = (
            "NO_TRADE\n"
            "Aucun avantage directionnel suffisamment clair "
            "n'est validé dans ce contexte."
        )

        risk_text = (
            "Préserver le capital et attendre une configuration "
            "offrant un meilleur rapport entre confirmation, "
            "invalidation et rendement potentiel."
        )


    return f"""
BIAS
{bias}

CONTEXT
{context}

DECISION
{decision_text}

RISK
{risk_text}
""".strip()


# ============================================================
# CONVERSION D'UN SPLIT
# ============================================================

def convert_split(
    split_name,
):
    input_path = (
        INPUT_DIR
        / f"{split_name}.parquet"
    )

    output_path = (
        OUTPUT_DIR
        / f"{split_name}.jsonl"
    )

    audit_path = (
        OUTPUT_DIR
        / f"{split_name}_audit.parquet"
    )


    print()
    print("=" * 70)
    print(split_name.upper())
    print("=" * 70)

    df = pd.read_parquet(
        input_path
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


    records = []
    audit_rows = []


    for _, row in df.iterrows():

        decision = determine_decision(
            row
        )

        # Cas ambigu OHLC.
        if decision is None:
            continue


        user_prompt = build_user_prompt(
            row
        )

        assistant_response = (
            build_assistant_response(
                row,
                decision,
            )
        )


        record = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
                {
                    "role": "assistant",
                    "content": assistant_response,
                },
            ]
        }


        records.append(
            record
        )


        # ----------------------------------------------------
        # Fichier d'audit séparé.
        #
        # Celui-ci contient les labels futurs,
        # mais ne sera JAMAIS fourni à Qwen.
        # ----------------------------------------------------

        audit_rows.append({
            "timestamp":
                row["timestamp"],

            "decision":
                decision,

            "bias_score":
                market_bias_score(row),

            "long_outcome_12h":
                row["long_outcome_12h"],

            "short_outcome_12h":
                row["short_outcome_12h"],

            "future_return_12h_pct":
                row["future_return_12h_pct"],

            "up_move_12h_atr":
                row["up_move_12h_atr"],

            "down_move_12h_atr":
                row["down_move_12h_atr"],
        })


    # ========================================================
    # SAVE JSONL
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


    # ========================================================
    # SAVE AUDIT
    # ========================================================

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_parquet(
        audit_path,
        index=False,
        compression="zstd",
    )


    # ========================================================
    # STATS
    # ========================================================

    print(
        "Exemples SFT :",
        f"{len(records):,}",
    )

    print()

    distribution = (
        audit["decision"]
        .value_counts()
    )

    for label, count in distribution.items():

        percentage = (
            count
            / len(audit)
            * 100
        )

        print(
            f"{label:12} "
            f"{count:5,} "
            f"({percentage:5.2f}%)"
        )


    print()
    print(
        "JSONL :",
        output_path,
    )

    print(
        "Audit :",
        audit_path,
    )


    return len(records)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CONSTRUCTION DATASET SFT")
    print("=" * 70)

    totals = {}

    for split_name in SPLITS:

        totals[
            split_name
        ] = convert_split(
            split_name
        )


    print()
    print("=" * 70)
    print("TERMINE")
    print("=" * 70)

    for name, count in totals.items():

        print(
            f"{name:12}: "
            f"{count:,}"
        )


if __name__ == "__main__":
    main()
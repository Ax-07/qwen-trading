from __future__ import annotations
"""Diagnostic descriptif des simulations V3.1 existantes (Train/Validation seulement).

Usage: python scripts/diagnose_structural_tp_v31.py
Ne recalcule ni entrées ni sorties, n'accède pas à Test, n'optimise aucun paramètre.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation")
METHODS = (("fractal", "1h"), ("fractal", "4h"), ("zigzag", "1h"), ("zigzag", "4h"))
MULTIPLES = (1., 1.5, 2., 3., 4., 5.)
REQUIRED = {"method", "timeframe", "multiple_r", "event_id", "side", "known_at",
            "eligibility", "portfolio_status", "reason", "net_return", "net_r",
            "gross_return", "duration_bars", "entry_at", "exit_at", "equity_before",
            "equity_after", "audit_status"}


def validate(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{split}: colonnes manquantes: {sorted(missing)}")
    f = frame.copy()
    for c in ("known_at", "entry_at", "exit_at"):
        f[c] = pd.to_datetime(f[c], utc=True, errors="raise")
    if f.duplicated(["method", "timeframe", "multiple_r", "event_id"]).any():
        raise ValueError(f"{split}: duplications de scenario")
    if set(zip(f.method, f.timeframe)) != set(METHODS):
        raise ValueError(f"{split}: configurations manquantes ou inattendues")
    if set(f.multiple_r) != set(MULTIPLES):
        raise ValueError(f"{split}: multiples R manquants ou inattendus")
    if not f.side.isin(["LONG", "SHORT"]).all():
        raise ValueError(f"{split}: side incoherent")
    if not f.portfolio_status.isin(["NOT_EVALUATED", "TAKEN", "SKIPPED_OVERLAP"]).all():
        raise ValueError(f"{split}: portfolio_status inattendu")
    eligible = f.eligibility.eq("ELIGIBLE")
    if ((f.portfolio_status.isin(["TAKEN", "SKIPPED_OVERLAP"])) & ~eligible).any():
        raise ValueError(f"{split}: portefeuille contient des non-eligibles")
    if f.loc[eligible, ["net_return", "gross_return", "net_r", "duration_bars", "entry_at", "exit_at"]].isna().any().any():
        raise ValueError(f"{split}: issue manquante pour eligible")
    if (~f.loc[eligible, "reason"].astype(str).str.match(r"^(TP|SL|TIME)(_|$)")).any():
        raise ValueError(f"{split}: reason inconnu")
    if f.loc[eligible, "duration_bars"].le(0).any() or f.loc[eligible, "duration_bars"].gt(12).any():
        raise ValueError(f"{split}: duree hors horizon verrouille 12h")
    for (method, tf), g in f.groupby(["method", "timeframe"]):
        ids = None
        for m, h in g.groupby("multiple_r"):
            here = set(h.event_id)
            if ids is not None and ids != here:
                raise ValueError(f"{split}/{method}/{tf}: ensemble candidats change selon R")
            ids = here
    return f


def pct(x):
    return float(x * 100) if pd.notna(x) else None


def stat(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {"mean": float(x.mean()), "median": float(x.median()),
            "p10": float(x.quantile(.1)), "p90": float(x.quantile(.9))}


def breakdown(f: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for (method, tf, multiple, side), group in f.groupby(["method", "timeframe", "multiple_r", "side"], sort=True):
        elig = group.loc[group.eligibility.eq("ELIGIBLE")]
        taken = group.loc[group.portfolio_status.eq("TAKEN")].sort_values("entry_at")
        causes = elig.reason.fillna("").str.extract(r"^(TP|SL|TIME)", expand=False).value_counts()
        taken_causes = taken.reason.fillna("").str.extract(r"^(TP|SL|TIME)", expand=False).value_counts()
        # Croissance composée directionnelle illustrative: les trades LONG/SHORT
        # sont une sous-séquence du portefeuille mixte, PAS un portefeuille recalculé.
        side_product = float(np.prod(1 + taken.net_return.to_numpy(dtype=float)) - 1) if len(taken) else 0.
        durations = stat(taken.duration_bars)
        rows.append(dict(split=split, method=method, timeframe=tf, multiple_r=float(multiple), side=side,
            candidates=len(group), eligible=len(elig), taken=len(taken),
            skipped_overlap=int(group.portfolio_status.eq("SKIPPED_OVERLAP").sum()),
            eligible_net_mean_pct=pct(elig.net_return.mean()),
            eligible_net_median_pct=pct(elig.net_return.median()),
            taken_net_mean_pct=pct(taken.net_return.mean()),
            taken_net_r_mean=float(taken.net_r.mean()) if len(taken) else None,
            direction_subset_compounded_pct=pct(side_product),
            eligible_tp=int(causes.get("TP", 0)), eligible_sl=int(causes.get("SL", 0)), eligible_time=int(causes.get("TIME", 0)),
            taken_tp=int(taken_causes.get("TP", 0)), taken_sl=int(taken_causes.get("SL", 0)), taken_time=int(taken_causes.get("TIME", 0)),
            taken_duration_mean_h=durations["mean"], taken_duration_median_h=durations["median"],
            taken_duration_p90_h=durations["p90"],
            # gross_return est APRES slippage, mais AVANT frais (selon simulateur).
            eligible_fee_effect_mean_bps=float((elig.gross_return - elig.net_return).mean() * 10000) if len(elig) else None,
            taken_fee_effect_mean_bps=float((taken.gross_return - taken.net_return).mean() * 10000) if len(taken) else None,
        ))
    return pd.DataFrame(rows)


def common_comparison(f: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    paired = []
    reference = ("zigzag", "4h")
    for multiple, x in f.groupby("multiple_r", sort=True):
        eligible_sets = {
            (method, tf): set(g.loc[g.eligibility.eq("ELIGIBLE"), "event_id"])
            for (method, tf), g in x.groupby(["method", "timeframe"])
        }
        shared = set.intersection(*(eligible_sets[m] for m in METHODS))
        for method, tf in METHODS:
            y = x.loc[(x.method == method) & (x.timeframe == tf) & x.event_id.isin(shared)]
            if len(y) != len(shared) or not y.eligibility.eq("ELIGIBLE").all():
                raise ValueError("Comparaison sur candidats communs incoherente")
            for side, z in y.groupby("side"):
                summaries.append(dict(split=split, multiple_r=float(multiple), method=method,
                    timeframe=tf, side=side, common_candidates=len(z),
                    net_mean_pct=pct(z.net_return.mean()), net_median_pct=pct(z.net_return.median()),
                    tp_rate_pct=pct(z.reason.str.startswith("TP").mean()),
                    sl_rate_pct=pct(z.reason.str.startswith("SL").mean()),
                    time_rate_pct=pct(z.reason.eq("TIME").mean())))
        ref = x.loc[(x.method == reference[0]) & (x.timeframe == reference[1]) & x.event_id.isin(shared),
                    ["event_id", "side", "net_return"]].rename(columns={"net_return": "reference_return"})
        for method, tf in METHODS:
            if (method, tf) == reference:
                continue
            comp = x.loc[(x.method == method) & (x.timeframe == tf) & x.event_id.isin(shared),
                        ["event_id", "side", "net_return"]]
            joined = ref.merge(comp, on=["event_id", "side"], validate="one_to_one")
            joined["difference"] = joined.reference_return - joined.net_return
            for side, z in joined.groupby("side"):
                paired.append(dict(split=split, multiple_r=float(multiple), side=side,
                    reference="zigzag 4h", comparator=f"{method} {tf}", n=len(z),
                    mean_paired_delta_pct=pct(z.difference.mean()),
                    median_paired_delta_pct=pct(z.difference.median()),
                    share_reference_better_pct=pct(z.difference.gt(0).mean())))
    return pd.DataFrame(summaries), pd.DataFrame(paired)


def chronology(f: pd.DataFrame, split: str) -> pd.DataFrame:
    records=[]
    for (method, tf, multiple), g in f.loc[f.portfolio_status.eq("TAKEN")].groupby(["method", "timeframe", "multiple_r"]):
        g=g.sort_values("entry_at")
        if (g.equity_after.isna() | g.equity_before.isna()).any():
            raise ValueError("Equity manquante")
        if not np.allclose(g.equity_after.to_numpy(),g.equity_before.to_numpy()*(1+g.net_return.to_numpy()),rtol=1e-8):
            raise ValueError("Equity incoherente")
        eq=g.equity_after.to_numpy()
        peak=np.maximum.accumulate(np.r_[1., eq])
        dd=(np.r_[1.,eq]/peak - 1.)
        for month, chunk in g.groupby(g.entry_at.dt.strftime("%Y-%m"), sort=True):
            month_return=float(np.prod(1+chunk.net_return.to_numpy())-1)
            records.append(dict(split=split, method=method, timeframe=tf, multiple_r=float(multiple),
                month=month, taken=len(chunk), monthly_compounded_pct=pct(month_return),
                mean_trade_net_pct=pct(chunk.net_return.mean()),
                # Drawdown sur l'intégralité du scénario (répété pour chaque mois pour commodité).
                scenario_max_drawdown_pct=pct(dd.min()),
                first_entry=str(chunk.entry_at.min()), last_entry=str(chunk.entry_at.max())))
    return pd.DataFrame(records)


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def run(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Sortie existante; refus d'ecraser: {output}")
    all_breakdown=[]; all_common=[]; all_pair=[]; all_chrono=[]; fingerprints={}
    for split in SPLITS:
        path=source / f"{split}_scenarios.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        f=validate(pd.read_parquet(path),split)
        all_breakdown.append(breakdown(f,split))
        common,paired=common_comparison(f,split)
        all_common.append(common);all_pair.append(paired)
        all_chrono.append(chronology(f,split))
        fingerprints[split]=dict(path=str(path),sha256=sha256(path),rows=len(f))
        print(f"{split.upper():10} {len(f):6} scenarios | "
              f"comparaison commune: {int(common.loc[common.multiple_r.eq(1.),'common_candidates'].sum()/4)} candidats (LONG+SHORT)")
        view=breakdown(f,split)
        view=view.loc[view.multiple_r.eq(1.5),["method","timeframe","side","eligible","taken","eligible_net_mean_pct","taken_tp","taken_sl","taken_time"]]
        print(view.to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    output.mkdir(parents=True,exist_ok=False)
    targets={"direction_outcomes.csv":pd.concat(all_breakdown,ignore_index=True),
             "common_candidates.csv":pd.concat(all_common,ignore_index=True),
             "paired_comparison.csv":pd.concat(all_pair,ignore_index=True),
             "portfolio_months.csv":pd.concat(all_chrono,ignore_index=True)}
    for name,frame in targets.items():
        frame.to_csv(output/name,index=False,float_format="%.8f")
    manifest={"version":"v31-structural-diagnostic-v1","inputs":fingerprints,
      "outputs":{name:len(frame) for name,frame in targets.items()},
      "scope":"Train/Validation uniquement; lecture seule; aucun nouveau trade, aucun Test, aucun entrainement",
      "caution":"Validation deja exploree; resultats descriptifs et non selection optimisee",
      "gross_return_note":"Inclut le slippage, exclut les frais; la difference gross-net mesure les frais seulement",
      "common_note":"Intersection des event_id eligibles pour les 4 methodes sur chaque multiple; performances individuelles, pas un portefeuille recalcule",
      "monthly_note":"Les rendements mensuels regroupent les transactions par mois d'entree, même si sortie dans le mois suivant"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nRapports crees :",output)
    print("Aucun Test charge. Aucune simulation ni optimisation lancee.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("data/evaluation/v31-structural-tp"))
    p.add_argument("--output",type=Path,default=Path("data/evaluation/v31-structural-diagnostic"))
    a=p.parse_args()
    run(a.source,a.output)


if __name__ == "__main__":
    main()

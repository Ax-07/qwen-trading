"""V4.7.1 read-only multi-observation diagnostic; NOT a training-label builder.

Only BTCUSDT Jan 2024 DEVELOPMENT_PILOT_NOT_TRAIN. Deterministic, spaced,
non-overlapping 60m outcome windows. Candidate proposals see 1m bars before T;
future bars are passed exclusively to V4.5.1 replay for hindsight diagnostics.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import pandas as pd
from candidate_learning_v470 import (_check_minute_frame, make_candidates, score_one,
    validate_pilot_observation, DEFAULT_MINUTES, DEFAULT_OBSERVATIONS, SPECS)

START = pd.Timestamp('2024-01-01T00:00:00Z')
END = pd.Timestamp('2024-02-01T00:00:00Z')
DEFAULT_FIRST = pd.Timestamp('2024-01-03T13:10:00Z')


def select_decisions(observations, *, count=12, first=DEFAULT_FIRST,
                     stride_minutes=2880, horizon_minutes=60):
    """Select exact known 5m timestamps, reject missing rather than cherry-pick.

    Stride > horizon +5 ensures forward outcome windows do not overlap.
    """
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 32:
        raise ValueError('count must be 1..32')
    if not isinstance(stride_minutes, int) or isinstance(stride_minutes, bool) or stride_minutes % 5 or stride_minutes < horizon_minutes + 10:
        raise ValueError('stride must be 5m-aligned, longer than outcome window')
    if not isinstance(horizon_minutes,int) or isinstance(horizon_minutes,bool) or not 5 <= horizon_minutes <= 240 or horizon_minutes % 5:
        raise ValueError('invalid horizon')
    t = pd.Timestamp(first)
    if t.tzinfo is None: raise ValueError('first must be timezone-aware')
    t=t.tz_convert('UTC')
    if t.second or t.microsecond or t.minute%5: raise ValueError('first not on 5m boundary')
    if not isinstance(observations,pd.DataFrame) or 'decision_at' not in observations or observations.columns.duplicated().any():
        raise ValueError('missing/duplicate observation columns')
    ts = pd.to_datetime(observations.decision_at, utc=True, errors='raise')
    if ts.isna().any() or not ts.is_unique or not ts.is_monotonic_increasing:
        raise ValueError('invalid observation chronology')
    chosen=[t+pd.Timedelta(minutes=i*stride_minutes) for i in range(count)]
    if chosen[0] < START+pd.Timedelta(minutes=65) or chosen[-1]+pd.Timedelta(minutes=horizon_minutes+5)>END:
        raise ValueError('pilot boundaries violated')
    missing=[x.isoformat() for x in chosen if x not in set(ts)]
    if missing: raise ValueError('missing scheduled observations: '+','.join(missing[:3]))
    return chosen


def _summarize_rows(rows):
    if not rows: raise ValueError('no diagnostics')
    ids=[spec.tag for spec in SPECS]
    if len({(r['decision_at'],r['id']) for r in rows})!=len(rows):
        raise ValueError('duplicate scored candidate')
    sessions=sorted({r['decision_at'] for r in rows})
    for t in sessions:
        if sorted(r['id'] for r in rows if r['decision_at']==t)!=sorted(ids):
            raise ValueError('missing candidate for session')
    summaries={}
    for tag in ids:
        group=[r for r in rows if r['id']==tag]
        statuses=Counter(r['status'] for r in group)
        reasons=Counter(r['exit_reason'] for r in group if r['exit_reason'] is not None)
        net=[float(r['net_pnl']) for r in group if r['status']=='CLOSED']
        if any(not math.isfinite(x) for x in net): raise ValueError('nonfinite pnl')
        summaries[tag]={
            'attempts':len(group),
            'preview_accepted':sum(bool(r['preview_accepted']) for r in group),
            'fill_rejections':sum(int(r['fill_rejections']) for r in group),
            'statuses':dict(sorted(statuses.items())),
            'exit_reasons':dict(sorted(reasons.items())),
            'closed_count':len(net),
            'closed_wins':sum(p>0 for p in net),
            'closed_losses':sum(p<0 for p in net),
            'closed_zero':sum(p==0 for p in net),
            'closed_net_pnl_sum_usdt':round(sum(net),8) if net else None,
            'closed_net_pnl_mean_usdt':round(sum(net)/len(net),8) if net else None,
            'closed_net_pnl_min_usdt':round(min(net),8) if net else None,
            'closed_net_pnl_max_usdt':round(max(net),8) if net else None,
        }
    return summaries


def audit_pilot(minutes, observations, *, count=12, first=DEFAULT_FIRST,
                stride_minutes=2880, horizon_minutes=60, scorer=score_one):
    """Read-only; independent hypothetical 10k-USDT account for each candidate.

    No compound account, portfolio equity curve, labels, or causal feedback.
    """
    f=_check_minute_frame(minutes)
    if f.timestamp.iloc[0]!=START or f.timestamp.iloc[-1]+pd.Timedelta(minutes=1)!=END:
        raise ValueError('requires exact, complete Jan 2024 BTCUSDT 1m dataset')
    decisions=select_decisions(observations,count=count,first=first,
                               stride_minutes=stride_minutes,horizon_minutes=horizon_minutes)
    rows=[]
    observation_missing=[]
    for t in decisions:
        meta=validate_pilot_observation(observations,t)
        observation_missing.append(meta['missing_feature_count'])
        # Short *exactly* ending at T, without including T or any later bar.
        history=f.loc[(f.timestamp>=t-pd.Timedelta(minutes=65)) & (f.timestamp<t)].copy()
        if len(history)!=65:
            raise ValueError('missing required pre-decision candles')
        # Candidate generation receives ONLY past data. Scorer receives a bounded
        # replay window and has sole access to future prices.
        candidates=make_candidates(history,t)
        assert len(candidates)==len(SPECS)
        replay_window=f.loc[(f.timestamp>=t-pd.Timedelta(minutes=5)) &
                            (f.timestamp<t+pd.Timedelta(minutes=horizon_minutes+5))].copy()
        for candidate in candidates:
            r=scorer(replay_window,t,candidate,horizon_minutes=horizon_minutes)
            if r['status'] not in ('CLOSED','NO_FILL','OPEN_UNRESOLVED'):
                raise ValueError('unexpected execution outcome')
            if r['status']!='CLOSED' and (r['net_pnl'] is not None or r['net_return_pct'] is not None):
                raise ValueError('incomplete outcome must not carry net pnl')
            rows.append({'decision_at':t.isoformat(),**r})
    return {'version':'V4.7.1','dataset_role':'DEVELOPMENT_PILOT_NOT_TRAIN',
            'symbol':'BTCUSDT','period':'2024-01','count':len(decisions),
            'horizon_minutes':horizon_minutes,'stride_minutes':stride_minutes,
            'wait_baseline':{'action':'WAIT','net_pnl_usdt_per_flat_episode':0.0,
                             'execution_fees_usdt':0.0},
            'observation_missing_features':{'min':min(observation_missing),
                        'max':max(observation_missing),'mean':round(sum(observation_missing)/len(observation_missing),3)},
            'by_candidate':_summarize_rows(rows),
            'rows':rows,
            'limitations':('Independent overlapping-free hypothetical trades; each starts FLAT '
                'with 10000 USDT. Summing hypothetical closed PnLs is not a strategy return. '
                'Unfilled/unresolved outcomes excluded from mean. No policy or learning labels; '
                'the selected timestamps are fixed calendar samples, not signal-driven trades.'),
            'training':False,'test_access':False,'orders':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['inspect-pilot'],default='inspect-pilot')
    p.add_argument('--minutes',default=DEFAULT_MINUTES)
    p.add_argument('--observations',default=DEFAULT_OBSERVATIONS)
    p.add_argument('--count',type=int,default=12)
    p.add_argument('--stride-minutes',type=int,default=2880)
    p.add_argument('--horizon-minutes',type=int,default=60)
    p.add_argument('--first',default=DEFAULT_FIRST.isoformat())
    a=p.parse_args(argv)
    report=audit_pilot(pd.read_parquet(a.minutes),pd.read_parquet(a.observations),
                       count=a.count,first=a.first,stride_minutes=a.stride_minutes,
                       horizon_minutes=a.horizon_minutes)
    print('V4.7.1 MULTI-CANDIDATE PILOT',json.dumps(report,indent=2,allow_nan=False))
    print('Read-only diagnostic; no labels, no Test, no training and no orders.')

if __name__=='__main__':main()

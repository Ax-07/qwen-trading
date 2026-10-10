"""V4.7.0 — candidate action diagnostic, never an automatic training-label maker.

One closed 5m decision T, trailing 1m volatility only before T, forward 1m
replay isolated per candidate (for offline diagnostics). Future prices never
enter candidate construction. No Qwen, no real orders, no Test evaluation.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from agent_actions_risk_v450 import Action, ActionType, Position, MarketContext, RiskPolicy, validate_action
from simulate_execution_v451 import replay, ExecutionConfig
from prototype_multitimeframe_v42 import validate_minutes

@dataclass(frozen=True)
class CandidateSpec:
    stop_multiple: float
    target_multiple: float
    tag: str

SPECS = (CandidateSpec(1., 1.5, 'sl1_tp1p5'), CandidateSpec(1., 2., 'sl1_tp2'),
         CandidateSpec(1.5, 2., 'sl1p5_tp2'), CandidateSpec(2., 3., 'sl2_tp3'))
DEFAULT_MINUTES = 'data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'
DEFAULT_OBSERVATIONS = 'data/evaluation/v442-observation-audit/pilot_observations.parquet'


def _check_minute_frame(frame):
    f = validate_minutes(frame).reset_index(drop=True)
    if f.empty or (f.timestamp.diff().iloc[1:] != pd.Timedelta(minutes=1)).any():
        raise ValueError('missing/gapped 1m candles')
    if f.timestamp.iloc[0].tzinfo is None: raise ValueError('UTC timestamps required')
    return f


def trailing_volatility(history: pd.DataFrame, decision_at: pd.Timestamp, lookback=60) -> float:
    """Mean 1m true range on candles COMPLETED no later than decision_at."""
    if not isinstance(lookback, int) or not 20 <= lookback <= 300:
        raise ValueError('lookback must be between 20 and 300 minutes')
    t = pd.Timestamp(decision_at)
    if t.tzinfo is None: raise ValueError('naive timestamp')
    t = t.tz_convert('UTC')
    if (t.minute % 5) or t.second or t.microsecond: raise ValueError('not 5m boundary')
    f = _check_minute_frame(history)
    if f.timestamp.iloc[-1] + pd.Timedelta(minutes=1) != t:
        raise ValueError('history must end exactly at decision; future/history gap')
    if len(f) < lookback + 1: raise ValueError('not enough completed candles')
    recent = f.iloc[-lookback:]
    previous = f.close.shift(1).iloc[-lookback:]
    tr = pd.concat([(recent.high-recent.low).rename('a'),
                    (recent.high-previous).abs().rename('b'),
                    (recent.low-previous).abs().rename('c')], axis=1).max(axis=1)
    atr = float(tr.mean())
    if not math.isfinite(atr) or atr <= 0: raise ValueError('invalid trailing volatility')
    return atr


def make_candidates(history, decision_at, *, config=ExecutionConfig(), policy=RiskPolicy(), specs=SPECS):
    """Proposals depend exclusively on <= T market history. Output preview verdicts."""
    f = _check_minute_frame(history)
    t = pd.Timestamp(decision_at)
    atr = trailing_volatility(f, t)
    price = float(f.close.iloc[-1])
    last = f.iloc[-1]
    liquidity = float(last.get('quote_volume', last.volume*price))
    ctx = MarketContext(t.to_pydatetime(),price,config.spread_bps,liquidity,
                        config.starting_equity,market_allows_short=False,quote_known_at=t.to_pydatetime())
    outputs = []
    for spec in specs:
        if not (math.isfinite(spec.stop_multiple) and spec.stop_multiple > 0 and
                math.isfinite(spec.target_multiple) and spec.target_multiple > 0):
            raise ValueError('bad candidate multiplier')
        sl = price - atr * spec.stop_multiple
        tp = price + atr * spec.target_multiple
        if sl <= 0: raise ValueError('nonpositive stop')
        action = Action(ActionType.OPEN_LONG, sl=sl, tp=tp)
        v = validate_action(action, Position(), ctx, policy)
        outputs.append({'id':spec.tag,'action':action,'preview_accepted':v.accepted,
                        'preview_code':v.code,'approved_qty_preview':v.approved_qty,
                        'atr_1m':atr,'reference_price':price})
    return outputs


def score_one(frame: pd.DataFrame, decision_at, candidate, *, horizon_minutes=60,
              config=ExecutionConfig(), policy=RiskPolicy()):
    """Future information ONLY inside replay; no future data returned as input to agent.

    Window starts T-5min and ends T+horizon+5min, allowing CLOSE at
    T+horizon to fill during the next 1m bar. All data must exist.
    """
    if not isinstance(horizon_minutes,int) or horizon_minutes < 5 or horizon_minutes % 5 or horizon_minutes > 240:
        raise ValueError('invalid horizon')
    t = pd.Timestamp(decision_at)
    if t.tzinfo is None: raise ValueError('naive time')
    t=t.tz_convert('UTC')
    f=_check_minute_frame(frame)
    left=t-pd.Timedelta(minutes=5)
    right=t+pd.Timedelta(minutes=horizon_minutes+5)
    sub=f.loc[(f.timestamp >= left)&(f.timestamp < right)].copy()
    if len(sub) != horizon_minutes+10 or sub.timestamp.iloc[0]!=left or sub.timestamp.iloc[-1]+pd.Timedelta(minutes=1)!=right:
        raise ValueError('insufficient full forward replay window')
    action=candidate['action']
    if not isinstance(action,Action): raise ValueError('candidate action missing')
    # Stop/target triggers are deterministic, conservative replay stop-first.
    events,summary=replay(sub,{t:action,t+pd.Timedelta(minutes=horizon_minutes):Action(ActionType.CLOSE)},config=config,policy=policy)
    entries=events.loc[events.event=='ENTRY']
    exits=events.loc[events.event=='EXIT']
    rejects=events.loc[events.event=='REJECT_AT_FILL']
    closed=len(entries)==1 and len(exits)==1 and summary['position_side']=='FLAT'
    result={'id':candidate['id'],'preview_accepted':bool(candidate['preview_accepted']),
            'preview_code':candidate['preview_code'],
            'entry_count':len(entries),'exit_count':len(exits),
            'fill_rejections':len(rejects),'closed':closed,
            'exit_reason':str(exits.iloc[0].reason) if len(exits)==1 else None,
            'net_pnl':round(float(summary['cash_after_realized']-config.starting_equity),8) if closed else None,
            'net_return_pct':round(100*(float(summary['cash_after_realized'])/config.starting_equity-1),8) if closed else None,
            'status':'CLOSED' if closed else ('NO_FILL' if len(entries)==0 else 'OPEN_UNRESOLVED')}
    return result



def validate_pilot_observation(observations: pd.DataFrame, decision_at):
    """Require one V4.4.2 144-variable observation exactly available at T.

    The candidate engine deliberately does not select candidates from hindsight
    or from future observation rows. Values are not used for outcome scoring.
    """
    if not isinstance(observations, pd.DataFrame) or observations.columns.duplicated().any():
        raise ValueError('invalid observations frame')
    features=[c for c in observations if c != 'decision_at']
    if 'decision_at' not in observations or len(features)!=144:
        raise ValueError('expected 144 V4.4.2 features')
    times=pd.to_datetime(observations['decision_at'],utc=True,errors='raise')
    if times.isna().any() or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('invalid observation chronology')
    t=pd.Timestamp(decision_at)
    if t.tzinfo is None: raise ValueError('naive decision')
    match=observations.loc[times==t.tz_convert('UTC'),features]
    if len(match)!=1: raise ValueError('decision must match one observation')
    import numpy as np
    if any(not pd.api.types.is_numeric_dtype(match[c]) for c in features):
        raise ValueError('nonnumeric pilot feature')
    values=match.iloc[0].to_numpy(dtype=float,na_value=float('nan'))
    if np.isinf(values).any(): raise ValueError('infinite pilot feature')
    return {'observation_at':t.isoformat(),'feature_count':144,
            'missing_feature_count':int(np.isnan(values).sum())}

def inspect_pilot(frame, decision_at, *, horizon_minutes=60, observations=None):
    """Diagnostic-only fixed Jan 2024 pilot. Not Train and not a dataset of labels."""
    f=_check_minute_frame(frame)
    t=pd.Timestamp(decision_at)
    if t.tzinfo is None: raise ValueError('naive time')
    t=t.tz_convert('UTC')
    if not (pd.Timestamp('2024-01-01T01:05:00Z')<=t<=pd.Timestamp('2024-01-30T23:00:00Z')):
        raise ValueError('outside allowed Jan 2024 development pilot')
    observation_meta=validate_pilot_observation(observations,t) if observations is not None else None
    history=f.loc[f.timestamp<t].copy()
    candidates=make_candidates(history,t)
    return {'version':'V4.7.0','dataset_role':'DEVELOPMENT_PILOT_NOT_TRAIN',
            'decision_at':t.isoformat(),'horizon_minutes':horizon_minutes,
            'observation':observation_meta,
            'candidates':[score_one(f,t,c,horizon_minutes=horizon_minutes) for c in candidates],
            'warning':'Outcomes are hindsight evaluation only; NOT training labels, policy or profit evidence.',
            'orders':False,'training':False,'test_access':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('inspect-pilot',),default='inspect-pilot')
    p.add_argument('--minutes',default=DEFAULT_MINUTES)
    p.add_argument('--observations',default=DEFAULT_OBSERVATIONS)
    p.add_argument('--decision-at',default='2024-01-03T13:10:00Z')
    p.add_argument('--horizon-minutes',type=int,default=60)
    a=p.parse_args(argv)
    report=inspect_pilot(pd.read_parquet(a.minutes),a.decision_at,
                         horizon_minutes=a.horizon_minutes,observations=pd.read_parquet(a.observations))
    print('V4.7.0 PILOT',json.dumps(report,indent=2,allow_nan=False))
    print('Read-only diagnostic; no label files, no training, no Test, no live orders.')

if __name__=='__main__':main()

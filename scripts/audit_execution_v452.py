"""V4.5.2 independent read-only event-ledger auditor for V4.5.1.

Reconstructs positions, fees, realized cash and marks directly from event fields.
No import of the simulator or its accounting helpers. No trading or training.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import pandas as pd

TOL = 1e-6
KNOWN = {'DECISION', 'ENTRY', 'EXIT', 'MODIFY', 'REJECT_AT_FILL', 'UNFILLED_END_OF_DATA', 'MARK_TO_MARKET_ONLY'}

def _ts(x, label):
    if pd.isna(x): raise ValueError(f'{label}: missing timestamp')
    t = pd.Timestamp(x)
    if t.tzinfo is None: raise ValueError(f'{label}: timestamp must be timezone-aware')
    return t.tz_convert('UTC')

def _num(row, key, i, *, positive=False, nonnegative=False):
    value = row.get(key)
    if pd.isna(value): raise ValueError(f'row {i} {key}: missing')
    try: x = float(value)
    except (TypeError, ValueError): raise ValueError(f'row {i} {key}: not numeric') from None
    if not math.isfinite(x) or (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(f'row {i} {key}: invalid numeric value')
    return x

def _close(a, b, name):
    if not math.isclose(float(a), float(b), abs_tol=TOL, rel_tol=1e-9):
        raise ValueError(f'{name}: ledger={a}, expected={b}')

def audit(events: pd.DataFrame, summary: dict, *, starting_equity=10000.0,
          fee_bps=7.0, minutes: pd.DataFrame | None = None) -> dict:
    """Raise ValueError for inconsistent or incomplete ledger. Return independent report.

    minutes mandatory for authenticating boundaries, data coverage and open-position MTM.
    Accepts V4.5.1 log from pd.read_csv (NaN in optional fields).
    """
    if not isinstance(summary, dict): raise ValueError('summary must be dict')
    if not isinstance(events, pd.DataFrame) or events.empty: raise ValueError('empty event log')
    if not math.isfinite(starting_equity) or starting_equity <= 0 or not math.isfinite(fee_bps) or fee_bps < 0:
        raise ValueError('invalid audit configuration')
    for c in ('at', 'event'):
        if c not in events: raise ValueError(f'missing required event column {c}')
    if not events['event'].isin(KNOWN).all(): raise ValueError('unknown event type')
    if minutes is None: raise ValueError('independent 1m prices are required')
    for c in ('timestamp', 'open', 'high', 'low', 'close'):
        if c not in minutes: raise ValueError(f'missing candle column {c}')
    times = pd.to_datetime(minutes.timestamp, utc=True, errors='raise')
    if times.empty or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('invalid minute chronology')
    if (times.diff().iloc[1:] != pd.Timedelta(minutes=1)).any() or len(minutes)%5:
        raise ValueError('incomplete 1m coverage')
    start, end = times.iloc[0], times.iloc[-1] + pd.Timedelta(minutes=1)
    if start.minute % 5 or start.second or start.microsecond: raise ValueError('non-aligned minute start')
    _close(_num({'n': summary.get('events')}, 'n', 'summary', nonnegative=True), len(events), 'summary events')
    if summary.get('data_start') is None or summary.get('data_end') is None:
        raise ValueError('summary missing interval')
    if _ts(summary['data_start'], 'data_start') != start or _ts(summary['data_end'], 'data_end') != end:
        raise ValueError('summary interval differs from independent 1m candles')
    if summary.get('no_real_orders') is not True: raise ValueError('no_real_orders must be true')
    prev = None
    cash = float(starting_equity)
    position = None
    decisions = {}
    last_decision = None
    closed = 0
    entries = 0
    rejected = 0
    total_fees = 0.0
    trade_records = []
    events_by_time = {}
    for i, row in enumerate(events.to_dict('records')):
        t = _ts(row['at'], f'row {i}')
        kind = row['event']
        if prev is not None and t < prev: raise ValueError('event chronology reversal')
        prev = t
        if t < start or t > end: raise ValueError('event outside dataset')
        events_by_time.setdefault(t, []).append(kind)
        if kind == 'DECISION':
            if t == start or t.minute % 5 or t.second or t.microsecond or t > end:
                raise ValueError('decision outside completed 5m boundary')
            if t in decisions: raise ValueError('duplicate DECISION')
            action = row.get('action')
            if action not in {'WAIT','OPEN_LONG','OPEN_SHORT','HOLD','CLOSE','MOVE_SL','MOVE_TP','MOVE_SL_TP'}:
                raise ValueError('invalid decision action')
            accepted = row.get('accepted')
            if isinstance(accepted, str): accepted = accepted.strip().lower() == 'true' if accepted.strip().lower() in ('true','false') else None
            if not isinstance(accepted, bool): raise ValueError('invalid accepted flag')
            decisions[t] = (action, accepted)
            last_decision = t
            continue
        if kind == 'ENTRY':
            if position is not None: raise ValueError('ENTRY while position open')
            origin = _ts(row.get('originating_decision'), f'row {i} originating_decision')
            if origin not in decisions or decisions[origin] != ('OPEN_LONG', True):
                raise ValueError('ENTRY without accepted OPEN_LONG decision')
            if t != origin or t >= end: # 1m timestamp at decision boundary is next candle opening
                raise ValueError('ENTRY must execute at following 1m open')
            if 'ENTRY' in events_by_time[t][:-1]: raise ValueError('duplicate ENTRY at same time')
            px = _num(row, 'price', i, positive=True); qty = _num(row, 'qty', i, positive=True)
            sl = _num(row, 'sl', i, positive=True); tp = _num(row, 'tp', i, positive=True)
            if not sl < px < tp: raise ValueError('invalid entry protection geometry')
            fee = _num(row, 'fee', i, nonnegative=True)
            _close(fee, px*qty*fee_bps/10000, 'entry fee')
            cash -= fee; total_fees += fee; entries += 1
            position = {'entry': px, 'qty': qty, 'sl':sl, 'tp':tp,'fee':fee,'opened_at':t}
            continue
        if kind == 'EXIT':
            if position is None: raise ValueError('EXIT without position')
            px = _num(row,'price',i,positive=True); qty = _num(row,'qty',i,positive=True)
            _close(qty, position['qty'], 'exit qty')
            fee = _num(row,'exit_fee',i,nonnegative=True)
            _close(fee, px*qty*fee_bps/10000,'exit fee')
            pnl = (px-position['entry'])*qty-fee-position['fee']
            _close(_num(row,'net_trade_pnl',i),pnl,'net trade pnl')
            cash += (px-position['entry'])*qty-fee
            _close(_num(row,'cash',i), cash,'cash at EXIT')
            reason = row.get('reason')
            if reason not in {'AGENT_CLOSE','STOP_GAP','STOP_INTRABAR','TARGET_GAP_CAPPED','TARGET_INTRABAR'}:
                raise ValueError('invalid exit reason')
            if reason in {'STOP_GAP','TARGET_GAP_CAPPED','STOP_INTRABAR','TARGET_INTRABAR'}:
                ix = times.searchsorted(t)
                if ix >= len(times) or times.iloc[ix] != t: raise ValueError('exit missing candle')
                candle = minutes.iloc[ix]
                if reason == 'STOP_INTRABAR' and float(candle.low) > position['sl']:
                    raise ValueError('STOP not touched')
                if reason == 'TARGET_INTRABAR' and float(candle.high) < position['tp']:
                    raise ValueError('TP not touched')
                if reason == 'STOP_GAP' and float(candle.open) > position['sl']:
                    raise ValueError('STOP gap not reached')
                if reason == 'TARGET_GAP_CAPPED' and float(candle.open) < position['tp']:
                    raise ValueError('TP gap not reached')
            if reason == 'AGENT_CLOSE' and (t not in decisions or decisions[t] != ('CLOSE',True)):
                raise ValueError('AGENT_CLOSE without approved CLOSE decision')
            trade_records.append({'opened_at':position['opened_at'].isoformat(), 'closed_at':t.isoformat(),
                                  'qty':qty,'entry_price':position['entry'],'exit_price':px,
                                  'entry_fee':position['fee'],'exit_fee':fee,'net_pnl':pnl,'reason':reason})
            total_fees += fee; closed += 1; position = None
            continue
        if kind == 'MODIFY':
            if position is None: raise ValueError('MODIFY without position')
            if t not in decisions or decisions[t][1] is not True or decisions[t][0] not in {'MOVE_SL','MOVE_TP','MOVE_SL_TP'}:
                raise ValueError('MODIFY without approved modification')
            sl = _num(row,'sl',i,positive=True); tp=_num(row,'tp',i,positive=True)
            if sl + TOL < position['sl']: raise ValueError('stop widening')
            if sl >= tp: raise ValueError('invalid modification geometry')
            position['sl'],position['tp'] = sl,tp
            continue
        if kind == 'REJECT_AT_FILL':
            rejected += 1
            if not isinstance(row.get('reason'),str) or not row.get('reason'): raise ValueError('missing rejection reason')
            continue
        if kind == 'UNFILLED_END_OF_DATA':
            if t != end: raise ValueError('UNFILLED_END_OF_DATA at wrong boundary')
            if not last_decision or last_decision != end or not decisions[end][1] or decisions[end][0] in ('WAIT','HOLD'):
                raise ValueError('unfilled with no pending action')
            if row.get('action') != decisions[end][0]: raise ValueError('unfilled action mismatch')
            continue
        if kind == 'MARK_TO_MARKET_ONLY':
            if t != end or position is None: raise ValueError('invalid MARK_TO_MARKET_ONLY')
            unrealized = (float(minutes.close.iloc[-1])-position['entry'])*position['qty']
            _close(_num(row,'unrealized_pnl',i),unrealized,'unrealized PnL')
    expected_decisions = pd.date_range(start + pd.Timedelta(minutes=5),end,freq='5min',tz='UTC')
    if len(decisions) != len(expected_decisions) or set(decisions) != set(expected_decisions):
        raise ValueError('missing DECISION rows')
    mark = float(minutes.close.iloc[-1]); equity = cash + ((mark-position['entry'])*position['qty'] if position else 0.0)
    _close(_num({'v':summary.get('cash_after_realized')},'v','summary'),cash,'summary cash')
    _close(_num({'v':summary.get('equity_mark_to_market')},'v','summary'),equity,'summary equity')
    if summary.get('trades_closed') != closed: raise ValueError('summary closed trade count')
    if summary.get('position_side') != ('LONG' if position else 'FLAT'): raise ValueError('summary position side')
    mtm_rows = (events.event == 'MARK_TO_MARKET_ONLY').sum()
    if mtm_rows != (1 if position else 0): raise ValueError('missing or unexpected mark-to-market')
    return {'status':'PASS','minutes':len(minutes),'decisions':len(decisions),'events':len(events),
            'entries':entries,'trades_closed':closed,'rejections_at_fill':rejected,
            'cash_reconstructed':cash,'equity_reconstructed':equity,'net_pnl_vs_initial':equity-starting_equity,
            'total_fees':total_fees,'position_side':'LONG' if position else 'FLAT', 'trades':trade_records,
            'no_real_orders':True}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events',default='data/evaluation/v451-simulator/BTCUSDT-2024-01/events.csv')
    ap.add_argument('--summary',default='data/evaluation/v451-simulator/BTCUSDT-2024-01/summary.json')
    ap.add_argument('--minutes',default='data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet')
    ap.add_argument('--output-dir',default='data/evaluation/v452-ledger-audit/BTCUSDT-2024-01')
    ap.add_argument('--starting-equity',type=float,default=10000.0)
    ap.add_argument('--fee-bps',type=float,default=7.0)
    a=ap.parse_args()
    out=Path(a.output_dir); rp=out/'audit_report.json'; tp=out/'reconstructed_trades.csv'
    if rp.exists() or tp.exists(): raise FileExistsError('audit output already exists; refusing overwrite')
    e=pd.read_csv(a.events); s=json.loads(Path(a.summary).read_text(encoding='utf-8'))
    m=pd.read_parquet(a.minutes)
    r=audit(e,s,starting_equity=a.starting_equity,fee_bps=a.fee_bps,minutes=m)
    out.mkdir(parents=True,exist_ok=True)
    trades=r.pop('trades')
    with rp.open('x',encoding='utf-8') as f: json.dump(r,f,indent=2)
    with tp.open('x',encoding='utf-8',newline='') as f:
        pd.DataFrame(trades,columns=['opened_at','closed_at','qty','entry_price','exit_price','entry_fee','exit_fee','net_pnl','reason']).to_csv(f,index=False)
    print('V4.5.2 PASS',r)
    print('Outputs:',rp,tp)
    print('Read-only audit of V4.5.1; no orders, no training, no Test evaluation.')

if __name__=='__main__': main()

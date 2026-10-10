"""Qwen Trading V4.3.2: causal BOS / CHoCH based on confirmed V4.3.1 swings.

No future observations, no labels and no trading. A break is a close crossing an
armed, previously confirmed fractal level. The first break establishes direction
and is tagged BOS; a reversal of the established direction is CHoCH.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prototype_multitimeframe_v42 import completed_bars, FREQS
from market_swings_v431 import confirmed_swings

EVENT_COLUMNS = ['event_type','event_direction','broken_level','level_pivot_at',
                 'level_confirmed_at','break_at','trend_direction','active_high',
                 'active_low','break_distance_pct','event_fired']


def detect_breaks(bars: pd.DataFrame, timeframe_minutes: int, left: int = 2,
                  right: int = 2) -> pd.DataFrame:
    """One row per closed bar. Segment state resets on missing candles.

    Tie: close == level is NOT a break. A newly confirmed swing is armed only
    after its confirmation close and only if not already violated at that time.
    """
    swings = confirmed_swings(bars, timeframe_minutes, left, right)
    b = bars.sort_values('timestamp').reset_index(drop=True)
    if b.empty:
        return pd.DataFrame(columns=['known_at', *EVENT_COLUMNS])
    segments = b.timestamp.diff().ne(pd.Timedelta(minutes=timeframe_minutes)).cumsum()
    results = []
    for _, ix in b.groupby(segments, sort=False).groups.items():
        high = low = None
        used_h = used_l = True
        trend = 0
        prev_close = None
        for i in ix:
            candle = b.iloc[i]
            swing = swings.iloc[i]
            close = float(candle.close)
            event = 'NONE'; direction = 0; broken = np.nan
            pivot_at = pd.NaT; confirmed_at = pd.NaT
            # Only levels confirmed on a prior candle may break now.
            bull = (high is not None and not used_h and prev_close is not None
                    and prev_close <= high[0] and close > high[0])
            bear = (low is not None and not used_l and prev_close is not None
                    and prev_close >= low[0] and close < low[0])
            # Both cannot trigger for valid high/low structure under normal
            # conditions. In anomalous overlapping levels, reject ambiguity.
            if bull and bear:
                raise ValueError('Cassures bullish et bearish simultanees')
            if bull or bear:
                direction = 1 if bull else -1
                selected = high if bull else low
                broken, pivot_at, confirmed_at = selected
                event = ('BULLISH_' if bull else 'BEARISH_') + ('CHOCH' if trend == -direction else 'BOS')
                trend = direction
                if bull: used_h = True
                else: used_l = True
            # Do not retroactively claim a break of a pivot first recognized
            # on this candle. A confirmation may replace an older armed level.
            if bool(swing.pivot_high_confirmed):
                price = float(swing.swing_high_price)
                high = (price, swing.swing_high_at, swing.swing_high_confirmed_at)
                used_h = close >= price
            if bool(swing.pivot_low_confirmed):
                price = float(swing.swing_low_price)
                low = (price, swing.swing_low_at, swing.swing_low_confirmed_at)
                used_l = close <= price
            results.append(dict(known_at=candle.known_at, event_type=event,
                                event_direction=direction, broken_level=broken,
                                level_pivot_at=pivot_at, level_confirmed_at=confirmed_at,
                                break_at=candle.known_at if direction else pd.NaT,
                                trend_direction=trend,
                                active_high=np.nan if high is None or used_h else high[0],
                                active_low=np.nan if low is None or used_l else low[0],
                                break_distance_pct=(close/broken-1) if direction else np.nan,
                                event_fired=bool(direction)))
            prev_close = close
    result = pd.DataFrame(results)
    for col in ('known_at','level_pivot_at','level_confirmed_at','break_at'):
        result[col] = pd.to_datetime(result[col], utc=True).astype('datetime64[ns, UTC]')
    return result


def build_break_snapshots(minutes_frame: pd.DataFrame, left: int = 2, right: int = 2):
    bars = {tf: completed_bars(minutes_frame, n) for tf, n in FREQS.items()}
    events = {tf: detect_breaks(bar, FREQS[tf], left, right) for tf, bar in bars.items()}
    out = bars['5m'][['timestamp','known_at','close']].rename(columns={
        'timestamp':'bar_open_5m','known_at':'decision_at','close':'close_5m'}).copy()
    out['decision_at'] = pd.to_datetime(out.decision_at, utc=True).astype('datetime64[ns, UTC]')
    out = out.sort_values('decision_at')
    for tf in FREQS:
        src = events[tf].rename(columns={'known_at':f'known_at_{tf}',
                                         **{k:f'{k}_{tf}' for k in EVENT_COLUMNS}})
        if src.empty:
            for field in src.columns:
                out[field] = np.nan
            continue
        key = f'known_at_{tf}'
        src[key] = pd.to_datetime(src[key], utc=True).astype('datetime64[ns, UTC]')
        out = pd.merge_asof(out, src.sort_values(key), left_on='decision_at',
                            right_on=key, direction='backward',
                            allow_exact_matches=True,
                            tolerance=pd.Timedelta(minutes=FREQS[tf]))
        # Context (trend, armed levels) may persist; event impulses must not.
        fresh = out.decision_at.eq(out[key])
        for field in ('event_fired',):
            col = f'{field}_{tf}'
            out[col] = out[col].eq(True) & fresh
        for field, neutral in [('event_type', 'NONE'), ('event_direction', 0)]:
            col = f'{field}_{tf}'
            out.loc[~fresh & out[key].notna(), col] = neutral
        for field in ('broken_level','level_pivot_at','level_confirmed_at',
                      'break_at','break_distance_pct'):
            out.loc[~fresh, f'{field}_{tf}'] = pd.NaT if field.endswith('_at') else np.nan
    return out.reset_index(drop=True), events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('data/evaluation/v421-binance-1m/BTCUSDT/2024-01.parquet'))
    p.add_argument('--output', type=Path, default=Path('data/evaluation/v432-bos-choch/BTCUSDT-2024-01.parquet'))
    p.add_argument('--left', type=int, default=2)
    p.add_argument('--right', type=int, default=2)
    args = p.parse_args()
    if not args.input.is_file(): p.error(f'Entrée absente : {args.input}')
    if args.output.exists(): p.error(f'Refus écrasement : {args.output}')
    snapshots, events = build_break_snapshots(pd.read_parquet(args.input), args.left, args.right)
    print('Snapshots 5m :', len(snapshots))
    for tf, e in events.items():
        print(f'{tf:>3}: ' + ', '.join(f'{name}={int((e.event_type==name).sum())}' for name in
              ('BULLISH_BOS','BEARISH_BOS','BULLISH_CHOCH','BEARISH_CHOCH')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(args.output, index=False)
    print('Parquet :', args.output.resolve())
    print('Aucune source modifiée. Aucun entraînement ni trading.')

if __name__ == '__main__': main()

"""Tests V4.3.2: python scripts/test_market_bos_choch_v432.py"""
import unittest
import pandas as pd
import numpy as np
from market_bos_choch_v432 import detect_breaks, build_break_snapshots
from prototype_multitimeframe_v42 import synthetic_minutes


def bars(highs, lows=None, closes=None, start='2024-01-01T00:00:00Z'):
    high=np.array(highs,dtype=float)
    low=np.array(lows if lows is not None else high-3, dtype=float)
    close=np.array(closes if closes is not None else (high+low)/2, dtype=float)
    t=pd.date_range(start, periods=len(high), freq='5min',tz='UTC')
    return pd.DataFrame({'timestamp':t,'known_at':t+pd.Timedelta(minutes=5),
                         'high':high,'low':low,'close':close})

class TestBosChoch(unittest.TestCase):
    def test_bos_only_after_confirmed_pivot(self):
        b=bars([101,102,110,106,104,105,109,112,113],
               closes=[100,101,107,104,103,104,108,111,112])
        e=detect_breaks(b,5)
        self.assertEqual(e.event_type.iloc[7], 'BULLISH_BOS')
        self.assertEqual(e.level_confirmed_at.iloc[7], b.known_at.iloc[4])
        self.assertFalse(e.event_fired.iloc[:7].any())

    def test_no_duplicate_break(self):
        b=bars([101,102,110,106,104,105,109,112,113,114,115],
               closes=[100,101,107,104,103,104,108,111,112,113,114])
        e=detect_breaks(b,5)
        self.assertEqual(int(e.event_fired.sum()),1)

    def test_choc_after_opposite_trend(self):
        b=bars([101,102,110,106,104,105,109,112,113,112,111,110,105,103],
               lows=[99,99,100,99,97,99,101,103,101,99,96,95,90,88],
               closes=[100,101,107,104,103,104,108,111,112,110,108,106,94,89])
        e=detect_breaks(b,5)
        self.assertIn('BULLISH_BOS',e.event_type.values)
        self.assertIn('BEARISH_CHOCH',e.event_type.values)

    def test_gap_resets_trend(self):
        a=bars([101,102,110,106,104,105,109,112],closes=[100,101,107,104,103,104,108,111])
        c=bars([101,102,110,106,104,105,109,112],closes=[100,101,107,104,103,104,108,111], start='2024-01-01T02:00:00Z')
        e=detect_breaks(pd.concat([a,c],ignore_index=True),5)
        self.assertEqual(e.event_type.iloc[15],'BULLISH_BOS')

    def test_prefix_stability(self):
        b=bars([101,102,110,106,104,105,109,112,113,114])
        a=detect_breaks(b.iloc[:7],5)
        z=detect_breaks(b,5)
        pd.testing.assert_frame_equal(a.reset_index(drop=True),z.iloc[:7].reset_index(drop=True))

    def test_no_htf_event_repeated(self):
        snap,_=build_break_snapshots(synthetic_minutes(600))
        for tf in ('15m','1h','4h'):
            mask=snap.decision_at.ne(snap[f'known_at_{tf}'])
            self.assertFalse(snap.loc[mask,f'event_fired_{tf}'].any())

    def test_no_future_context(self):
        snap,_=build_break_snapshots(synthetic_minutes(520))
        before=snap[snap.decision_at==pd.Timestamp('2024-01-01T03:55:00Z')].iloc[0]
        after=snap[snap.decision_at==pd.Timestamp('2024-01-01T04:00:00Z')].iloc[0]
        self.assertTrue(pd.isna(before.known_at_4h))
        self.assertEqual(after.known_at_4h,pd.Timestamp('2024-01-01T04:00:00Z'))

    def test_equality_not_break(self):
        b=bars([101,102,110,106,104,105,109,110],closes=[100,101,107,104,103,104,108,110])
        self.assertFalse(detect_breaks(b,5).event_fired.any())

    def test_gap_drops_incomplete_bar(self):
        snap,_=build_break_snapshots(synthetic_minutes(500).drop(index=[13]).reset_index(drop=True))
        self.assertEqual(len(snap),99)

if __name__=='__main__': unittest.main(verbosity=2)

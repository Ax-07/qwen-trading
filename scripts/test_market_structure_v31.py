from __future__ import annotations
import unittest
import pandas as pd
import numpy as np
from build_market_structure_v31 import pivot_features, build_split, aggregate_completed, EVENT_COLUMNS

class TestStructure(unittest.TestCase):
    def bars(self, highs, lows):
        idx = pd.date_range("2026-01-01T01:00:00Z", periods=len(highs), freq="h")
        closes = (np.array(highs)+np.array(lows))/2
        return pd.DataFrame({"open":closes,"high":highs,"low":lows,"close":closes},index=idx)

    def test_pivot_delay(self):
        bars=self.bars([2,3,8,4,3,2,5,6], [1,2,3,2,1,0,2,3])
        x=pivot_features(bars,left=2,right=2)
        self.assertTrue(x.loc[4,"pivot_high_confirmed"])
        self.assertFalse(x.loc[:3,"pivot_high_confirmed"].any())
        self.assertEqual(x.loc[4,"last_swing_high"],8)
        self.assertTrue(pd.isna(x.loc[3,"last_swing_high"]))
        self.assertEqual(x.loc[4,"swing_high_at"],bars.index[2])

    def test_no_future_prefix(self):
        b=self.bars([1,3,5,4,2,4,3,2,6,2,1],[0,1,2,1,0,1,0,0,2,0,-1])
        full=pivot_features(b,2,2)
        for n in range(1,len(b)+1):
            partial=pivot_features(b.iloc[:n],2,2)
            pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True),partial.reset_index(drop=True),
                                          check_dtype=False)

    def test_multitimeframe_causality_and_preserved_rows(self):
        idx=pd.date_range("2026-01-01T00:00:00Z",periods=24*20,freq="h")
        x=pd.DataFrame({"timestamp":idx,"open":100.,"high":102.,"low":98.,"close":100.})
        result=build_split(x)
        self.assertEqual(len(result),len(x))
        for tf in ("1h","4h","1d"):
            available=result[f"{tf}_available_at"]
            self.assertTrue((available.dropna()<=result.loc[available.notna(),"known_at"]).all())
        self.assertTrue(result.loc[0,"1d_high_structure"] == "NONE" or pd.isna(result.loc[0,"1d_high_structure"]))

    def test_mixed_datetime_precisions(self):
        idx=pd.date_range("2026-01-01T00:00:00Z",periods=24*12,freq="h")
        base=pd.DataFrame({"timestamp":idx,"open":100.,"high":102.,"low":98.,"close":100.})
        # Simule les timestamps issus de Parquet (us ou ms).
        for unit in ("us", "ms", "ns"):
            data=base.copy()
            data["timestamp"]=data["timestamp"].dt.as_unit(unit)
            out=build_split(data)
            self.assertEqual(len(out),len(data))
            self.assertEqual(str(out["known_at"].dtype),"datetime64[ns, UTC]")
            for tf in ("1h","4h","1d"):
                self.assertEqual(str(out[f"{tf}_available_at"].dtype),"datetime64[ns, UTC]")

    def test_event_impulses_not_repeated_htf(self):
        idx=pd.date_range("2026-01-01T00:00:00Z", periods=24*26, freq="h")
        # Structure variable dans toutes les resolutions, y compris 1D.
        day=np.arange(len(idx)) / 24
        values=100+10*np.sin(day*1.5)+4*np.sin(day*0.3)
        x=pd.DataFrame({"timestamp":idx,"open":values,"high":values+2,
                        "low":values-2,"close":values})
        out=build_split(x,left=2,right=2)
        for tf in ("1h","4h","1d"):
            raw=pivot_features(aggregate_completed(x,tf),2,2)
            for event in EVENT_COLUMNS:
                self.assertEqual(int(out[f"{tf}_{event}"].sum()),
                                 int(raw[event].sum()), (tf,event))
                active=out[out[f"{tf}_{event}"]]
                self.assertTrue((active["known_at"]==active[f"{tf}_available_at"]).all())
        self.assertGreater(int(out["1d_pivot_high_confirmed"].sum()+out["1d_pivot_low_confirmed"].sum()),0)
        self.assertLess(int(out["1d_pivot_high_confirmed"].sum()+out["1d_pivot_low_confirmed"].sum()),26)

    def test_htf_complete_only(self):
        idx=pd.date_range("2026-01-01T01:00Z",periods=7,freq="h")
        x=pd.DataFrame({"timestamp":idx,"open":100.,"high":102.,"low":98.,"close":100.})
        bars=aggregate_completed(x,"4h")
        self.assertEqual(len(bars),1)
        self.assertEqual(bars.index[0],pd.Timestamp("2026-01-01T08:00Z"))

if __name__=="__main__":
    unittest.main(verbosity=2)

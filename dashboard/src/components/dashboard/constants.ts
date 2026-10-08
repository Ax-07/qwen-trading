export const API = "http://127.0.0.1:8000";

export const FEATURE_GROUPS = [
  { title: "OHLCV", fields: ["open", "high", "low", "close", "volume"] },
  {
    title: "Indicateurs 1H",
    fields: [
      "ema20",
      "ema50",
      "ema200",
      "rsi14",
      "atr14",
      "atr14_pct",
      "volume_ratio",
      "trend_1h",
    ],
  },
  {
    title: "Tendance 4H",
    fields: [
      "4h_close",
      "4h_ema20",
      "4h_ema50",
      "4h_rsi14",
      "4h_atr14_pct",
      "4h_trend",
    ],
  },
  {
    title: "Tendance 1D",
    fields: [
      "1d_close",
      "1d_ema20",
      "1d_ema50",
      "1d_rsi14",
      "1d_atr14_pct",
      "1d_trend",
      "trend_alignment",
    ],
  },
  {
    title: "Returns & Volatilité",
    fields: [
      "return_1h",
      "return_3h",
      "return_6h",
      "return_24h",
      "log_return_1h",
      "volatility_24h",
      "volatility_72h",
    ],
  },
  {
    title: "Breakouts & Structure",
    fields: [
      "prev_high_24h",
      "prev_low_24h",
      "prev_high_72h",
      "prev_low_72h",
      "breakout_24h",
      "breakdown_24h",
      "distance_high_24h_pct",
      "distance_low_24h_pct",
      "distance_high_72h_pct",
      "distance_low_72h_pct",
    ],
  },
  {
    title: "Autres features",
    fields: [
      "distance_ema20_pct",
      "distance_ema50_pct",
      "distance_ema200_pct",
      "ema20_50_spread_pct",
      "ema50_200_spread_pct",
      "volume_sma20",
      "range_pct",
      "body_pct",
      "upper_wick_pct",
      "lower_wick_pct",
    ],
  },
] as const;

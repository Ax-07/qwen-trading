export type Dataset = "train" | "validation" | "test";
export type FeatureValue = number | boolean | null;

export type DatasetInfo = {
  name: Dataset;
  available: boolean;
  start: string | null;
  end: string | null;
  rows: number | null;
  message: string | null;
};

export type CandleDetail = {
  time: number;
  timestamp: string;
  available_at: string;
  features: Record<string, FeatureValue>;
};

export type PredictionDetail = {
  available: boolean;
  message?: string;
  predictions?: {
    qwen: string;
    heuristic: string;
    generated: string;
    bias_score: number;
  };
  evaluation?: {
    target: string;
    decision: string;
    long_result: string;
    short_result: string;
    future_return_12h_pct: number | null;
    up_move_12h_atr: number | null;
    down_move_12h_atr: number | null;
  };
  comparison?: {
    qwen_agrees_with_heuristic: boolean;
    qwen_matches_target: boolean;
    heuristic_matches_target: boolean;
  };
};

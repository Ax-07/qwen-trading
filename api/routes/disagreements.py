from pathlib import Path
from typing import Literal

import polars as pl
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[2]
MARKET_PATH = ROOT / 'data' / 'processed' / 'btc_usdc_1h_labeled.parquet'
EVALUATION_PATHS = {
    'validation': ROOT / 'data' / 'evaluation' / 'qwen3.5-9b-trading-v2-validation' / 'results.parquet',
    'test': ROOT / 'data' / 'evaluation' / 'qwen3.5-9b-trading-v2-fast' / 'results_fast.parquet',
}
Dataset = Literal['validation', 'test']
router = APIRouter(prefix='/api/analytics', tags=['analytics'])


def classify_outcome(decision, long_value, short_value):
    if decision == 'NO_TRADE':
        return 'NO_TRADE'
    value = long_value if decision == 'LONG_BIAS' else short_value if decision == 'SHORT_BIAS' else None
    if value == 1:
        return 'TP'
    if value == -1:
        return 'SL'
    if value == 0:
        return 'UNRESOLVED'
    return 'UNKNOWN'


def classify_disagreement(qwen, heuristic, q_result, h_result):
    if qwen == 'NO_TRADE' and heuristic != 'NO_TRADE':
        kind = 'QWEN_ABSTAINS'
        category = {'SL': 'SL_AVOIDED', 'TP': 'TP_MISSED'}.get(h_result, 'UNDETERMINED')
    elif heuristic == 'NO_TRADE' and qwen != 'NO_TRADE':
        kind = 'QWEN_ADDS_TRADE'
        category = {'TP': 'TP_ADDED', 'SL': 'SL_ADDED'}.get(q_result, 'UNDETERMINED')
    else:
        kind, category = 'OPPOSITE_DIRECTION', 'UNDETERMINED'
    return kind, category


@router.get('/disagreements')
def get_disagreements(dataset: Dataset = 'test'):
    evaluation_path = EVALUATION_PATHS[dataset]
    if not evaluation_path.is_file() or not MARKET_PATH.is_file():
        raise HTTPException(status_code=404, detail='Fichier évaluation ou marché introuvable')

    predictions = (
        pl.scan_parquet(evaluation_path)
        .filter(pl.col('qwen_prediction') != pl.col('heuristic_prediction'))
        .select('timestamp', 'qwen_prediction', 'heuristic_prediction', 'target', 'bias_score',
                'long_outcome_12h', 'short_outcome_12h')
    )
    market = (
        pl.scan_parquet(MARKET_PATH)
        .select('timestamp', 'available_at', 'close', 'rsi14', 'atr14_pct', 'volume_ratio',
                'trend_1h', '4h_trend', '1d_trend', 'trend_alignment', 'features_ready')
    )
    df = predictions.join(market, on='timestamp', how='left').sort('timestamp').collect()
    rows = []
    for item in df.iter_rows(named=True):
        qwen, heuristic = item['qwen_prediction'], item['heuristic_prediction']
        q_result = classify_outcome(qwen, item['long_outcome_12h'], item['short_outcome_12h'])
        h_result = classify_outcome(heuristic, item['long_outcome_12h'], item['short_outcome_12h'])
        kind, category = classify_disagreement(qwen, heuristic, q_result, h_result)
        available_at = item['available_at']
        rows.append({
            'time': int(item['timestamp'].timestamp()),
            'timestamp': item['timestamp'].isoformat(),
            'available_at': available_at.isoformat() if available_at is not None else None,
            'qwen': qwen,
            'heuristic': heuristic,
            'target': item['target'],
            'bias_score': item['bias_score'],
            'qwen_result': q_result,
            'heuristic_result': h_result,
            'type': kind,
            'category': category,
            'features': {
                'close': item['close'], 'rsi14': item['rsi14'], 'atr14_pct': item['atr14_pct'],
                'volume_ratio': item['volume_ratio'], 'trend_1h': item['trend_1h'],
                'trend_4h': item['4h_trend'], 'trend_1d': item['1d_trend'],
                'trend_alignment': item['trend_alignment'], 'features_ready': item['features_ready'],
            },
        })
    categories = ['SL_AVOIDED', 'TP_MISSED', 'TP_ADDED', 'SL_ADDED', 'UNDETERMINED']
    return {
        'dataset': dataset,
        'count': len(rows),
        'missing_market_rows': sum(row['available_at'] is None for row in rows),
        'counts': {
            'qwen_abstains': sum(row['type'] == 'QWEN_ABSTAINS' for row in rows),
            'qwen_adds_trade': sum(row['type'] == 'QWEN_ADDS_TRADE' for row in rows),
            'opposite_direction': sum(row['type'] == 'OPPOSITE_DIRECTION' for row in rows),
            'by_category': {category: sum(row['category'] == category for row in rows) for category in categories},
        },
        'rows': rows,
    }

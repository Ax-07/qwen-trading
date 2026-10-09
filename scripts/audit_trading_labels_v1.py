from __future__ import annotations

"""Audit read-only des labels first-touch BTC/USDC (4/12/24h).

Aucun modele entraine. Aucun seuil optimise. Les lignes TEST sont decrites,
jamais utilisees pour selectionner une strategie. Sorties CSV nouvelles uniquement.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path('data/processed/btc_usdc_1h_labeled.parquet')
SPLITS = Path('data/splits')
OUT = Path('data/evaluation/trading-labels-audit-v1')
HORIZONS = (4, 12, 24)
TP_ATR, SL_ATR = 1.5, 1.0


def audit_outcome(df, h, side, alternate_entry=False):
    """Retourne outcomes recalcules, delais premier toucher et incoherences.

    alternate_entry utilise open[i+1] au lieu de close[i], mais conserve ATR[i],
    les memes 12 bougies futures, et des stop/target au meme multiple d'ATR.
    Il ne constitue PAS une simulation d'execution ni un nouveau label valide.
    """
    n = len(df)
    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    close = df['close'].to_numpy(dtype=float)
    opening = df['open'].to_numpy(dtype=float)
    atr = df['atr14'].to_numpy(dtype=float)
    observed = np.full(n, np.nan)
    delay = np.full(n, np.nan)
    simultaneous = np.zeros(n, dtype=bool)
    entry_gap = np.full(n, np.nan)
    for i in range(n - h):
        entry = opening[i + 1] if alternate_entry else close[i]
        if np.isfinite(close[i]) and np.isfinite(opening[i+1]) and close[i] != 0:
            entry_gap[i] = 100.0 * (opening[i+1] / close[i] - 1.0)
        if not np.isfinite(entry) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if side == 'LONG':
            tp, sl = entry + TP_ATR * atr[i], entry - SL_ATR * atr[i]
        else:
            tp, sl = entry - TP_ATR * atr[i], entry + SL_ATR * atr[i]
        observed[i] = 0.0
        for k in range(i+1, i+h+1):
            tp_hit = high[k] >= tp if side == 'LONG' else low[k] <= tp
            sl_hit = low[k] <= sl if side == 'LONG' else high[k] >= sl
            if tp_hit or sl_hit:
                delay[i] = k-i
                if tp_hit and sl_hit:
                    simultaneous[i] = True
                    observed[i] = np.nan
                else:
                    observed[i] = 1.0 if tp_hit else -1.0
                break
    return observed, delay, simultaneous, entry_gap


def stats(values):
    s = pd.Series(values)
    tp, sl, no = (int((s == x).sum()) for x in (1, -1, 0))
    ambig = int(s.isna().sum())
    resolved = tp+sl
    return dict(n=len(s), tp=tp, sl=sl, no_touch=no, ambiguous_or_missing=ambig,
                tp_pct=100*tp/len(s) if len(s) else np.nan,
                sl_pct=100*sl/len(s) if len(s) else np.nan,
                no_touch_pct=100*no/len(s) if len(s) else np.nan,
                win_rate_resolved=100*tp/resolved if resolved else np.nan,
                r_per_signal_proxy=(TP_ATR*tp-SL_ATR*sl)/len(s) if len(s) else np.nan)


def main():
    print('=' * 78)
    print('AUDIT DES LABELS V1 - DESCRIPTIF, SANS ENTRAINEMENT')
    print('=' * 78)
    df = pd.read_parquet(DATA).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if df['timestamp'].isna().any() or df['timestamp'].duplicated().any():
        raise ValueError('Timestamps manquants ou dupliques')
    if not df['timestamp'].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Bougies horaires non contigues : audit interrompu')
    df['split'] = 'OUTSIDE_SPLITS'
    for split in ('train', 'validation', 'test'):
        src = pd.read_parquet(SPLITS / f'{split}.parquet', columns=['timestamp'])
        stamps = pd.to_datetime(src['timestamp'], utc=True)
        if stamps.duplicated().any() or not stamps.isin(df['timestamp']).all():
            raise ValueError(f'Split {split}: timestamp invalide')
        mask = df['timestamp'].isin(stamps)
        if (df.loc[mask, 'split'] != 'OUTSIDE_SPLITS').any():
            raise ValueError(f'Splits chevauchants: {split}')
        df.loc[mask, 'split'] = split.upper()
        print(f'{split.upper():10s}: {len(src):5d} lignes')
    df['atr_band'] = pd.cut(df['atr14_pct'],
        bins=[-np.inf, .5, 1., 1.5, np.inf],
        labels=['<=0.5%', '0.5-1%', '1-1.5%', '>1.5%']).astype('string').fillna('MISSING')
    print(f'Dataset complet: {len(df)} bougies')
    print('NB : split attribue par timestamp, resultats decrits sans tuning du Test.')

    outcome_rows, vol_rows, delay_rows, comparison_rows, gap_rows = [], [], [], [], []
    split_order = ['TRAIN', 'VALIDATION', 'TEST']
    for h in HORIZONS:
        for side in ('LONG', 'SHORT'):
            computed, delay, simultaneous, entry_gap = audit_outcome(df, h, side)
            original = df[f'{side.lower()}_outcome_{h}h'].to_numpy(dtype=float)
            for split in split_order:
                sel = (df['split'] == split).to_numpy()
                if not sel.any():
                    continue
                expected = original[sel]
                actual = computed[sel]
                # Use equal_nan to assert strict agreement on first-touch labels.
                agreement = np.isclose(expected, actual, equal_nan=True)
                mismatches = int((~agreement).sum())
                if mismatches:
                    sample = df.loc[sel, 'timestamp'].iloc[np.where(~agreement)[0][:3]].tolist()
                    raise RuntimeError(f'{split} {side} {h}h: {mismatches} divergences de labels, ex: {sample}')
                usable = sel & df['sample_ready'].fillna(False).to_numpy()
                base = stats(original[usable])
                outcome_rows.append({'split': split, 'side': side, 'horizon_h':h, **base,
                    'simultaneous':int(simultaneous[usable].sum()),
                    'other_nan':int(np.isnan(computed[usable]).sum()-simultaneous[usable].sum())})
                for band in ['<=0.5%', '0.5-1%', '1-1.5%', '>1.5%']:
                    subset = usable & (df['atr_band'] == band).to_numpy()
                    if subset.sum():
                        vol_rows.append({'split':split, 'side':side, 'horizon_h':h, 'atr_band':band,
                                         **stats(original[subset])})
                hit = usable & np.isfinite(delay)
                delay_rows.append({'split':split,'side':side,'horizon_h':h,
                                   'touches':int(hit.sum()),
                                   'median_hours':float(np.median(delay[hit])) if hit.any() else np.nan,
                                   'p90_hours':float(np.quantile(delay[hit],.9)) if hit.any() else np.nan,
                                   'touches_within_4h':int((delay[hit] <= 4).sum())})
                if h == 12:
                    alt, _, alt_simult, _ = audit_outcome(df, h, side, alternate_entry=True) if split == split_order[0] else (None,None,None,None)
                    # Compute once per side / horizon; cache for remaining splits.
                    if split == split_order[0]:
                        alt_cached = alt
                    valid_pairs = usable & np.isfinite(original) & np.isfinite(alt_cached)
                    changes = int((original[valid_pairs] != alt_cached[valid_pairs]).sum())
                    comparison_rows.append({'split':split,'side':side,
                        'comparable_non_ambiguous':int(valid_pairs.sum()),
                        'changed_labels_close_vs_next_open':changes,
                        'change_pct':100*changes/valid_pairs.sum() if valid_pairs.any() else np.nan,
                        'original_ambiguous':int((usable & simultaneous).sum()),
                        'next_open_ambiguous':int((usable & alt_simult).sum()) if split == split_order[0] else int((usable & np.isnan(alt_cached)).sum())})
                    finite_gap = usable & np.isfinite(entry_gap)
                    gap_rows.append({'split':split,'side':side,'n':int(finite_gap.sum()),
                        'gap_abs_median_pct':float(np.median(np.abs(entry_gap[finite_gap]))) if finite_gap.any() else np.nan,
                        'gap_abs_p95_pct':float(np.quantile(np.abs(entry_gap[finite_gap]), .95)) if finite_gap.any() else np.nan})

    outcome_df = pd.DataFrame(outcome_rows)
    vol_df = pd.DataFrame(vol_rows)
    delay_df = pd.DataFrame(delay_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    gap_df = pd.DataFrame(gap_rows)

    print('\n' + '='*78)
    print('OUTCOMES - sample_ready, TP/SL/NO_TOUCH/AMBIGU')
    print('='*78)
    cols = ['split','side','horizon_h','n','tp','sl','no_touch','simultaneous',
            'other_nan','win_rate_resolved','r_per_signal_proxy']
    print(outcome_df[cols].to_string(index=False, float_format=lambda x:f'{x:.3f}'))
    print('\n' + '='*78)
    print('DELAI DE PREMIER TOUCHER (en heures)')
    print('='*78)
    print(delay_df.to_string(index=False, float_format=lambda x:f'{x:.2f}'))
    print('\n' + '='*78)
    print('SENSIBILITE 12H : close[i] vs open[i+1]')
    print('='*78)
    print(comparison_df.to_string(index=False, float_format=lambda x:f'{x:.2f}'))
    print('\n' + '='*78)
    print('REGIMES ATR : premiers exemples sur VALIDATION')
    print('='*78)
    print(vol_df.loc[(vol_df.split=='VALIDATION') & (vol_df.horizon_h==12),
          ['side','atr_band','n','tp','sl','no_touch','ambiguous_or_missing',
           'r_per_signal_proxy']].to_string(index=False, float_format=lambda x:f'{x:.3f}'))

    # Fenetres se chevauchant : signal horaire a duree fixe, borne haute.
    # Calcul indicatif : overlap evenementiel dynamique non estime comme simultaneite de positions.
    print('\nCHEVAUCHEMENT THEORIQUE DES FENETRES DE 12H:')
    for split in split_order:
        stamps = df.loc[(df.split==split) & df.sample_ready.fillna(False),'timestamp'].sort_values()
        if len(stamps):
            close_pairs = int((stamps.diff().dropna()<pd.Timedelta(hours=12)).sum())
            print(f'  {split}: {close_pairs}/{max(len(stamps)-1,0)} paires de signaux successifs a moins de 12h')
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('outcome_distribution',outcome_df),
                        ('atr_regime_distribution',vol_df),
                        ('first_touch_delays',delay_df),
                        ('next_open_sensitivity',comparison_df),
                        ('entry_gap_summary',gap_df)]:
        frame.to_csv(OUT/f'{name}.csv', index=False)
    print(f'\nCSV produits : {OUT.resolve()}')
    print('Labels recomputes et verifies sur toutes les observations des splits.')
    print('NO_TOUCH=0R uniquement dans proxy; ni frais, ni sortie a 12H, ni capital.')
    print('TEST descriptif seulement : ne pas choisir de regles en fonction de ses resultats.')


if __name__ == '__main__':
    main()

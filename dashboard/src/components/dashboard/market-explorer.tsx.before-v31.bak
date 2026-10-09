import TradingChart, {
  type Candle,
  type SignalFilter,
  type SignalSettings,
  type TradingSignal,
} from "@/components/TradingChart";
import type { Dataset } from "./types";
import { FilterButton } from "./ui";

export function MarketExplorer({
  dataset,
  candles,
  signals,
  signalSettings,
  selectedTime,
  focusedTime,
  loading,
  error,
  signalError,
  showInspectors,
  onSignalSettingsChange,
  onSelectCandle,
  onToggleInspectors,
  onClearFocus,
}: {
  dataset: Dataset;
  candles: Candle[];
  signals: TradingSignal[];
  signalSettings: SignalSettings;
  selectedTime: number | null;
  focusedTime: number | null;
  loading: boolean;
  error: string | null;
  signalError: string | null;
  showInspectors: boolean;
  onSignalSettingsChange: (settings: SignalSettings) => void;
  onSelectCandle: (time: number) => void;
  onToggleInspectors: () => void;
  onClearFocus: () => void;
}) {
  const setSignalSettings = (
    update: (current: SignalSettings) => SignalSettings,
  ) => onSignalSettingsChange(update(signalSettings));
  const stats = [
    ["Signaux", signals.length, "text-slate-100"],
    [
      "LONG",
      signals.filter((signal) => signal.qwen === "LONG_BIAS").length,
      "text-emerald-400",
    ],
    [
      "SHORT",
      signals.filter((signal) => signal.qwen === "SHORT_BIAS").length,
      "text-rose-400",
    ],
    [
      "TP",
      signals.filter((signal) => signal.qwen_result === "TP").length,
      "text-emerald-400",
    ],
    [
      "SL",
      signals.filter((signal) => signal.qwen_result === "SL").length,
      "text-rose-400",
    ],
  ] as const;
  return (
    <section
      id="market-chart"
      className="scroll-mt-24 overflow-hidden rounded-xl border border-[#24334d] bg-[#0c1729] shadow-[0_8px_30px_rgba(0,0,0,.13)]"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#24334d] px-4 py-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">
            <span className="mr-2 inline-grid h-8 w-8 place-items-center rounded-full bg-orange-500 text-base text-white">
              ₿
            </span>{" "}
            BTCUSDC{" "}
            <span className="ml-2 rounded-md border border-blue-500/25 bg-blue-500/10 px-2 py-1 align-middle text-[10px] font-medium text-blue-300">
              {dataset.toUpperCase()}
            </span>
          </h2>
          <p className="mt-1 text-[11px] text-slate-500">
            Market Explorer · Données Binance historiques
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-[11px]">
          <span className="rounded-md border border-[#2a3c56] px-3 py-2 text-slate-300">
            {candles.length} bougies
          </span>
          <span className="rounded-md border border-[#2a3c56] px-3 py-2 text-slate-300">
            {signals.length} lignes d’évaluation
          </span>
          <button
            type="button"
            onClick={onToggleInspectors}
            className="rounded-md border border-blue-500/30 bg-blue-500/10 px-3 py-2 font-medium text-blue-200 hover:bg-blue-500/20"
          >
            {showInspectors
              ? "Masquer les inspecteurs"
              : "Afficher les inspecteurs"}
          </button>
        </div>
      </div>
      <div className="border-b border-[#24334d] px-4 py-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-[.12em] text-slate-500">
            Vue de marché · fenêtre chargée
          </span>
          <span className="text-[10px] text-slate-500">
            {candles.length} bougies · {signals.length} évaluations
          </span>
        </div>
        <div className="grid grid-cols-5 gap-2">
          {stats.map(([label, value, color]) => (
            <div
              key={label}
              className="flex min-w-0 items-center justify-between gap-2 rounded-lg border border-[#294260] bg-[#11243c] px-3 py-2.5"
            >
              <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                {label}
              </span>
              <strong className={`text-base font-bold tabular-nums ${color}`}>
                {value}
              </strong>
            </div>
          ))}
        </div>
      </div>
      <SignalControls settings={signalSettings} onChange={setSignalSettings} />
      {signalError && (
        <p className="mx-4 mt-3 rounded-md border border-rose-500/30 bg-rose-500/5 p-3 text-xs text-rose-300">
          {signalError}
        </p>
      )}
      {focusedTime !== null && (
        <div className="mx-4 mt-3 flex flex-wrap items-center justify-between gap-2 rounded-md border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-300">
          <span>
            Exploration d’un désaccord historique ·{" "}
            {new Date(focusedTime * 1000)
              .toISOString()
              .slice(0, 16)
              .replace("T", " ")}{" "}
            UTC
          </span>
          <button
            type="button"
            onClick={onClearFocus}
            className="rounded-md border border-amber-500/40 px-3 py-1.5 transition hover:bg-amber-500/10"
          >
            Retour aux 500 dernières bougies
          </button>
        </div>
      )}
      {loading ? (
        <p className="px-5 py-24 text-center text-sm text-slate-400">
          Chargement du graphique…
        </p>
      ) : error ? (
        <p className="p-6 text-sm text-red-400">{error}</p>
      ) : candles.length === 0 ? (
        <p className="p-8 text-sm text-slate-400">
          Aucune bougie disponible pour cette période.
        </p>
      ) : (
        <div className="px-3 py-2">
          <TradingChart
            key={`${dataset}-${focusedTime ?? "latest"}`}
            candles={candles}
            signals={signals}
            signalSettings={signalSettings}
            selectedTime={selectedTime}
            onSelectCandle={onSelectCandle}
          />
        </div>
      )}
      <div className="border-t border-[#24334d] px-4 py-3 text-[11px] text-slate-500">
        Source : FastAPI / Parquet · Analyse historique, pas de trading en
        direct
      </div>
    </section>
  );
}

function SignalControls({
  settings,
  onChange,
}: {
  settings: SignalSettings;
  onChange: (update: (current: SignalSettings) => SignalSettings) => void;
}) {
  const filters: [SignalFilter, string][] = [
    ["all", "Tous"],
    ["win", "Gagnants (TP)"],
    ["loss", "Perdants (SL)"],
    ["disagree", "Désaccords"],
  ];
  return (
    <div className="border-b border-[#24334d] bg-[#0d1a2e] px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="shrink-0 text-[11px] font-semibold uppercase tracking-wide text-slate-200">
          Signal Explorer
        </span>
        {(
          [
            ["qwen", "Qwen V2"],
            ["heuristic", "Heuristique"],
            ["target", "Target (futur)"],
          ] as const
        ).map(([key, label]) => (
          <label
            key={key}
            className="inline-flex cursor-pointer items-center gap-1.5 whitespace-nowrap text-[11px] text-slate-300"
          >
            <input
              type="checkbox"
              checked={settings[key]}
              onChange={(event) =>
                onChange((current) => ({
                  ...current,
                  [key]: event.target.checked,
                }))
              }
              className="accent-blue-500"
            />
            {label}
          </label>
        ))}
        <span className="ml-auto text-[10px] text-slate-500">
          Targets : évaluation a posteriori
        </span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {filters.map(([id, label]) => (
          <FilterButton
            key={id}
            active={settings.filter === id}
            onClick={() => onChange((current) => ({ ...current, filter: id }))}
          >
            {label}
          </FilterButton>
        ))}
        <span className="ml-auto text-[10px] text-slate-500">
          ▲ LONG · ▼ SHORT
        </span>
      </div>
    </div>
  );
}

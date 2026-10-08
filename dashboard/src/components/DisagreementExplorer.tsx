"use client";

import { useEffect, useMemo, useState } from "react";

type Dataset = "train" | "validation" | "test";
type Category =
  "SL_AVOIDED" | "TP_MISSED" | "TP_ADDED" | "SL_ADDED" | "UNDETERMINED";
type DisagreementRow = {
  time: number;
  timestamp: string;
  available_at: string | null;
  qwen: string;
  heuristic: string;
  target: string;
  bias_score: number;
  qwen_result: string;
  heuristic_result: string;
  type: string;
  category: Category;
  features: {
    close: number | null;
    rsi14: number | null;
    atr14_pct: number | null;
    volume_ratio: number | null;
    trend_1h: number | null;
    trend_4h: number | null;
    trend_1d: number | null;
    trend_alignment: number | null;
    features_ready: boolean | null;
  };
};
type ResponseData = {
  dataset: string;
  count: number;
  missing_market_rows: number;
  counts: {
    qwen_abstains: number;
    qwen_adds_trade: number;
    opposite_direction: number;
    by_category: Record<Category, number>;
  };
  rows: DisagreementRow[];
};
const FILTERS: { value: Category | "ALL"; label: string }[] = [
  { value: "ALL", label: "Tous" },
  { value: "SL_AVOIDED", label: "SL évités" },
  { value: "TP_MISSED", label: "TP manqués" },
  { value: "TP_ADDED", label: "TP ajoutés" },
  { value: "SL_ADDED", label: "SL ajoutés" },
  { value: "UNDETERMINED", label: "Indéterminés" },
];
const LABELS: Record<Category, string> = {
  SL_AVOIDED: "SL évité",
  TP_MISSED: "TP manqué",
  TP_ADDED: "TP ajouté",
  SL_ADDED: "SL ajouté",
  UNDETERMINED: "Indéterminé",
};
function fmt(value: number | null): string {
  return value === null || !Number.isFinite(value)
    ? "—"
    : value.toLocaleString("fr-FR", { maximumFractionDigits: 2 });
}
function dt(value: string): string {
  return new Date(value).toISOString().replace("T", " ").slice(0, 16) + " UTC";
}
function decisionColor(value: string) {
  return value === "LONG_BIAS"
    ? "text-emerald-400"
    : value === "SHORT_BIAS"
      ? "text-red-400"
      : "text-slate-400";
}
function categoryColor(value: Category) {
  return value === "SL_AVOIDED" || value === "TP_ADDED"
    ? "text-emerald-400"
    : value === "TP_MISSED" || value === "SL_ADDED"
      ? "text-red-400"
      : "text-slate-400";
}
export default function DisagreementExplorer({
  dataset,
  selectedTime,
  onSelect,
}: {
  dataset: Dataset;
  selectedTime: number | null;
  onSelect: (time: number) => void;
}) {
  const [data, setData] = useState<ResponseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<Category | "ALL">("ALL");
  const [search, setSearch] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    if (dataset === "train") return () => controller.abort();
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/api/analytics/disagreements?dataset=${dataset}`,
          { signal: controller.signal },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result: ResponseData = await response.json();
        if (!controller.signal.aborted) setData(result);
      } catch (err) {
        if (!controller.signal.aborted)
          setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [dataset]);

  const activeData = data?.dataset === dataset ? data : null;
  const rows = useMemo(() => {
    if (!activeData) return [];
    const query = search.trim().toLowerCase();
    return activeData.rows.filter((row) => {
      if (filter !== "ALL" && row.category !== filter) return false;
      return (
        !query ||
        [row.timestamp, row.qwen, row.heuristic, row.target, row.category]
          .join(" ")
          .toLowerCase()
          .includes(query)
      );
    });
  }, [activeData, filter, search]);

  const categories: { id: Category; label: string; tone: string }[] = [
    { id: "SL_AVOIDED", label: "SL évités", tone: "text-emerald-400" },
    { id: "TP_MISSED", label: "TP manqués", tone: "text-rose-400" },
    { id: "TP_ADDED", label: "TP ajoutés", tone: "text-blue-400" },
    { id: "SL_ADDED", label: "SL ajoutés", tone: "text-amber-400" },
    { id: "UNDETERMINED", label: "Indéterminés", tone: "text-slate-400" },
  ];
  return (
    <section className="q-panel overflow-hidden text-slate-100">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#283c59] px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="text-xl text-blue-400">⚖</span>
          <div>
            <h2 className="text-base font-semibold">Disagreement Explorer</h2>
            <p className="text-[11px] text-slate-500">
              Quand Qwen V2 et l&apos;heuristique prennent des décisions
              différentes
            </p>
          </div>
        </div>
        <span className="text-[10px] font-medium text-slate-400">
          {dataset.toUpperCase()}
        </span>
      </div>
      <div className="p-4">
        {dataset === "train" ? (
          <p className="text-sm text-slate-400">
            Pas de résultats Qwen V2 sur Train.
          </p>
        ) : loading || !activeData ? (
          <p className="text-sm text-slate-400">
            {error ?? "Chargement des désaccords…"}
          </p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
              <div className="rounded-lg border border-[#283c59] bg-[#11213b] p-3">
                <p className="text-[10px] text-slate-400">Désaccords</p>
                <p className="mt-1 text-xl font-bold">{activeData.count}</p>
                <p className="text-[10px] text-slate-500">
                  {activeData.counts.qwen_abstains} abstentions Qwen
                </p>
              </div>
              {categories.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  onClick={() =>
                    setFilter(filter === item.id ? "ALL" : item.id)
                  }
                  className={`rounded-lg border p-3 text-left transition ${filter === item.id ? "border-blue-500 bg-blue-500/15" : "border-[#283c59] bg-[#11213b] hover:border-[#42678d]"}`}
                >
                  <p className="text-[10px] text-slate-400">{item.label}</p>
                  <p className={`mt-1 text-xl font-bold ${item.tone}`}>
                    {activeData.counts.by_category[item.id]}
                  </p>
                  <p className="text-[10px] text-slate-500">
                    {activeData.count
                      ? fmt(
                          (activeData.counts.by_category[item.id] /
                            activeData.count) *
                            100,
                        )
                      : "—"}{" "}
                    % du total
                  </p>
                </button>
              ))}
            </div>
            {activeData.missing_market_rows > 0 && (
              <p className="text-xs text-amber-300">
                {activeData.missing_market_rows} événements sans features marché
                correspondantes.
              </p>
            )}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-1.5">
                {FILTERS.map((opt) => (
                  <button
                    type="button"
                    key={opt.value}
                    onClick={() => setFilter(opt.value)}
                    className={`rounded-md border px-3 py-2 text-[11px] font-medium transition ${filter === opt.value ? "border-blue-500 bg-blue-600 text-white" : "border-[#293d5b] bg-[#102038] text-slate-400 hover:text-white"}`}
                  >
                    {opt.label}
                    {opt.value !== "ALL"
                      ? ` (${activeData.counts.by_category[opt.value]})`
                      : ""}
                  </button>
                ))}
              </div>
              <input
                type="search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="⌕ Rechercher date ou décision…"
                className="w-full rounded-md border border-[#324765] bg-[#0a182b] px-3 py-2 text-xs text-slate-100 outline-none focus:border-blue-500 sm:w-72"
              />
            </div>
            <div className="max-h-[460px] overflow-auto rounded-lg border border-[#283c59] q-scroll">
              <table className="q-table min-w-[950px]">
                <thead className="sticky top-0 z-10">
                  <tr>
                    {[
                      "Date UTC",
                      "Qwen V2",
                      "Heuristique",
                      "Catégorie",
                      "Résultat Qwen",
                      "Résultat Heur.",
                      "RSI14",
                      "ATR %",
                      "Volume ratio",
                      "Bias",
                    ].map((t) => (
                      <th key={t}>{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr
                      key={row.time}
                      tabIndex={0}
                      role="button"
                      onClick={() => onSelect(row.time)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          onSelect(row.time);
                        }
                      }}
                      data-selected={row.time === selectedTime}
                      className={`cursor-pointer transition ${selectedTime === row.time ? "bg-amber-500/10 outline outline-1 outline-amber-500/30" : ""}`}
                    >
                      <td className="whitespace-nowrap font-mono">
                        {dt(row.timestamp)}
                      </td>
                      <td>
                        <span
                          className={`font-semibold ${decisionColor(row.qwen)}`}
                        >
                          {row.qwen}
                        </span>
                      </td>
                      <td>
                        <span
                          className={`font-semibold ${decisionColor(row.heuristic)}`}
                        >
                          {row.heuristic}
                        </span>
                      </td>
                      <td>
                        <span
                          className={`font-semibold ${categoryColor(row.category)}`}
                        >
                          {LABELS[row.category]}
                        </span>
                      </td>
                      <td>{row.qwen_result}</td>
                      <td>{row.heuristic_result}</td>
                      <td>{fmt(row.features.rsi14)}</td>
                      <td>{fmt(row.features.atr14_pct)}</td>
                      <td>{fmt(row.features.volume_ratio)}</td>
                      <td>{row.bias_score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!rows.length && (
                <p className="p-5 text-center text-xs text-slate-400">
                  Aucun événement pour ces filtres.
                </p>
              )}
            </div>
            <div className="flex flex-wrap justify-between gap-2 text-[10px] text-slate-500">
              <span>
                {rows.length} événements affichés · Clique sur une ligne pour
                inspecter la bougie.
              </span>
              <span>
                Résultats TP/SL a posteriori sur 12H · Pas de trades exécutés
              </span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

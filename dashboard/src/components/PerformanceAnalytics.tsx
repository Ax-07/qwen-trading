"use client";

import { useEffect, useState } from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Dataset = "train" | "validation" | "test";

type TradeStats = {
  samples: number;
  long: number;
  short: number;
  no_trade: number;
  directional_signals: number;
  tp: number;
  sl: number;
  unresolved: number;
  unknown: number;
  resolved: number;
  coverage_pct: number | null;
  win_rate_pct: number | null;
  total_r: number;
  expected_r: number | null;
};

type Summary = {
  dataset: string;
  samples: number;
  parameters: {
    tp_atr: number;
    sl_atr: number;
    horizon_hours: number;
    tp_r: number;
    sl_r: number;
    expected_r_denominator: string;
  };
  qwen: TradeStats;
  heuristic: TradeStats;
  agreement: {
    samples: number;
    agreements: number;
    disagreements: number;
    agreement_pct: number | null;
  };
  disagreements: {
    samples: number;
    qwen: TradeStats;
    heuristic: TradeStats;
    qwen_trade_heuristic_abstains: number;
    heuristic_trade_qwen_abstains: number;
    both_trade_different_direction: number;
    qwen_tp_other_not_tp: number;
    heuristic_tp_other_not_tp: number;
    both_tp: number;
    both_sl: number;
  };
  by_direction: {
    qwen: {
      long: TradeStats;
      short: TradeStats;
    };
    heuristic: {
      long: TradeStats;
      short: TradeStats;
    };
  };
  disclaimer: string;
};

type View = "overview" | "directions" | "disagreements";

function fmt(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toLocaleString("fr-FR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function signed(value: number | null, digits = 4): string {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${fmt(value, digits)}`;
}

function valueColor(value: number | null): string {
  if (value === null) return "text-slate-400";
  if (value > 0) return "text-emerald-400";
  if (value < 0) return "text-red-400";
  return "text-slate-200";
}

function Metric({
  title,
  value,
  detail,
  tone,
}: {
  title: string;
  value: string;
  detail?: string;
  tone?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-[#0e1829] p-4">
      <p className="text-xs text-slate-400">{title}</p>
      <p className={`mt-2 text-2xl font-semibold ${tone ?? "text-white"}`}>
        {value}
      </p>
      {detail && <p className="mt-2 text-xs text-slate-500">{detail}</p>}
    </div>
  );
}

function ResultTable({
  qwen,
  heuristic,
}: {
  qwen: TradeStats;
  heuristic: TradeStats;
}) {
  const rows: {
    label: string;
    qwen: string;
    heuristic: string;
  }[] = [
    {
      label: "LONG",
      qwen: fmt(qwen.long, 0),
      heuristic: fmt(heuristic.long, 0),
    },
    {
      label: "SHORT",
      qwen: fmt(qwen.short, 0),
      heuristic: fmt(heuristic.short, 0),
    },
    {
      label: "NO_TRADE",
      qwen: fmt(qwen.no_trade, 0),
      heuristic: fmt(heuristic.no_trade, 0),
    },
    {
      label: "Signaux directionnels",
      qwen: fmt(qwen.directional_signals, 0),
      heuristic: fmt(heuristic.directional_signals, 0),
    },
    {
      label: "TP",
      qwen: fmt(qwen.tp, 0),
      heuristic: fmt(heuristic.tp, 0),
    },
    {
      label: "SL",
      qwen: fmt(qwen.sl, 0),
      heuristic: fmt(heuristic.sl, 0),
    },
    {
      label: "Non résolus",
      qwen: fmt(qwen.unresolved, 0),
      heuristic: fmt(heuristic.unresolved, 0),
    },
    {
      label: "Inconnus",
      qwen: fmt(qwen.unknown, 0),
      heuristic: fmt(heuristic.unknown, 0),
    },
    {
      label: "Win rate résolus",
      qwen: `${fmt(qwen.win_rate_pct)} %`,
      heuristic: `${fmt(heuristic.win_rate_pct)} %`,
    },
    {
      label: "Coverage",
      qwen: `${fmt(qwen.coverage_pct)} %`,
      heuristic: `${fmt(heuristic.coverage_pct)} %`,
    },
    {
      label: "Expected R",
      qwen: signed(qwen.expected_r),
      heuristic: signed(heuristic.expected_r),
    },
    {
      label: "Total R",
      qwen: signed(qwen.total_r, 2),
      heuristic: signed(heuristic.total_r, 2),
    },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-left text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-slate-400">
            <th className="py-3 font-medium">Métrique</th>
            <th className="py-3 text-right font-medium">Qwen V2</th>
            <th className="py-3 text-right font-medium">Heuristique</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-slate-800/70">
              <td className="py-2.5 text-slate-300">{row.label}</td>
              <td className="py-2.5 text-right font-mono text-slate-100">
                {row.qwen}
              </td>
              <td className="py-2.5 text-right font-mono text-slate-100">
                {row.heuristic}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RateBar({
  label,
  value,
  detail,
}: {
  label: string;
  value: number | null;
  detail?: string;
}) {
  const percentage = Math.max(0, Math.min(100, value ?? 0));

  return (
    <div className="space-y-2">
      <div className="flex justify-between gap-3 text-xs">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono text-slate-100">{fmt(value)} %</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-blue-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
      {detail && <p className="text-xs text-slate-500">{detail}</p>}
    </div>
  );
}

function DirectionCard({ title, stats }: { title: string; stats: TradeStats }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-[#0e1829] p-4">
      <h4 className="mb-4 font-semibold">{title}</h4>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-slate-400">Signaux</p>
          <p className="mt-1 text-xl font-semibold">
            {fmt(stats.directional_signals, 0)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Win rate</p>
          <p className="mt-1 text-xl font-semibold">
            {fmt(stats.win_rate_pct)} %
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">TP / SL</p>
          <p className="mt-1 text-sm font-mono">
            <span className="text-emerald-400">{stats.tp}</span>
            {" / "}
            <span className="text-red-400">{stats.sl}</span>
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Expected R</p>
          <p
            className={`mt-1 text-lg font-semibold ${valueColor(stats.expected_r)}`}
          >
            {signed(stats.expected_r)} R
          </p>
        </div>
      </div>
      <p className="mt-4 text-xs text-slate-500">
        {stats.unresolved} non résolus · {stats.unknown} inconnus
      </p>
    </div>
  );
}

export default function PerformanceAnalytics({
  dataset,
}: {
  dataset: Dataset;
}) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("overview");
  useEffect(() => {
    const controller = new AbortController();
    if (dataset === "train") return () => controller.abort();
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/api/analytics/summary?dataset=${dataset}`,
          { signal: controller.signal },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data: Summary = await response.json();
        if (!controller.signal.aborted) setSummary(data);
      } catch (e) {
        if (!controller.signal.aborted) setError(String(e));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [dataset]);
  const data = summary?.dataset === dataset ? summary : null;
  const stats = (
    label: string,
    val: number | null,
    helper: string,
    colored = false,
  ) => (
    <div
      className="rounded-lg border border-[#273b59] bg-[#11213b] px-4 py-3"
      key={label}
    >
      <p className="text-[11px] text-slate-400">{label}</p>
      <div
        className={`mt-1.5 text-xl font-bold tabular-nums ${colored ? valueColor(val) : "text-slate-100"}`}
      >
        {colored ? signed(val) : fmt(val, 0)}
      </div>
      <p className="mt-1 text-[10px] text-slate-500">{helper}</p>
    </div>
  );
  return (
    <section className="q-panel overflow-hidden text-slate-100">
      <Tabs
        value={view}
        onValueChange={(value) => setView(value as View)}
        className="flex flex-wrap items-center gap-2"
      >
        <TabsList className="border border-[#293c5b] bg-[#101f37]">
          <TabsTrigger
            value="overview"
            className="text-xs text-slate-400 data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            Vue globale
          </TabsTrigger>
          <TabsTrigger
            value="directions"
            className="text-xs text-slate-400 data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            LONG vs SHORT
          </TabsTrigger>
          <TabsTrigger
            value="disagreements"
            className="text-xs text-slate-400 data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            Désaccords
          </TabsTrigger>
        </TabsList>
        <span className="ml-2 text-[10px] text-slate-500">
          {dataset.toUpperCase()}
        </span>
      </Tabs>
      <div className="p-4">
        {dataset === "train" ? (
          <p className="py-4 text-xs text-slate-400">
            Aucune évaluation Qwen V2 sur Train.
          </p>
        ) : loading || !data ? (
          <p className="py-5 text-xs text-slate-400">
            {error ?? "Chargement des statistiques…"}
          </p>
        ) : (
          <>
            {view === "overview" && (
              <div className="space-y-3">
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6">
                  {stats(
                    "Expected R · Qwen",
                    data.qwen.expected_r,
                    "Sur trades résolus",
                    true,
                  )}
                  {stats(
                    "Expected R · Heuristique",
                    data.heuristic.expected_r,
                    "Sur trades résolus",
                    true,
                  )}
                  {stats(
                    "Accord modèles",
                    data.agreement.agreement_pct,
                    "%",
                    false,
                  )}
                  {stats("Observations", data.samples, "Dataset complet")}
                  {stats(
                    "Win rate · Qwen",
                    data.qwen.win_rate_pct,
                    "% des trades résolus",
                  )}
                  {stats(
                    "Win rate · Heuristique",
                    data.heuristic.win_rate_pct,
                    "% des trades résolus",
                  )}
                </div>
                <div className="overflow-x-auto rounded-lg border border-[#283c59]">
                  <table className="q-table min-w-[840px]">
                    <thead>
                      <tr>
                        <th>Modèle</th>
                        <th>LONG</th>
                        <th>SHORT</th>
                        <th>NO_TRADE</th>
                        <th>TP / SL</th>
                        <th>Non résolus</th>
                        <th>Couverture</th>
                        <th>Win rate</th>
                        <th>Expected R</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(
                        [
                          ["Qwen V2", data.qwen],
                          ["Heuristique", data.heuristic],
                        ] as [string, TradeStats][]
                      ).map(([label, r]) => (
                        <tr key={label}>
                          <td className="font-semibold text-slate-100">
                            {label}
                          </td>
                          <td>{r.long}</td>
                          <td>{r.short}</td>
                          <td>{r.no_trade}</td>
                          <td>
                            <span className="text-emerald-400">{r.tp}</span> /{" "}
                            <span className="text-rose-400">{r.sl}</span>
                          </td>
                          <td>{r.unresolved}</td>
                          <td>{fmt(r.coverage_pct)} %</td>
                          <td>{fmt(r.win_rate_pct)} %</td>
                          <td
                            className={`font-semibold ${valueColor(r.expected_r)}`}
                          >
                            {signed(r.expected_r)} R
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {view === "directions" && (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 2xl:grid-cols-4">
                {(
                  [
                    ["Qwen · LONG", data.by_direction.qwen.long],
                    ["Qwen · SHORT", data.by_direction.qwen.short],
                    ["Heuristique · LONG", data.by_direction.heuristic.long],
                    ["Heuristique · SHORT", data.by_direction.heuristic.short],
                  ] as [string, TradeStats][]
                ).map(([name, r]) => (
                  <div
                    key={name}
                    className="rounded-lg border border-[#283c59] bg-[#11213b] p-4"
                  >
                    <h3 className="mb-4 text-xs font-semibold">{name}</h3>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-[10px] text-slate-400">Signaux</p>
                        <p className="text-xl font-bold">
                          {r.directional_signals}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-slate-400">Win rate</p>
                        <p className="text-xl font-bold">
                          {fmt(r.win_rate_pct)} %
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-slate-400">TP / SL</p>
                        <p className="text-sm">
                          <span className="text-emerald-400">{r.tp}</span> /{" "}
                          <span className="text-rose-400">{r.sl}</span>
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-slate-400">Expected R</p>
                        <p
                          className={`text-sm font-bold ${valueColor(r.expected_r)}`}
                        >
                          {signed(r.expected_r)} R
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {view === "disagreements" && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  {stats(
                    "Désaccords",
                    data.disagreements.samples,
                    "Échantillon complet",
                  )}
                  {stats(
                    "Qwen trade seul",
                    data.disagreements.qwen_trade_heuristic_abstains,
                    "Heuristique abstention",
                  )}
                  {stats(
                    "Heuristique trade seule",
                    data.disagreements.heuristic_trade_qwen_abstains,
                    "Qwen abstention",
                  )}
                  {stats(
                    "Directions opposées",
                    data.disagreements.both_trade_different_direction,
                    "Deux signaux directionnels",
                  )}
                </div>
                <div className="overflow-x-auto rounded-lg border border-[#283c59]">
                  <table className="q-table">
                    <thead>
                      <tr>
                        <th>Sur les désaccords</th>
                        <th>Qwen V2</th>
                        <th>Heuristique</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Trades résolus</td>
                        <td>{data.disagreements.qwen.resolved}</td>
                        <td>{data.disagreements.heuristic.resolved}</td>
                      </tr>
                      <tr>
                        <td>TP / SL</td>
                        <td>
                          {data.disagreements.qwen.tp} /{" "}
                          {data.disagreements.qwen.sl}
                        </td>
                        <td>
                          {data.disagreements.heuristic.tp} /{" "}
                          {data.disagreements.heuristic.sl}
                        </td>
                      </tr>
                      <tr>
                        <td>Expected R</td>
                        <td>{signed(data.disagreements.qwen.expected_r)} R</td>
                        <td>
                          {signed(data.disagreements.heuristic.expected_r)} R
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            <p className="mt-3 text-[10px] leading-relaxed text-slate-500">
              TP = +{fmt(data.parameters.tp_r)} R · SL ={" "}
              {signed(data.parameters.sl_r)} R · Horizon{" "}
              {data.parameters.horizon_hours}H. Statistiques historiques, hors
              frais, slippage et positions simultanées. Ce n’est pas un backtest
              exécutable.
            </p>
          </>
        )}
      </div>
    </section>
  );
}

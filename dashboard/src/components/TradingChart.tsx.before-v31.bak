
"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";

import { VerticalLine } from
  "@tradingview/lwc-plugin-vertical-line";

// --------------------------------------------------
// TYPES
// --------------------------------------------------

export type Candle = {
  time: number;
  available_at: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ema20: number | null;
  ema50: number | null;
  ema200: number | null;
  rsi14: number | null;
};

export type TradingSignal = {
  time: number;
  qwen: string;
  heuristic: string;
  target: string;
  bias_score: number;
  qwen_result: string;
  heuristic_result: string;
  target_result: string;
  disagreement: boolean;
};

export type SignalSource =
  | "qwen"
  | "heuristic"
  | "target";

export type SignalFilter =
  | "all"
  | "win"
  | "loss"
  | "disagree";

export type SignalSettings = {
  qwen: boolean;
  heuristic: boolean;
  target: boolean;
  filter: SignalFilter;
};

type Props = {
  candles: Candle[];
  signals?: TradingSignal[];
  signalSettings?: SignalSettings;
  selectedTime?: number | null;
  onSelectCandle?: (timestamp: number) => void;
};

type EmaField = "ema20" | "ema50" | "ema200";

type VisibleSignal = {
  time: number;
  source: SignalSource;
  decision: "LONG_BIAS" | "SHORT_BIAS";
  result: string;
};

const SELECTION_COLOR = "#fbbf24";

const DEFAULT_SETTINGS: SignalSettings = {
  qwen: true,
  heuristic: false,
  target: false,
  filter: "all",
};

const SOURCES: SignalSource[] = [
  "qwen",
  "heuristic",
  "target",
];

// --------------------------------------------------
// FILTRAGE DES SIGNAUX
// --------------------------------------------------

function getVisibleSignals(
  candles: Candle[],
  signals: TradingSignal[],
  settings: SignalSettings
): VisibleSignal[] {
  const validTimes = new Set(
    candles.map((c) => c.time)
  );

  const result: VisibleSignal[] = [];

  for (const signal of signals) {
    if (!validTimes.has(signal.time)) {
      continue;
    }

    if (
      settings.filter === "disagree" &&
      !signal.disagreement
    ) {
      continue;
    }

    for (const source of SOURCES) {
      if (!settings[source]) {
        continue;
      }

      const decision = signal[source];

      if (
        decision !== "LONG_BIAS" &&
        decision !== "SHORT_BIAS"
      ) {
        continue;
      }

      const outcomeKey =
        `${source}_result` as
          | "qwen_result"
          | "heuristic_result"
          | "target_result";

      const outcome = signal[outcomeKey];

      if (
        settings.filter === "win" &&
        outcome !== "TP"
      ) {
        continue;
      }

      if (
        settings.filter === "loss" &&
        outcome !== "SL"
      ) {
        continue;
      }

      result.push({
        time: signal.time,
        source,
        decision,
        result: outcome,
      });
    }
  }

  return result.sort(
    (a, b) => a.time - b.time
  );
}

// --------------------------------------------------
// MARQUEURS
// --------------------------------------------------

function buildMarkers(
  signals: VisibleSignal[],
  source: SignalSource
): SeriesMarker<Time>[] {
  const markers: SeriesMarker<Time>[] = [];

  for (const signal of signals) {
    if (signal.source !== source) {
      continue;
    }

    const isLong =
      signal.decision === "LONG_BIAS";

    let shape:
      | "arrowUp"
      | "arrowDown"
      | "circle"
      | "square";

    let color: string;
    let size: number;

    if (source === "qwen") {
      shape = isLong ? "arrowUp" : "arrowDown";
      color = isLong ? "#22c55e" : "#ef4444";
      size = 1.25;
    } else if (source === "heuristic") {
      shape = "circle";
      color = isLong ? "#34d399" : "#fb7185";
      size = 0.8;
    } else {
      shape = "square";
      color = isLong ? "#60a5fa" : "#c084fc";
      size = 0.8;
    }

    markers.push({
      time: signal.time as UTCTimestamp,
      position: isLong ? "belowBar" : "aboveBar",
      shape,
      color,
      size,
    });
  }

  return markers;
}

// --------------------------------------------------
// STATISTIQUES DES SIGNAUX VISIBLES
// --------------------------------------------------

function getSignalStats(signals: VisibleSignal[]) {
  return {
    total: signals.length,

    long: signals.filter(
      (s) => s.decision === "LONG_BIAS"
    ).length,

    short: signals.filter(
      (s) => s.decision === "SHORT_BIAS"
    ).length,

    tp: signals.filter(
      (s) => s.result === "TP"
    ).length,

    sl: signals.filter(
      (s) => s.result === "SL"
    ).length,

    other: signals.filter(
      (s) => s.result !== "TP" && s.result !== "SL"
    ).length,
  };
}

// --------------------------------------------------
// TRADING CHART
// --------------------------------------------------

export default function TradingChart({
  candles,
  signals = [],
  signalSettings = DEFAULT_SETTINGS,
  selectedTime = null,
  onSelectCandle,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const onSelectRef = useRef(onSelectCandle);

  const updateSelectionRef = useRef<
    ((time: number | null) => void) | null
  >(null);

  const markersRef = useRef<
    Partial<
      Record<
        SignalSource,
        ISeriesMarkersPluginApi<Time>
      >
    >
  >({});

  const [copyStatus, setCopyStatus] = useState("");

  const visibleSignals = useMemo(
    () =>
      getVisibleSignals(
        candles,
        signals,
        signalSettings
      ),
    [candles, signals, signalSettings]
  );

  const stats = useMemo(
    () => getSignalStats(visibleSignals),
    [visibleSignals]
  );

  useEffect(() => {
    onSelectRef.current = onSelectCandle;
  }, [onSelectCandle]);

  // --------------------------------------------------
  // INITIALISATION DU GRAPHIQUE
  // --------------------------------------------------

  useEffect(() => {
    const container = containerRef.current;

    if (!container || candles.length === 0) {
      return;
    }

    const chart = createChart(container, {
      autoSize: true,

      layout: {
        background: {
          type: ColorType.Solid,
          color: "#0b1220",
        },
        textColor: "#94a3b8",
        attributionLogo: true,

        panes: {
          separatorColor: "#334155",
          separatorHoverColor: "#475569",
          enableResize: true,
        },
      },

      grid: {
        vertLines: {
          color: "#1e293b",
        },
        horzLines: {
          color: "#1e293b",
        },
      },

      rightPriceScale: {
        borderColor: "#334155",
      },

      timeScale: {
        borderColor: "#334155",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
        barSpacing: 8,
      },

      crosshair: {
        mode: CrosshairMode.Normal,
      },
    });

    chartRef.current = chart;

    // --------------------------------------------------
    // PANE 0 : CHANDELIERS
    // --------------------------------------------------

    const candleSeries = chart.addSeries(
      CandlestickSeries,
      {
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderVisible: false,
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
      },
      0
    );

    candleSeries.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
    );

    // --------------------------------------------------
    // EMA
    // --------------------------------------------------

    function addEma(
      field: EmaField,
      color: string
    ) {
      const series = chart.addSeries(
        LineSeries,
        {
          color,
          lineWidth: 2,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        },
        0
      );

      series.setData(
        candles
          .filter(
            (c) =>
              c[field] !== null &&
              Number.isFinite(c[field])
          )
          .map((c) => ({
            time: c.time as UTCTimestamp,
            value: c[field] as number,
          }))
      );
    }

    addEma("ema20", "#f59e0b");
    addEma("ema50", "#3b82f6");
    addEma("ema200", "#a855f7");

    // --------------------------------------------------
    // PANE 1 : VOLUME
    // --------------------------------------------------

    const volumeSeries = chart.addSeries(
      HistogramSeries,
      {
        priceFormat: {
          type: "volume",
        },
        priceScaleId: "right",
        priceLineVisible: false,
        lastValueVisible: false,
      },
      1
    );

    volumeSeries.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        value: c.volume,
        color:
          c.close >= c.open
            ? "rgba(34,197,94,0.65)"
            : "rgba(239,68,68,0.65)",
      }))
    );

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.15,
        bottom: 0,
      },
    });

    // --------------------------------------------------
    // PANE 2 : RSI
    // --------------------------------------------------

    const rsiSeries = chart.addSeries(
      LineSeries,
      {
        color: "#a78bfa",
        lineWidth: 2,
        priceLineVisible: false,

        priceFormat: {
          type: "price",
          precision: 2,
          minMove: 0.01,
        },
      },
      2
    );

    rsiSeries.setData(
      candles
        .filter(
          (c) =>
            c.rsi14 !== null &&
            Number.isFinite(c.rsi14)
        )
        .map((c) => ({
          time: c.time as UTCTimestamp,
          value: c.rsi14 as number,
        }))
    );

    const firstTime =
      candles[0].time as UTCTimestamp;

    const lastTime =
      candles[candles.length - 1]
        .time as UTCTimestamp;

    for (const level of [30, 70]) {
      const line = chart.addSeries(
        LineSeries,
        {
          color: "#64748b",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        },
        2
      );

      line.setData([
        { time: firstTime, value: level },
        { time: lastTime, value: level },
      ]);
    }

    // --------------------------------------------------
    // DIMENSIONS DES PANNEAUX
    // --------------------------------------------------

    const panes = chart.panes();

    if (panes.length >= 3) {
      panes[1].setHeight(130);
      panes[2].setHeight(150);
    }

    chart.timeScale().setVisibleLogicalRange({
      from: Math.max(0, candles.length - 150),
      to: candles.length + 5,
    });

    // --------------------------------------------------
    // MARQUEURS PAR SOURCE
    // --------------------------------------------------

    // Les sources sont séparées pour que chaque
    // catégorie puisse être actualisée sans
    // modifier la sélection orange.

    for (const source of SOURCES) {
      markersRef.current[source] =
        createSeriesMarkers(candleSeries, []);
    }

    // --------------------------------------------------
    // SÉLECTION ORANGE PERMANENTE
    // --------------------------------------------------

    const selectionMarkers = createSeriesMarkers(
      candleSeries,
      []
    );

    let priceLine: VerticalLine | null = null;
    let volumeLine: VerticalLine | null = null;
    let rsiLine: VerticalLine | null = null;

    function clearSelection() {
      selectionMarkers.setMarkers([]);

      if (priceLine) {
        candleSeries.detachPrimitive(priceLine);
        priceLine = null;
      }

      if (volumeLine) {
        volumeSeries.detachPrimitive(volumeLine);
        volumeLine = null;
      }

      if (rsiLine) {
        rsiSeries.detachPrimitive(rsiLine);
        rsiLine = null;
      }
    }

    function updateSelection(time: number | null) {
      clearSelection();

      if (time === null) {
        return;
      }

      const candle = candles.find(
        (c) => c.time === time
      );

      if (!candle) {
        return;
      }

      const t = candle.time as UTCTimestamp;

      selectionMarkers.setMarkers([
        {
          time: t,
          position: "aboveBar",
          color: SELECTION_COLOR,
          shape: "circle",
          text: "SELECT",
          size: 1,
        },
      ]);

      priceLine = new VerticalLine(t, {
        color: SELECTION_COLOR,
        width: 2,
        showLabel: true,
        labelText: "SELECT",
        labelBackgroundColor: "#b45309",
        labelTextColor: "#ffffff",
      });

      volumeLine = new VerticalLine(t, {
        color: SELECTION_COLOR,
        width: 1,
        showLabel: false,
      });

      rsiLine = new VerticalLine(t, {
        color: SELECTION_COLOR,
        width: 1,
        showLabel: false,
      });

      candleSeries.attachPrimitive(priceLine);
      volumeSeries.attachPrimitive(volumeLine);
      rsiSeries.attachPrimitive(rsiLine);
    }

    updateSelectionRef.current = updateSelection;

    // --------------------------------------------------
    // CLIC SUR UNE BOUGIE
    // --------------------------------------------------

    const handleClick = (
      param: { time?: unknown }
    ) => {
      if (typeof param.time !== "number") {
        return;
      }

      updateSelection(param.time);
      onSelectRef.current?.(param.time);
    };

    chart.subscribeClick(handleClick);

    // --------------------------------------------------
    // NETTOYAGE
    // --------------------------------------------------

    return () => {
      chart.unsubscribeClick(handleClick);

      updateSelectionRef.current = null;

      clearSelection();

      markersRef.current = {};

      chartRef.current = null;

      chart.remove();
    };
  }, [candles]);

  // --------------------------------------------------
  // ACTUALISER LES SIGNAUX SANS RECRÉER LE CHART
  // --------------------------------------------------

  useEffect(() => {
    for (const source of SOURCES) {
      const plugin = markersRef.current[source];

      if (!plugin) {
        continue;
      }

      plugin.setMarkers(
        buildMarkers(visibleSignals, source)
      );
    }
  }, [visibleSignals]);

  // --------------------------------------------------
  // ACTUALISER LA SÉLECTION
  // --------------------------------------------------

  useEffect(() => {
    updateSelectionRef.current?.(selectedTime);
  }, [selectedTime, candles]);

  // --------------------------------------------------
  // COPIER LE GRAPHIQUE
  // --------------------------------------------------

  async function copyChartImage() {
    const chart = chartRef.current;

    if (!chart) {
      setCopyStatus("Graphique indisponible");
      return;
    }

    try {
      setCopyStatus("Copie en cours...");

      const canvas = chart.takeScreenshot(
        true,
        true
      );

      const blob = await new Promise<Blob>(
        (resolve, reject) => {
          canvas.toBlob((result) => {
            if (result) {
              resolve(result);
            } else {
              reject(
                new Error("PNG indisponible")
              );
            }
          }, "image/png");
        }
      );

      await navigator.clipboard.write([
        new ClipboardItem({
          "image/png": blob,
        }),
      ]);

      setCopyStatus("Image copiée !");
    } catch (error) {
      console.error(error);
      setCopyStatus("Échec de la copie");
    }
  }

  // --------------------------------------------------
  // INTERFACE
  // --------------------------------------------------

  return (
    <div className="w-full">
      {/* Indicateurs et bouton copie */}
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-3 text-xs">
          <span className="text-amber-400">
            ● EMA20
          </span>

          <span className="text-blue-400">
            ● EMA50
          </span>

          <span className="text-purple-400">
            ● EMA200
          </span>

          <span className="text-slate-400">
            ▮ Volume
          </span>

          <span className="text-violet-300">
            ● RSI14
          </span>
        </div>

        <div className="flex items-center gap-3">
          {copyStatus && (
            <span className="text-xs text-slate-400">
              {copyStatus}
            </span>
          )}

          <button
            type="button"
            onClick={copyChartImage}
            className="rounded-md border border-[#34517a] bg-[#173052] px-3 py-1.5 text-[11px] font-medium text-blue-100 transition hover:bg-[#23426c]"
          >
            📋 Copier le graphique
          </button>
        </div>
      </div>

      {/* Légende des sources */}
      <div className="mb-2 flex flex-wrap gap-4 text-[11px] text-slate-400">
        <span>
          <span className="text-emerald-400">▲</span>{" "}
          <span className="text-red-400">▼</span>{" "}
          Qwen V2
        </span>

        <span>
          <span className="text-emerald-300">●</span>{" "}
          <span className="text-rose-400">●</span>{" "}
          Heuristique
        </span>

        <span>
          <span className="text-blue-400">■</span>{" "}
          <span className="text-purple-400">■</span>{" "}
          Target
        </span>
      </div>

      {/* Sélection courante */}
      {selectedTime !== null && (
        <div className="mb-2 text-xs text-amber-300">
          ◆ Bougie sélectionnée :{" "}
          {new Date(selectedTime * 1000)
            .toISOString()
            .replace("T", " ")
            .slice(0, 16)}{" "}
          UTC
        </div>
      )}

      {/* Graphique TradingView */}
      <div
        ref={containerRef}
        className="h-[505px] w-full"
      />
    </div>
  );
}

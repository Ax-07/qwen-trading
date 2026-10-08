"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toBlob } from "html-to-image";
import {
  type Candle,
  type SignalSettings,
  type TradingSignal,
} from "@/components/TradingChart";
import PerformanceAnalytics from "@/components/PerformanceAnalytics";
import DisagreementExplorer from "@/components/DisagreementExplorer";
import { API } from "@/components/dashboard/constants";
import { DatasetToolbar } from "@/components/dashboard/dataset-toolbar";
import { DashboardHeader } from "@/components/dashboard/header";
import {
  AIInspector,
  CandleInspector,
} from "@/components/dashboard/inspectors";
import { MarketExplorer } from "@/components/dashboard/market-explorer";
import type {
  CandleDetail,
  Dataset,
  DatasetInfo,
  PredictionDetail,
} from "@/components/dashboard/types";

export default function Home() {
  const dashboardRef = useRef<HTMLElement>(null);
  const [dataset, setDataset] = useState<Dataset>("test");
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [signalSettings, setSignalSettings] = useState<SignalSettings>({
    qwen: true,
    heuristic: false,
    target: false,
    filter: "all",
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [signalError, setSignalError] = useState<string | null>(null);
  const [selectedTime, setSelectedTime] = useState<number | null>(null);
  const [showInspectors, setShowInspectors] = useState(true);
  const [focusedTime, setFocusedTime] = useState<number | null>(null);
  const [detail, setDetail] = useState<CandleDetail | null>(null);
  const [prediction, setPrediction] = useState<PredictionDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [captureStatus, setCaptureStatus] = useState("");
  const activeDataset = datasets.find((item) => item.name === dataset);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API}/api/datasets`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<{ datasets: DatasetInfo[] }>;
      })
      .then((data) => {
        if (!controller.signal.aborted) setDatasets(data.datasets);
      })
      .catch((fetchError) => {
        if (!controller.signal.aborted) console.error("Datasets :", fetchError);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    async function loadCandles() {
      setLoading(true);
      setError(null);
      setCandles([]);
      try {
        const params = new URLSearchParams({ dataset, limit: "500" });
        if (focusedTime !== null) {
          const span = 72 * 3600;
          params.set(
            "start",
            new Date((focusedTime - span) * 1000).toISOString(),
          );
          params.set(
            "end",
            new Date((focusedTime + span) * 1000).toISOString(),
          );
        }
        const response = await fetch(`${API}/api/candles?${params}`, {
          signal: controller.signal,
        });
        if (!response.ok)
          throw new Error(`Erreur bougies : HTTP ${response.status}`);
        const data = (await response.json()) as { candles: Candle[] };
        if (!controller.signal.aborted) setCandles(data.candles);
      } catch (loadError) {
        if (!controller.signal.aborted) setError(String(loadError));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void loadCandles();
    return () => controller.abort();
  }, [dataset, focusedTime]);

  useEffect(() => {
    const controller = new AbortController();
    setSignals([]);
    setSignalError(null);
    if (dataset === "train" || candles.length === 0)
      return () => controller.abort();
    async function loadSignals() {
      try {
        const params = new URLSearchParams({
          dataset,
          limit: "2000",
          start: new Date(candles[0].time * 1000).toISOString(),
          end: new Date(candles[candles.length - 1].time * 1000).toISOString(),
        });
        const response = await fetch(`${API}/api/signals?${params}`, {
          signal: controller.signal,
        });
        if (!response.ok)
          throw new Error(`Erreur signaux : HTTP ${response.status}`);
        const data = (await response.json()) as { signals: TradingSignal[] };
        if (!controller.signal.aborted) setSignals(data.signals);
      } catch (loadError) {
        if (!controller.signal.aborted) setSignalError(String(loadError));
      }
    }
    void loadSignals();
    return () => controller.abort();
  }, [dataset, candles]);

  useEffect(() => {
    if (candles.length && selectedTime === null && !loading)
      setSelectedTime(focusedTime ?? candles[candles.length - 1].time);
  }, [candles, selectedTime, loading, focusedTime]);

  useEffect(() => {
    const controller = new AbortController();
    if (selectedTime === null) return () => controller.abort();
    setDetail(null);
    setPrediction(null);
    setDetailError(null);
    setPredictionError(null);
    setDetailLoading(true);
    setPredictionLoading(dataset !== "train");
    async function loadDetail() {
      try {
        const response = await fetch(`${API}/api/candles/${selectedTime}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as CandleDetail;
        if (!controller.signal.aborted) setDetail(data);
      } catch (loadError) {
        if (!controller.signal.aborted) setDetailError(String(loadError));
      } finally {
        if (!controller.signal.aborted) setDetailLoading(false);
      }
    }
    async function loadPrediction() {
      if (dataset === "train") return;
      try {
        const response = await fetch(
          `${API}/api/predictions/${selectedTime}?dataset=${dataset}`,
          { signal: controller.signal },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as PredictionDetail;
        if (!controller.signal.aborted) setPrediction(data);
      } catch (loadError) {
        if (!controller.signal.aborted) setPredictionError(String(loadError));
      } finally {
        if (!controller.signal.aborted) setPredictionLoading(false);
      }
    }
    void loadDetail();
    void loadPrediction();
    return () => controller.abort();
  }, [selectedTime, dataset]);

  const changeDataset = useCallback(
    (next: Dataset) => {
      if (next === dataset) return;
      setSelectedTime(null);
      setFocusedTime(null);
      setDetail(null);
      setPrediction(null);
      setSignals([]);
      setDetailError(null);
      setPredictionError(null);
      setDetailLoading(false);
      setPredictionLoading(false);
      setDataset(next);
    },
    [dataset],
  );

  const navigateToDisagreement = useCallback((time: number) => {
    setSelectedTime(time);
    setFocusedTime(time);
    requestAnimationFrame(() =>
      document
        .getElementById("market-chart")
        ?.scrollIntoView({ behavior: "smooth", block: "start" }),
    );
  }, []);

  const clearFocusedTime = useCallback(() => {
    setFocusedTime(null);
    setSelectedTime(null);
    setDetail(null);
    setPrediction(null);
  }, []);

  async function copyDashboard() {
    if (!dashboardRef.current || capturing) return;
    setCapturing(true);
    setCaptureStatus("Capture en cours…");
    try {
      const pendingBlob = toBlob(dashboardRef.current, {
        backgroundColor: "#070d18",
        pixelRatio: 1.5,
        filter: (node) =>
          !(
            node instanceof HTMLElement &&
            node.dataset.captureExclude === "true"
          ),
      }).then((blob) => {
        if (!blob) throw new Error("PNG indisponible");
        return blob;
      });
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": pendingBlob }),
      ]);
      setCaptureStatus("Interface copiée !");
    } catch (captureError) {
      console.error(captureError);
      setCaptureStatus("Échec de la copie");
    } finally {
      setCapturing(false);
    }
  }

  return (
    <main
      ref={dashboardRef}
      className="min-h-screen bg-[#060e1e] text-slate-100"
    >
      <DashboardHeader />
      <div className="q-workspace mx-auto max-w-[1920px] space-y-3 px-3 py-3 sm:px-4 xl:px-5">
        <DatasetToolbar
          dataset={dataset}
          datasets={datasets}
          activeDataset={activeDataset}
          capturing={capturing}
          captureStatus={captureStatus}
          onDatasetChange={changeDataset}
          onRefresh={() => window.location.reload()}
          onCopy={copyDashboard}
        />
        <div
          className={`grid items-start gap-4 ${showInspectors ? "xl:grid-cols-[minmax(0,3fr)_minmax(490px,2fr)]" : "grid-cols-1"}`}
        >
          <div className="min-w-0 space-y-3">
            <MarketExplorer
              dataset={dataset}
              candles={candles}
              signals={signals}
              signalSettings={signalSettings}
              selectedTime={selectedTime}
              focusedTime={focusedTime}
              loading={loading}
              error={error}
              signalError={signalError}
              showInspectors={showInspectors}
              onSignalSettingsChange={setSignalSettings}
              onSelectCandle={setSelectedTime}
              onToggleInspectors={() => setShowInspectors((value) => !value)}
              onClearFocus={clearFocusedTime}
            />
          </div>
          <div
            className={`min-w-0 space-y-3 ${showInspectors ? "" : "hidden"}`}
          >
            <AIInspector
              dataset={dataset}
              selectedTime={selectedTime}
              detail={prediction}
              loading={predictionLoading}
              error={predictionError}
            />
            <CandleInspector
              detail={detail}
              loading={detailLoading}
              error={detailError}
            />
          </div>
        </div>
        <div id="analytics" className="scroll-mt-24">
          <PerformanceAnalytics dataset={dataset} />
        </div>
        <div id="disagreements" className="scroll-mt-24">
          <DisagreementExplorer
            dataset={dataset}
            selectedTime={selectedTime}
            onSelect={navigateToDisagreement}
          />
        </div>
        <footer className="pb-3 text-center text-[11px] text-slate-600">
          Qwen Trading Research · Graphiques par{" "}
          <a
            href="https://www.tradingview.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-slate-400"
          >
            TradingView
          </a>
        </footer>
      </div>
    </main>
  );
}

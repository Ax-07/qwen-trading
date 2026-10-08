import { useQuery } from "@tanstack/react-query";
import type { Candle, TradingSignal } from "@/components/TradingChart";
import { API } from "@/components/dashboard/constants";
import type {
  CandleDetail,
  Dataset,
  DatasetInfo,
  PredictionDetail,
} from "@/components/dashboard/types";

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export function useDatasetsQuery() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: ({ signal }) =>
      fetchJson<{ datasets: DatasetInfo[] }>(`${API}/api/datasets`, signal),
    staleTime: 5 * 60 * 1000,
  });
}

export function useCandlesQuery(dataset: Dataset, focusedTime: number | null) {
  const params = new URLSearchParams({ dataset, limit: "500" });
  if (focusedTime !== null) {
    const span = 72 * 3600;
    params.set("start", new Date((focusedTime - span) * 1000).toISOString());
    params.set("end", new Date((focusedTime + span) * 1000).toISOString());
  }
  return useQuery({
    queryKey: ["candles", dataset, focusedTime],
    queryFn: ({ signal }) =>
      fetchJson<{ candles: Candle[] }>(`${API}/api/candles?${params}`, signal),
    placeholderData: (previous) => previous,
  });
}

export function useSignalsQuery(dataset: Dataset, candles: Candle[]) {
  const params = new URLSearchParams({ dataset, limit: "2000" });
  if (candles.length) {
    params.set("start", new Date(candles[0].time * 1000).toISOString());
    params.set(
      "end",
      new Date(candles[candles.length - 1].time * 1000).toISOString(),
    );
  }
  return useQuery({
    queryKey: [
      "signals",
      dataset,
      candles[0]?.time,
      candles[candles.length - 1]?.time,
    ],
    queryFn: ({ signal }) =>
      fetchJson<{ signals: TradingSignal[] }>(
        `${API}/api/signals?${params}`,
        signal,
      ),
    enabled: dataset !== "train" && candles.length > 0,
  });
}

export function useCandleDetailQuery(selectedTime: number | null) {
  return useQuery({
    queryKey: ["candle-detail", selectedTime],
    queryFn: ({ signal }) =>
      fetchJson<CandleDetail>(`${API}/api/candles/${selectedTime}`, signal),
    enabled: selectedTime !== null,
  });
}

export function usePredictionQuery(
  dataset: Dataset,
  selectedTime: number | null,
) {
  return useQuery({
    queryKey: ["prediction", dataset, selectedTime],
    queryFn: ({ signal }) =>
      fetchJson<PredictionDetail>(
        `${API}/api/predictions/${selectedTime}?dataset=${dataset}`,
        signal,
      ),
    enabled: dataset !== "train" && selectedTime !== null,
  });
}

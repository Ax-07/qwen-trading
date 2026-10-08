import { create } from "zustand";
import type { SignalSettings } from "@/components/TradingChart";
import type { Dataset } from "@/components/dashboard/types";

type DashboardState = {
  dataset: Dataset;
  selectedTime: number | null;
  focusedTime: number | null;
  showInspectors: boolean;
  signalSettings: SignalSettings;
  setDataset: (dataset: Dataset) => void;
  setSelectedTime: (time: number | null) => void;
  setFocusedTime: (time: number | null) => void;
  setShowInspectors: (visible: boolean) => void;
  setSignalSettings: (settings: SignalSettings) => void;
  resetSelection: () => void;
};

export const useDashboardStore = create<DashboardState>((set) => ({
  dataset: "test",
  selectedTime: null,
  focusedTime: null,
  showInspectors: true,
  signalSettings: {
    qwen: true,
    heuristic: false,
    target: false,
    filter: "all",
  },
  setDataset: (dataset) => set({ dataset }),
  setSelectedTime: (selectedTime) => set({ selectedTime }),
  setFocusedTime: (focusedTime) => set({ focusedTime }),
  setShowInspectors: (showInspectors) => set({ showInspectors }),
  setSignalSettings: (signalSettings) => set({ signalSettings }),
  resetSelection: () => set({ selectedTime: null, focusedTime: null }),
}));

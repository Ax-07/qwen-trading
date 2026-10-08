import type { Dataset, DatasetInfo } from "./types";
import { formatUtc } from "./formatters";

export function DatasetToolbar({
  dataset,
  datasets,
  activeDataset,
  capturing,
  captureStatus,
  onDatasetChange,
  onRefresh,
  onCopy,
}: {
  dataset: Dataset;
  datasets: DatasetInfo[];
  activeDataset?: DatasetInfo;
  capturing: boolean;
  captureStatus: string;
  onDatasetChange: (dataset: Dataset) => void;
  onRefresh: () => void;
  onCopy: () => void;
}) {
  return (
    <div className="q-workspace-toolbar flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#24334d] bg-[#0c1729] px-4 py-3">
      <div className="flex flex-wrap items-center gap-4">
        <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
          Jeu de données
        </span>
        <div className="flex gap-1 rounded-lg border border-[#273850] bg-[#071224] p-1">
          {(["train", "validation", "test"] as Dataset[]).map((name) => (
            <button
              key={name}
              type="button"
              disabled={
                datasets.find((item) => item.name === name)?.available === false
              }
              onClick={() => onDatasetChange(name)}
              className={`rounded-md px-4 py-2 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-35 ${dataset === name ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20" : "text-slate-400 hover:bg-white/5 hover:text-white"}`}
            >
              {name === "train"
                ? "Train"
                : name === "validation"
                  ? "Validation"
                  : "Test"}
            </button>
          ))}
        </div>
        <span className="text-[11px] text-slate-400">
          {activeDataset?.start && activeDataset.end
            ? `${formatUtc(activeDataset.start)} → ${formatUtc(activeDataset.end)} · ${activeDataset.rows ?? "—"} lignes`
            : (activeDataset?.message ?? "Chargement des périodes…")}
        </span>
      </div>
      <div className="flex items-center gap-2" data-capture-exclude="true">
        <span className="hidden items-center gap-2 text-[11px] text-slate-400 xl:inline-flex">
          Comparaison :{" "}
          <strong className="inline-flex items-center rounded-lg border border-[#2b496a] bg-[#112944] px-3 py-2 text-xs font-medium text-slate-200">
            Qwen V2 / Heuristique
          </strong>
        </span>
        <button
          type="button"
          onClick={onRefresh}
          className="inline-flex items-center justify-center rounded-lg border border-[#2b496a] bg-[#112944] px-3 py-2.5 text-xs font-semibold text-slate-100 transition hover:bg-[#193957]"
          title="Recharger les données affichées"
        >
          ↻ Actualiser
        </button>
        <button
          type="button"
          onClick={onCopy}
          disabled={capturing}
          className="rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-blue-500 disabled:opacity-50"
        >
          {capturing ? "Capture…" : "▣ Copier l’interface"}
        </button>
        {captureStatus && (
          <span className="text-[11px] text-slate-400">{captureStatus}</span>
        )}
      </div>
    </div>
  );
}

import { FEATURE_GROUPS } from "./constants";
import { formatFeature, formatTimestamp, formatUtc } from "./formatters";
import type { CandleDetail, Dataset, PredictionDetail } from "./types";
import { Badge, Line, Panel } from "./ui";

export function AIInspector({
  dataset,
  selectedTime,
  detail,
  loading,
  error,
}: {
  dataset: Dataset;
  selectedTime: number | null;
  detail: PredictionDetail | null;
  loading: boolean;
  error: string | null;
}) {
  const predictions = detail?.predictions;
  const evaluation = detail?.evaluation;
  const comparison = detail?.comparison;
  return (
    <Panel
      title="✦ AI Inspector"
      subtitle="Qwen V2 et heuristique · décision historique"
      action={
        <span className="text-[11px] text-slate-400">
          {formatTimestamp(selectedTime)} UTC
        </span>
      }
    >
      {loading ? (
        <p className="p-5 text-xs text-slate-400">Chargement de l’analyse…</p>
      ) : error ? (
        <p className="text-xs text-rose-400">{error}</p>
      ) : dataset === "train" ? (
        <p className="text-xs text-slate-400">
          Aucune prédiction V2 sur Train.
        </p>
      ) : !predictions || !evaluation || !comparison ? (
        <p className="py-7 text-center text-xs text-slate-500">
          Sélectionne une bougie disponible dans l’évaluation.
        </p>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-[minmax(0,1fr)_minmax(105px,.58fr)_minmax(0,1fr)] gap-2">
            <ModelCard
              title="✦ Qwen V2"
              value={predictions.qwen}
              matches={comparison.qwen_matches_target}
              result={
                predictions.qwen === "LONG_BIAS"
                  ? evaluation.long_result
                  : predictions.qwen === "SHORT_BIAS"
                    ? evaluation.short_result
                    : "NO_TRADE"
              }
            />
            <div className="flex flex-col items-center justify-center rounded-xl border border-[#284c63] bg-gradient-to-b from-[#103049] to-[#0c2136] p-2 text-center">
              <span
                className={`text-xl ${comparison.qwen_agrees_with_heuristic ? "text-emerald-400" : "text-amber-400"}`}
              >
                {comparison.qwen_agrees_with_heuristic ? "✓" : "≠"}
              </span>
              <span className="mt-1 text-xs font-semibold">
                {comparison.qwen_agrees_with_heuristic ? "Accord" : "Désaccord"}
              </span>
              <span className="mt-1 text-[10px] text-slate-400">
                Entre les modèles
              </span>
            </div>
            <ModelCard
              title="⚙ Heuristique"
              value={predictions.heuristic}
              matches={comparison.heuristic_matches_target}
              result={
                predictions.heuristic === "LONG_BIAS"
                  ? evaluation.long_result
                  : predictions.heuristic === "SHORT_BIAS"
                    ? evaluation.short_result
                    : "NO_TRADE"
              }
            />
          </div>
          <div className="rounded-lg border border-[#284063] bg-[#11223c] px-3 py-2.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">
                Bias score · valeur issue du dataset
              </span>
              <span className="font-mono font-bold text-blue-300">
                {formatFeature(predictions.bias_score, 2)}
              </span>
            </div>
          </div>
          <div className="rounded-lg border border-amber-500/20 bg-amber-500/[.035] p-3">
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-amber-300">
              Évaluation a posteriori · horizon 12H
            </p>
            <div className="grid grid-cols-3 gap-2">
              <EvaluationBadge label="Target" value={evaluation.target} />
              <EvaluationBadge label="LONG" value={evaluation.long_result} />
              <EvaluationBadge label="SHORT" value={evaluation.short_result} />
            </div>
            <div className="mt-2 grid grid-cols-3 gap-3 text-[11px]">
              <Metric
                label="Return 12H"
                value={`${formatFeature(evaluation.future_return_12h_pct)} %`}
              />
              <Metric
                label="Up move ATR"
                value={formatFeature(evaluation.up_move_12h_atr)}
              />
              <Metric
                label="Down move ATR"
                value={formatFeature(evaluation.down_move_12h_atr)}
              />
            </div>
          </div>
          <details className="rounded-lg border border-[#283c59] px-3 py-2">
            <summary className="cursor-pointer text-xs font-semibold text-slate-300">
              Réponse brute Qwen
            </summary>
            <pre className="mt-3 max-h-52 overflow-auto whitespace-pre-wrap break-words text-[11px] text-slate-400">
              {predictions.generated}
            </pre>
          </details>
          <p className="text-[10px] text-amber-300/70">
            Target, TP et SL sont des résultats futurs, jamais des features
            d’entrée.
          </p>
        </div>
      )}
    </Panel>
  );
}

function ModelCard({
  title,
  value,
  matches,
  result,
}: {
  title: string;
  value: string;
  matches: boolean;
  result: string;
}) {
  return (
    <div className="rounded-xl border border-[#2b4669] bg-gradient-to-b from-[#152a48] to-[#0f2037] p-3">
      <p className="mb-2 text-xs font-semibold text-blue-200">{title}</p>
      <Badge value={value} />
      <div className="mt-2">
        <Line label="Correspond au target" value={matches ? "Oui" : "Non"} />
        <Line label="Résultat 12H" value={<Badge value={result} />} />
      </div>
    </div>
  );
}
function EvaluationBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-[#334055] p-2">
      <span className="mb-1 block text-[10px] text-slate-500">{label}</span>
      <Badge value={value} />
    </div>
  );
}
function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="block text-slate-500">{label}</span>
      <b>{value}</b>
    </div>
  );
}

export function CandleInspector({
  detail,
  loading,
  error,
}: {
  detail: CandleDetail | null;
  loading: boolean;
  error: string | null;
}) {
  return (
    <Panel
      title="◈ Candle Inspector"
      subtitle="Informations disponibles à la clôture"
      action={
        <span className="text-[11px] text-slate-400">
          {detail ? formatUtc(detail.timestamp) : "Données 1H"}
        </span>
      }
    >
      {loading ? (
        <p className="p-4 text-xs text-slate-400">Chargement…</p>
      ) : error ? (
        <p className="text-xs text-rose-400">{error}</p>
      ) : !detail ? (
        <p className="py-8 text-center text-xs text-slate-500">
          Sélectionne une bougie sur le graphique.
        </p>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <InfoCard
              label="Début bougie"
              value={formatUtc(detail.timestamp)}
            />
            <InfoCard
              label="Features connues à"
              value={formatUtc(detail.available_at)}
              accent
            />
          </div>
          <div className="grid gap-2 md:grid-cols-3">
            {FEATURE_GROUPS.slice(0, 3).map((group, index) => (
              <FeatureGroup
                key={group.title}
                title={group.title}
                fields={group.fields}
                detail={detail}
                digits={index === 0 ? 2 : 4}
              />
            ))}
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {FEATURE_GROUPS.slice(3).map((group) => (
              <details
                key={group.title}
                className="rounded-lg border border-[#284063] bg-[#0c1a2f] px-3 py-2"
              >
                <summary className="cursor-pointer text-xs font-medium text-slate-300">
                  {group.title}
                </summary>
                <div className="mt-2">
                  {group.fields.map((field) => (
                    <Line
                      key={field}
                      label={field}
                      value={formatFeature(detail.features[field], 5)}
                    />
                  ))}
                </div>
              </details>
            ))}
          </div>
          <p className="text-[10px] text-emerald-400">
            Features prêtes : {formatFeature(detail.features.features_ready)} ·
            Aucun outcome futur dans ce panneau
          </p>
        </div>
      )}
    </Panel>
  );
}

function InfoCard({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-[#284063] bg-[#11223c] p-2">
      <span className="text-slate-500">{label}</span>
      <p className={`mt-1 ${accent ? "text-emerald-400" : ""}`}>{value}</p>
    </div>
  );
}
function FeatureGroup({
  title,
  fields,
  detail,
  digits,
}: {
  title: string;
  fields: readonly string[];
  detail: CandleDetail;
  digits: number;
}) {
  return (
    <div className="rounded-xl border border-[#2a4363] bg-gradient-to-b from-[#122741] to-[#0e1d34] p-3">
      <h3 className="mb-2 border-b border-white/10 pb-2 text-[11px] font-semibold text-blue-200">
        {title}
      </h3>
      {fields.map((field) => (
        <Line
          key={field}
          label={field}
          value={formatFeature(detail.features[field], digits)}
        />
      ))}
    </div>
  );
}

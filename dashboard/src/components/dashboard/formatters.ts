import type { FeatureValue } from "./types";

export function formatFeature(
  value: FeatureValue | undefined,
  digits = 4,
): string {
  if (value == null) return "—";
  if (typeof value === "boolean") return value ? "Oui" : "Non";
  if (!Number.isFinite(value)) return "—";
  return value.toLocaleString("fr-FR", { maximumFractionDigits: digits });
}

export function formatUtc(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "—"
    : `${date.toISOString().replace("T", " ").slice(0, 16)} UTC`;
}

export function formatTimestamp(time: number | null): string {
  return time
    ? new Date(time * 1000).toISOString().slice(0, 16).replace("T", " ")
    : "—";
}

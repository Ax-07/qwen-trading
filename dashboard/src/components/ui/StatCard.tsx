import type { ReactNode } from "react";

type Tone = "neutral" | "positive" | "negative" | "warning" | "info";

type Props = {
  label: string;
  value: ReactNode;
  description?: ReactNode;
  icon?: ReactNode;
  tone?: Tone;
  className?: string;
};

const toneClass: Record<Tone, string> = {
  neutral: "text-slate-100",
  positive: "text-emerald-400",
  negative: "text-rose-400",
  warning: "text-amber-400",
  info: "text-blue-400",
};

export default function StatCard({ label, value, description, icon, tone = "neutral", className = "" }: Props) {
  return (
    <div className={`min-w-0 rounded-xl border border-[#20324c] bg-[#112039] p-3.5 shadow-sm lg:p-4 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium text-slate-400">{label}</p>
        {icon && <span className="text-slate-400" aria-hidden="true">{icon}</span>}
      </div>
      <div className={`q-data mt-2 text-xl font-semibold tracking-tight lg:text-2xl ${toneClass[tone]}`}>{value}</div>
      {description && <div className="mt-1.5 text-xs text-slate-400">{description}</div>}
    </div>
  );
}

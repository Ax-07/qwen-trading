import type { ReactNode } from "react";

type Tone = "neutral" | "positive" | "negative" | "warning" | "info";

type Props = {
  children: ReactNode;
  tone?: Tone;
  icon?: ReactNode;
  className?: string;
};

const variants: Record<Tone, string> = {
  neutral: "border-slate-600/60 bg-slate-700/20 text-slate-300",
  positive: "border-emerald-700/70 bg-emerald-500/10 text-emerald-300",
  negative: "border-rose-700/70 bg-rose-500/10 text-rose-300",
  warning: "border-amber-700/70 bg-amber-500/10 text-amber-300",
  info: "border-blue-700/70 bg-blue-500/10 text-blue-300",
};

export default function Pill({ children, tone = "neutral", icon, className = "" }: Props) {
  return (
    <span className={`inline-flex max-w-full items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-semibold leading-4 ${variants[tone]} ${className}`}>
      {icon && <span aria-hidden="true">{icon}</span>}
      {children}
    </span>
  );
}

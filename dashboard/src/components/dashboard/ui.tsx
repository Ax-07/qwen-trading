import type { ReactNode } from "react";
import { Badge as ShadcnBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function Badge({ value }: { value: string | undefined }) {
  const theme =
    value === "LONG_BIAS" || value === "TP"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
      : value === "SHORT_BIAS" || value === "SL"
        ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
        : "border-slate-600/60 bg-slate-700/30 text-slate-300";
  return (
    <ShadcnBadge
      variant="outline"
      className={`rounded-md px-2 py-1 text-[11px] font-semibold ${theme}`}
    >
      {value ?? "—"}
    </ShadcnBadge>
  );
}

export function Line({
  label,
  value,
  valueClass = "",
}: {
  label: string;
  value: ReactNode;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/[0.06] py-1.5 text-[11px] last:border-0">
      <span className="text-slate-400">{label}</span>
      <span
        className={`text-right font-medium tabular-nums text-slate-100 ${valueClass}`}
      >
        {value}
      </span>
    </div>
  );
}

export function Panel({
  title,
  subtitle,
  children,
  action,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <Card className="rounded-xl border-[#24334d] bg-[#0c1729] text-slate-100 shadow-[0_8px_30px_rgba(0,0,0,.13)]">
      <CardHeader className="flex flex-row items-start justify-between gap-3 border-b border-[#24334d] px-4 py-4 space-y-0">
        <div>
          <CardTitle className="text-sm font-semibold tracking-tight text-slate-100">
            {title}
          </CardTitle>
          <CardDescription className="mt-1 text-[11px] text-slate-500">
            {subtitle}
          </CardDescription>
        </div>
        {action}
      </CardHeader>
      <CardContent className="p-4">{children}</CardContent>
    </Card>
  );
}

export function FilterButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <Button
      type="button"
      size="sm"
      variant={active ? "default" : "outline"}
      onClick={onClick}
      className={
        active
          ? "border-blue-500 bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500"
          : "border-[#30415d] bg-[#111e33] px-3 py-1.5 text-xs text-slate-400 hover:border-blue-400/50 hover:bg-[#162741] hover:text-white"
      }
    >
      {children}
    </Button>
  );
}

"use client";

type Option<T extends string> = { value: T; label: string; disabled?: boolean };

type Props<T extends string> = {
  options: readonly Option<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
  className?: string;
};

export default function SegmentedTabs<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  className = "",
}: Props<T>) {
  return (
    <div role="group" aria-label={ariaLabel} className={`inline-flex max-w-full flex-wrap items-center gap-1 rounded-lg border border-[#263953] bg-[#091425] p-1 ${className}`}>
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={selected}
            disabled={option.disabled}
            onClick={() => onChange(option.value)}
            className={`rounded-md px-3.5 py-2 text-xs font-medium transition-colors sm:px-4 ${
              selected
                ? "bg-blue-600 text-white shadow-[0_2px_10px_#1e63f144]"
                : "text-slate-400 hover:bg-[#182a43] hover:text-slate-100"
            } disabled:cursor-not-allowed disabled:opacity-35`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

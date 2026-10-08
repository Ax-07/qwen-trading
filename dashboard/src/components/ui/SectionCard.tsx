import type { HTMLAttributes, ReactNode } from "react";

type Props = HTMLAttributes<HTMLElement> & {
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  bodyClassName?: string;
};

export default function SectionCard({
  title,
  subtitle,
  icon,
  action,
  children,
  className = "",
  bodyClassName = "",
  ...props
}: Props) {
  return (
    <section className={`q-panel min-w-0 overflow-hidden ${className}`} {...props}>
      {(title || subtitle || action || icon) && (
        <div className="q-panel-header">
          <div className="flex min-w-0 items-center gap-3">
            {icon && <div className="shrink-0 text-blue-400" aria-hidden="true">{icon}</div>}
            <div className="min-w-0">
              {title && <h2 className="text-base font-semibold tracking-tight text-slate-100 lg:text-lg">{title}</h2>}
              {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
            </div>
          </div>
          {action && <div className="flex shrink-0 items-center gap-2">{action}</div>}
        </div>
      )}
      <div className={`p-4 lg:p-[18px] ${bodyClassName}`}>{children}</div>
    </section>
  );
}

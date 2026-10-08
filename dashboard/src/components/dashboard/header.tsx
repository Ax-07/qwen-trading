export function DashboardHeader() {
  return (
    <header className="sticky top-0 z-30 border-b border-[#1c2d47] bg-[#071226]/95 px-4 backdrop-blur-xl xl:px-6">
      <div className="mx-auto flex min-h-[62px] max-w-[1920px] items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="q-brandmark grid h-10 w-10 place-items-center rounded-xl border border-blue-400/30 bg-blue-500/15 text-xl text-blue-300">
            ◈
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight">Qwen Trading</h1>
            <p className="text-[11px] text-slate-500">
              AI Market Research Dashboard
            </p>
          </div>
        </div>
        <nav
          className="hidden items-center gap-1 lg:flex"
          aria-label="Navigation principale"
        >
          <a
            href="#market-chart"
            className="inline-flex items-center gap-2 rounded-lg border-b-2 border-blue-400 bg-blue-600/25 px-3 py-2.5 text-xs font-semibold text-blue-100"
          >
            ◈ &nbsp;Tableau de bord
          </a>
          <a
            href="#disagreements"
            className="inline-flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium text-slate-300 transition hover:bg-blue-500/10 hover:text-white"
          >
            ⌕ &nbsp;Exploration
          </a>
          <a
            href="#analytics"
            className="inline-flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium text-slate-300 transition hover:bg-blue-500/10 hover:text-white"
          >
            ◈ &nbsp;Analyses
          </a>
          <span
            className="inline-flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium text-slate-500"
            title="À venir"
          >
            ⌁ &nbsp;Stratégies
          </span>
          <span
            className="inline-flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium text-slate-500"
            title="À venir"
          >
            ◤ &nbsp;Données
          </span>
          <span
            className="inline-flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium text-slate-500"
            title="À venir"
          >
            ⚙ &nbsp;Paramètres
          </span>
        </nav>
        <div className="flex items-center gap-2 lg:gap-3">
          <span className="hidden items-center gap-2 text-[11px] text-emerald-400 xl:inline-flex">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Données historiques
          </span>
          <span className="inline-flex items-center whitespace-nowrap rounded-lg border border-[#2b496a] bg-[#112944] px-3 py-2 text-xs text-slate-200">
            ₿ &nbsp;BTC/USDC <span className="text-slate-500">· 1H</span>
          </span>
          <span className="q-header-chip hidden 2xl:inline-flex">FR</span>
        </div>
      </div>
    </header>
  );
}

// Small, reusable building blocks so every page looks consistent.
import type { ReactNode } from "react";
import type { Verdict } from "../api";
import { useChart } from "../theme";

export const VERDICT_STYLE: Record<Verdict, { label: string; hex: string; badge: string }> = {
  BENIGN: { label: "Benign", hex: "#34d399", badge: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30" },
  KNOWN_ATTACK: { label: "Known attack", hex: "#fbbf24", badge: "bg-amber-500/15 text-amber-300 ring-amber-500/30" },
  UNKNOWN_ANOMALY: { label: "Unknown anomaly", hex: "#e879f9", badge: "bg-fuchsia-500/15 text-fuchsia-300 ring-fuchsia-500/30" },
};
export const VERDICTS: Verdict[] = ["BENIGN", "KNOWN_ATTACK", "UNKNOWN_ANOMALY"];

export function Card(props: { title?: string; subtitle?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg shadow-black/20 ${props.className ?? ""}`}>
      {(props.title || props.right) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {props.title && <h2 className="text-sm font-semibold tracking-wide text-slate-100">{props.title}</h2>}
            {props.subtitle && <p className="mt-0.5 text-xs text-slate-400">{props.subtitle}</p>}
          </div>
          {props.right}
        </header>
      )}
      {props.children}
    </section>
  );
}

export function Stat(props: { label: string; value: ReactNode; hint?: string; accent?: string }) {
  const { tone } = useChart();
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">{props.label}</div>
      <div className="mt-1 font-mono text-2xl font-semibold" style={{ color: tone(props.accent ?? "#e2e8f0") }}>
        {props.value}
      </div>
      {props.hint && <div className="mt-0.5 text-xs text-slate-500">{props.hint}</div>}
    </div>
  );
}

export function Segmented<T extends string>(props: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; hint?: string }[];
}) {
  return (
    <div className="inline-flex rounded-xl border border-slate-800 bg-slate-950 p-1">
      {props.options.map((o) => (
        <button
          key={o.value}
          title={o.hint}
          onClick={() => props.onChange(o.value)}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
            props.value === o.value ? "bg-sky-500/20 text-sky-200 ring-1 ring-sky-500/40" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function VerdictBadge({ v }: { v: Verdict }) {
  const s = VERDICT_STYLE[v];
  return <span className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ${s.badge}`}>{s.label}</span>;
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />;
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{message}</div>;
}

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
export const num = (x: number) => x.toLocaleString("en-GB");

// Evidence: the evaluation results, read from reports/tables via /evidence.
import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type Evidence as Ev, type ModelId, type TableRow } from "../api";
import { Card, ErrorBox, Segmented, Spinner, Stat, num, pct } from "../components/ui";
import { fileLabel } from "./Replay";
import { useChart } from "../theme";

const METHODS = ["ae_only", "lgbm_only", "cascade", "parallel"] as const;
type Method = (typeof METHODS)[number];
const METHOD: Record<Method, { label: string; hex: string }> = {
  ae_only: { label: "AE only", hex: "#a78bfa" },
  lgbm_only: { label: "LightGBM only", hex: "#60a5fa" },
  cascade: { label: "Cascade", hex: "#64748b" },
  parallel: { label: "Parallel (DualGuard)", hex: "#f472b6" },
};

const n = (v: TableRow[string] | undefined) => (typeof v === "number" ? v : Number(v ?? NaN));
const find = (rows: TableRow[] | undefined, name: string) => rows?.find((r) => r.name === name);

export default function Evidence() {
  const [model, setModel] = useState<ModelId>("B");
  const [ev, setEv] = useState<Ev | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setEv(null); setError(null);
    api.evidence(model).then(setEv).catch((e) => setError((e as Error).message));
  }, [model]);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Evidence</h1>
          <p className="mt-1 text-sm text-slate-400">
            Test-set results at thresholds fixed on validation (2% benign false-alarm budget). Test data was opened once.
          </p>
        </div>
        <Segmented<ModelId> value={model} onChange={setModel}
          options={[{ value: "A", label: "Known (A)" }, { value: "B", label: "Zero-Day (B)" }]} />
      </header>
      {error && <ErrorBox message={error} />}
      {!ev && !error && <div className="flex items-center gap-3 text-sm text-slate-400"><Spinner /> Loading…</div>}
      {ev && <Body ev={ev} model={model} />}
    </div>
  );
}

function Body({ ev, model }: { ev: Ev; model: ModelId }) {
  const c = useChart();
  const sum = ev.test_summary;
  const P = find(sum, "parallel"), L = find(sum, "lgbm_only");
  const det = (ev.test_detection ?? []).filter((r) => r.name !== "BENIGN" && r.name !== "MACRO (attacks)");
  const macroTest = find(ev.test_detection, "MACRO (attacks)");
  const macroVal = find(ev.val_detection, "MACRO (attacks)");
  const fprTest = find(ev.test_detection, "BENIGN");
  const fprVal = find(ev.val_detection, "BENIGN");

  return (
    <div className="space-y-6">
      {P && L && (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <Stat label="DualGuard macro detection" value={pct(n(P.macro_detection))} accent={METHOD.parallel.hex} />
            <Stat label="LightGBM-only macro" value={pct(n(L.macro_detection))} accent={METHOD.lgbm_only.hex} />
            <Stat label="DualGuard false alarms" value={pct(n(P.benign_false_alarm_rate), 2)} hint="share of benign test flows" />
            <Stat label="LightGBM-only false alarms" value={pct(n(L.benign_false_alarm_rate), 2)} />
            <Stat label="DualGuard alert precision" value={pct(n(P.alert_precision))} hint={`${num(n(P.alerts))} alerts`} />
          </div>
          <Card>
            <p className="text-sm leading-relaxed text-slate-300">
              {model === "B"
                ? "Every attack family in this test set (Thursday–Friday) was absent from training. "
                : "All attack families in this test set were seen in training. "}
              DualGuard (parallel) detects <b className="text-white">{pct(n(P.macro_detection))}</b> of attack families on average,
              against <b className="text-white">{pct(n(L.macro_detection))}</b> for LightGBM alone, with false-alarm rates of{" "}
              <b className="text-white">{pct(n(P.benign_false_alarm_rate), 2)}</b> and <b className="text-white">{pct(n(L.benign_false_alarm_rate), 2)}</b>.
              {n(P.benign_false_alarm_rate) > 0.03 &&
                " The false-alarm rate is above the 2% validation budget: Thursday–Friday benign traffic differs from Monday–Wednesday (see the per-capture breakdown below)."}
            </p>
          </Card>
        </>
      )}

      <Card title="Detection rate per attack family" subtitle="Share of each family's test flows that raised an alarm.">
        <div className="h-80">
          <ResponsiveContainer>
            <BarChart data={det}>
              <CartesianGrid stroke={c.grid} vertical={false} />
              <XAxis dataKey="name" tick={c.axis} />
              <YAxis tick={c.axis} domain={[0, 1]} tickFormatter={(v: number) => pct(v, 0)} />
              <Tooltip {...c.tooltip} formatter={(v) => pct(Number(v))} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {METHODS.map((m) => <Bar key={m} dataKey={m} name={METHOD[m].label} fill={METHOD[m].hex} radius={[3, 3, 0, 0]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="All methods (test)" subtitle="Same false-alarm budget for every method.">
          <table className="w-full text-xs">
            <thead className="text-[11px] uppercase tracking-wider text-slate-500">
              <tr className="border-b border-slate-800">
                <th className="py-2 text-left">Method</th><th className="text-right">False alarms</th>
                <th className="text-right">Macro det.</th><th className="text-right">Overall det.</th>
                <th className="text-right">Alerts</th><th className="text-right">Precision</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {(sum ?? []).map((r) => (
                <tr key={String(r.name)} className="border-b border-slate-800/60">
                  <td className="py-2 font-sans" style={{ color: c.tone(METHOD[r.name as Method]?.hex ?? "#e2e8f0") }}>{METHOD[r.name as Method]?.label ?? r.name}</td>
                  <td className="text-right">{pct(n(r.benign_false_alarm_rate), 2)}</td>
                  <td className="text-right">{pct(n(r.macro_detection))}</td>
                  <td className="text-right">{pct(n(r.overall_detection))}</td>
                  <td className="text-right">{num(n(r.alerts))}</td>
                  <td className="text-right">{pct(n(r.alert_precision))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Validation vs test" subtitle="Did the validation picture hold on unseen test data?">
          <table className="w-full text-xs">
            <thead className="text-[11px] uppercase tracking-wider text-slate-500">
              <tr className="border-b border-slate-800">
                <th className="py-2 text-left">Method</th>
                <th className="text-right">Macro val</th><th className="text-right">Macro test</th>
                <th className="text-right">False alarms val</th><th className="text-right">False alarms test</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {METHODS.map((m) => (
                <tr key={m} className="border-b border-slate-800/60">
                  <td className="py-2 font-sans" style={{ color: c.tone(METHOD[m].hex) }}>{METHOD[m].label}</td>
                  <td className="text-right">{macroVal ? pct(n(macroVal[m])) : "–"}</td>
                  <td className="text-right">{macroTest ? pct(n(macroTest[m])) : "–"}</td>
                  <td className="text-right">{fprVal ? pct(n(fprVal[m]), 2) : "–"}</td>
                  <td className="text-right">{fprTest ? pct(n(fprTest[m]), 2) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {model === "B" && (
            <p className="mt-3 text-xs text-slate-500">
              Scenario B validation contains only the families LightGBM was trained on, so its validation scores say nothing about unseen attacks.
            </p>
          )}
        </Card>
      </div>

      {ev.test_naming && <Naming rows={ev.test_naming} />}
      {model === "B" && ev.posthoc_by_file && <PerCapture rows={ev.posthoc_by_file} />}

      <div className="grid gap-6 lg:grid-cols-2">
        {ev.ae_history && (
          <Card title="Autoencoder training" subtitle="Mean squared reconstruction error on benign flows; early stopping on validation.">
            <div className="h-64">
              <ResponsiveContainer>
                <LineChart data={ev.ae_history}>
                  <CartesianGrid stroke={c.grid} />
                  <XAxis dataKey="epoch" tick={c.axis} />
                  <YAxis scale="log" domain={["auto", "auto"]} tick={c.axis} tickFormatter={(v: number) => v.toFixed(3)} />
                  <Tooltip {...c.tooltip} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line dataKey="train_loss" name="train" stroke="#38bdf8" dot={false} strokeWidth={2} />
                  <Line dataKey="val_loss" name="validation" stroke="#f472b6" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
        {ev.lgbm_importance && (
          <Card title="What LightGBM relies on" subtitle="Top 15 features by share of total split gain.">
            <div className="h-64">
              <ResponsiveContainer>
                <BarChart data={ev.lgbm_importance} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" tick={c.axis} tickFormatter={(v: number) => pct(v, 0)} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: c.text, fontSize: 10 }} width={150} />
                  <Tooltip {...c.tooltip} formatter={(v) => pct(Number(v))} />
                  <Bar dataKey="share" name="share of gain" fill="#60a5fa" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

/** What the parallel hybrid answered for each true family: rows = truth, cells shaded by row share. */
function Naming({ rows }: { rows: TableRow[] }) {
  const colours = useChart();   // not 'c': the column loop below already uses c
  const cols = Object.keys(rows[0] ?? {}).filter((k) => k.startsWith("parallel|"));
  return (
    <Card title="What DualGuard called each family (parallel, test)"
          subtitle="Rows = true family. '?' = UNKNOWN_ANOMALY. In Scenario B no unseen family can be named correctly.">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-[11px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-slate-800">
              <th className="py-2 text-left">True ↓ / output →</th>
              {cols.map((c) => <th key={c} className="px-2 text-right">{c.split("|")[1]}</th>)}
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => {
              const total = cols.reduce((s, c) => s + n(r[c]), 0) || 1;
              return (
                <tr key={String(r.true)} className="border-b border-slate-800/60">
                  <td className="py-1.5 font-sans text-slate-300">{r.true}</td>
                  {cols.map((c) => {
                    const share = n(r[c]) / total;
                    return (
                      <td key={c} className="px-2 text-right"
                          style={{ background: `rgba(56,189,248,${Math.min(share, 1) * 0.45})`, color: share > 0 ? colours.strong : colours.faint }}>
                        {num(n(r[c]))}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** Post-hoc: AE false-alarm and detection rates per Thu–Fri capture file (analysis only, nothing re-tuned). */
function PerCapture({ rows }: { rows: TableRow[] }) {
  const c = useChart();
  const data = rows.map((r) => ({ ...r, file: fileLabel(String(r.SourceFile)) }));
  return (
    <Card title="Where the false alarms come from (post-hoc)"
          subtitle="AE alarm rate per Thursday–Friday capture. Dashed line = 2% validation budget. Analysis only; nothing re-tuned.">
      <div className="h-72">
        <ResponsiveContainer>
          <BarChart data={data}>
            <CartesianGrid stroke={c.grid} vertical={false} />
            <XAxis dataKey="file" tick={{ fill: c.tick, fontSize: 10 }} interval={0} />
            <YAxis tick={c.axis} domain={[0, 1]} tickFormatter={(v: number) => pct(v, 0)} />
            <Tooltip {...c.tooltip} formatter={(v) => pct(Number(v), 2)} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <ReferenceLine y={0.02} stroke="#fbbf24" strokeDasharray="4 4" />
            <Bar dataKey="benign_false_alarm_rate" name="benign false-alarm rate" fill="#fb7185" radius={[3, 3, 0, 0]} />
            <Bar dataKey="attack_detection_rate" name="attack detection rate" fill="#38bdf8" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

// Analyse: upload a CSV of flows (or pick a demo sample), get a verdict for every flow,
// and see WHY each verdict was given.
import { FileUp, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type Mode, type ModelId, type PredictResponse, type ResultRow, type Verdict } from "../api";
import { Card, ErrorBox, Segmented, Spinner, Stat, VERDICTS, VERDICT_STYLE, VerdictBadge, num, pct } from "../components/ui";
import { useChart } from "../theme";


/** Plain-English reason for a verdict, built only from the numbers the API returned. */
export function explain(r: ResultRow, mode: Mode): string {
  const ratio = `${r.ae_ratio.toFixed(2)}×`;
  const lg = `${r.lgbm_family} (${pct(r.lgbm_confidence)})`;
  if (r.verdict === "KNOWN_ATTACK")
    return `LightGBM recognises a trained attack family: ${lg}.` +
      (r.ae_ratio > 1 ? ` The autoencoder agrees it is abnormal (error ${ratio} its threshold).` : " The autoencoder alone would have let it through.");
  if (r.verdict === "UNKNOWN_ANOMALY")
    return `The autoencoder cannot reconstruct this flow (error ${ratio} its threshold), but LightGBM cannot name an attack (top guess ${lg}). Treated as a possible new attack.`;
  if (mode === "cascade")
    return `The autoencoder gate finds it normal (error ${ratio} its threshold), so LightGBM was not consulted.`;
  return `The autoencoder finds it normal (error ${ratio} its threshold) and LightGBM does not name an attack (top guess ${lg}).`;
}

export default function Analyse() {
  const [model, setModel] = useState<ModelId>("B");
  const [mode, setMode] = useState<Mode>("parallel");
  const [samples, setSamples] = useState<string[]>([]);
  const [file, setFile] = useState<{ blob: Blob; name: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<PredictResponse | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { api.samples().then(setSamples).catch(() => setSamples([])); }, []);

  async function run(blob: Blob, name: string, m = model, md = mode) {
    setBusy(true); setError(null);
    try { setRes(await api.predictCsv(blob, name, m, md)); }
    catch (e) { setError((e as Error).message); setRes(null); }
    finally { setBusy(false); }
  }

  async function loadSample(name: string) {
    try {
      const blob = await api.sampleFile(name);
      setFile({ blob, name });
      await run(blob, name);
    } catch (e) { setError((e as Error).message); }
  }

  function pick(f: File | undefined) {
    if (!f) return;
    setFile({ blob: f, name: f.name });
    run(f, f.name);
  }

  // Re-score the same file when the model or mode changes.
  function change(m: ModelId, md: Mode) {
    setModel(m); setMode(md);
    if (file) run(file.blob, file.name, m, md);
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-white">Analyse network flows</h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload CICFlowMeter flows as CSV. Every flow is scored live by both stages: the autoencoder (does it look normal?)
          and LightGBM (is it a known attack?).
        </p>
      </header>

      <Card>
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-slate-500">Model</div>
            <Segmented<ModelId>
              value={model}
              onChange={(m) => change(m, mode)}
              options={[
                { value: "A", label: "Known", hint: "Trained on all five days: knows all 8 attack families" },
                { value: "B", label: "Zero-Day", hint: "Trained Mon–Wed only: Thu–Fri attack types are new to it" },
              ]}
            />
          </div>
          <div>
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-slate-500">Decision logic</div>
            <Segmented<Mode>
              value={mode}
              onChange={(md) => change(model, md)}
              options={[
                { value: "parallel", label: "Parallel (default)", hint: "Both stages see every flow" },
                { value: "cascade", label: "Cascade", hint: "Autoencoder first; only abnormal flows reach LightGBM" },
              ]}
            />
          </div>
          {file && (
            <div className="ml-auto text-right text-xs text-slate-400">
              <div className="text-slate-500">current file</div>
              <div className="font-mono text-slate-200">{file.name}</div>
            </div>
          )}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Upload CSV" subtitle="Raw CIC-IDS2017 format. A Label column is optional (used only for comparison)." className="lg:col-span-2">
          <button
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]); }}
            className={`flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed py-10 transition ${
              dragging ? "border-sky-400 bg-sky-500/10" : "border-slate-700 hover:border-slate-500"
            }`}
          >
            <FileUp className="h-8 w-8 text-slate-500" />
            <span className="text-sm text-slate-300">Drop a CSV here or click to choose</span>
            <span className="text-xs text-slate-500">up to 200,000 flows</span>
          </button>
          <input ref={inputRef} type="file" accept=".csv" hidden onChange={(e) => pick(e.target.files?.[0])} />
        </Card>

        <Card title="Demo samples" subtitle="300 flows each, from the evaluated test sets.">
          <div className="grid grid-cols-1 gap-1.5">
            {samples.length === 0 && <p className="text-xs text-slate-500">No samples (run scripts/07_make_samples.py).</p>}
            {samples.map((s) => (
              <button
                key={s}
                onClick={() => loadSample(s)}
                className="flex items-center justify-between rounded-lg border border-slate-800 px-3 py-1.5 text-left text-xs text-slate-300 hover:border-sky-500/40 hover:bg-sky-500/5"
              >
                <span className="font-mono">{s.replace(".csv", "")}</span>
                <Play className="h-3 w-3 text-slate-500" />
              </button>
            ))}
          </div>
        </Card>
      </div>

      {busy && <div className="flex items-center gap-3 text-sm text-slate-400"><Spinner /> Scoring flows…</div>}
      {error && <ErrorBox message={error} />}
      {res && !busy && <Results res={res} />}
    </div>
  );
}

function Results({ res }: { res: PredictResponse }) {
  const c = useChart();
  const rows = res.results;
  const labelled = rows.length > 0 && rows[0].true_family !== undefined;

  // Agreement with the true labels, if the file had them (computed on the returned rows).
  const truth = useMemo(() => {
    if (!labelled) return null;
    const att = rows.filter((r) => r.true_family !== "BENIGN");
    const ben = rows.filter((r) => r.true_family === "BENIGN");
    return {
      attacks: att.length,
      detected: att.filter((r) => r.verdict !== "BENIGN").length,
      benign: ben.length,
      falseAlarms: ben.filter((r) => r.verdict !== "BENIGN").length,
    };
  }, [rows, labelled]);

  const pie = VERDICTS.map((v) => ({ name: VERDICT_STYLE[v].label, value: res.verdict_counts[v] ?? 0, v }))
    .filter((d) => d.value > 0);

  const byFamily = useMemo(() => {
    if (!labelled) return [];
    const m = new Map<string, Record<string, number | string>>();
    for (const r of rows) {
      const k = r.true_family ?? "?";
      const e = m.get(k) ?? { family: k, BENIGN: 0, KNOWN_ATTACK: 0, UNKNOWN_ANOMALY: 0 };
      e[r.verdict] = (e[r.verdict] as number) + 1;
      m.set(k, e);
    }
    return [...m.values()];
  }, [rows, labelled]);

  const scatter = useMemo(
    () => VERDICTS.map((v) => ({
      v,
      points: rows.filter((r) => r.verdict === v).slice(0, 1500).map((r) => ({
        x: 1 - r.p_benign, y: Math.max(r.ae_ratio, 0.01), row: r.input_row,
      })),
    })),
    [rows],
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Flows scored" value={num(res.n_scored)} hint={res.n_invalid ? `${num(res.n_invalid)} invalid skipped` : "all rows valid"} />
        {VERDICTS.map((v) => (
          <Stat key={v} label={VERDICT_STYLE[v].label} value={num(res.verdict_counts[v] ?? 0)}
                hint={pct((res.verdict_counts[v] ?? 0) / Math.max(res.n_scored, 1))} accent={VERDICT_STYLE[v].hex} />
        ))}
        <Stat label="Scoring time" value={`${res.elapsed_ms.toFixed(0)} ms`}
              hint={`${(res.elapsed_ms / Math.max(res.n_scored, 1)).toFixed(3)} ms / flow`} />
        {truth && (
          <Stat label="vs. true labels"
                value={truth.attacks ? pct(truth.detected / truth.attacks) : "–"}
                hint={`attacks detected · false alarms ${truth.benign ? pct(truth.falseAlarms / truth.benign) : "–"}`} />
        )}
      </div>
      {res.truncated && (
        <p className="text-xs text-amber-300/80">
          Showing the first {num(rows.length)} rows in charts and table; counts above cover all {num(res.n_scored)} flows.
        </p>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Verdicts" subtitle={`Model ${res.model} · ${res.mode}`}>
          <div className="h-56">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pie} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2} stroke="none">
                  {pie.map((d) => <Cell key={d.v} fill={VERDICT_STYLE[d.v].hex} />)}
                </Pie>
                <Tooltip {...c.tooltip} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Two-stage map" className="lg:col-span-2"
              subtitle="Each dot is a flow. Up = the autoencoder finds it abnormal; right = LightGBM suspects an attack.">
          <div className="h-56">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid stroke={c.grid} />
                <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fill: c.tick, fontSize: 11 }}
                       label={{ value: "LightGBM: P(attack)", position: "insideBottom", offset: -10, fill: c.muted, fontSize: 11 }} />
                <YAxis type="number" dataKey="y" scale="log" domain={[0.01, "auto"]} allowDataOverflow
                       tick={{ fill: c.tick, fontSize: 11 }}
                       label={{ value: "AE error ÷ threshold", angle: -90, position: "insideLeft", fill: c.muted, fontSize: 11 }} />
                <ReferenceLine y={1} stroke="#e879f9" strokeDasharray="4 4"
                               label={{ value: "AE threshold", fill: "#e879f9", fontSize: 10, position: "insideTopLeft" }} />
                <ReferenceLine x={0.5} stroke="#fbbf24" strokeDasharray="4 4"
                               label={{ value: "τ = 0.5", fill: "#fbbf24", fontSize: 10, position: "insideTopRight" }} />
                <Tooltip {...c.tooltip} cursor={{ strokeDasharray: "3 3" }} />
                {scatter.map((s) => (
                  <Scatter key={s.v} name={VERDICT_STYLE[s.v].label} data={s.points} fill={VERDICT_STYLE[s.v].hex} fillOpacity={0.6} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {labelled && (
        <Card title="Verdicts by true family" subtitle="From the file's Label column (never used for the decision).">
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={byFamily} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid stroke={c.grid} horizontal={false} />
                <XAxis type="number" tick={{ fill: c.tick, fontSize: 11 }} />
                <YAxis type="category" dataKey="family" tick={{ fill: c.text, fontSize: 12 }} width={90} />
                <Tooltip {...c.tooltip} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {VERDICTS.map((v) => (
                  <Bar key={v} dataKey={v} name={VERDICT_STYLE[v].label} stackId="s" fill={VERDICT_STYLE[v].hex} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      <FlowTable rows={rows} mode={res.mode} labelled={labelled} />
    </div>
  );
}

function FlowTable({ rows, mode, labelled }: { rows: ResultRow[]; mode: Mode; labelled: boolean }) {
  const [filter, setFilter] = useState<Verdict | "ALL">("ALL");
  const [limit, setLimit] = useState(50);
  const [open, setOpen] = useState<number | null>(null);
  const shown = rows.filter((r) => filter === "ALL" || r.verdict === filter);

  return (
    <Card title="Flows" subtitle="Click a row to see why it got its verdict."
          right={
            <Segmented<Verdict | "ALL">
              value={filter}
              onChange={(f) => { setFilter(f); setLimit(50); }}
              options={[{ value: "ALL", label: "All" }, ...VERDICTS.map((v) => ({ value: v, label: VERDICT_STYLE[v].label }))]}
            />
          }>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-800 text-[11px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="py-2 pr-3">#</th>
              <th className="pr-3">Verdict</th>
              <th className="pr-3">Family</th>
              {labelled && <th className="pr-3">True label</th>}
              <th className="pr-3">AE error ÷ threshold</th>
              <th className="pr-3">LightGBM top guess</th>
            </tr>
          </thead>
          <tbody>
            {shown.slice(0, limit).map((r) => (
              <FlowRow key={r.input_row} r={r} mode={mode} labelled={labelled}
                       open={open === r.input_row} toggle={() => setOpen(open === r.input_row ? null : r.input_row)} />
            ))}
          </tbody>
        </table>
      </div>
      {shown.length > limit && (
        <button onClick={() => setLimit(limit + 100)} className="mt-3 text-xs text-sky-300 hover:text-sky-200">
          Show more ({num(shown.length - limit)} remaining)
        </button>
      )}
      {shown.length === 0 && <p className="py-4 text-xs text-slate-500">No flows with this verdict.</p>}
    </Card>
  );
}

function FlowRow({ r, mode, labelled, open, toggle }:
  { r: ResultRow; mode: Mode; labelled: boolean; open: boolean; toggle: () => void }) {
  const barWidth = Math.min(r.ae_ratio / 3, 1) * 100;          // full bar = 3× threshold
  return (
    <>
      <tr onClick={toggle} className="cursor-pointer border-b border-slate-800/60 hover:bg-slate-800/30">
        <td className="py-2 pr-3 font-mono text-slate-500">{r.input_row}</td>
        <td className="pr-3"><VerdictBadge v={r.verdict} /></td>
        <td className="pr-3 font-mono text-slate-300">{r.family}</td>
        {labelled && <td className="pr-3 text-slate-400">{r.true_label}</td>}
        <td className="pr-3">
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full" style={{ width: `${barWidth}%`, background: r.ae_ratio > 1 ? "#e879f9" : "#34d399" }} />
            </div>
            <span className="font-mono text-slate-300">{r.ae_ratio.toFixed(2)}×</span>
          </div>
        </td>
        <td className="pr-3 font-mono text-slate-300">
          {r.lgbm_family} <span className="text-slate-500">{pct(r.lgbm_confidence)}</span>
        </td>
      </tr>
      {open && (
        <tr className="bg-slate-900/80">
          <td colSpan={labelled ? 6 : 5} className="px-3 py-3 text-xs leading-relaxed text-slate-300">
            {explain(r, mode)}
          </td>
        </tr>
      )}
    </>
  );
}

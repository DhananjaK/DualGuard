// Live replay: stream the pre-computed TEST-set verdicts (Step 6) in capture order,
// like an analyst's alert console. Nothing is re-scored here.
import { Pause, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Mode, type ModelId, type ReplayRow, type Verdict } from "../api";
import { Card, ErrorBox, Segmented, Stat, VERDICT_STYLE, VerdictBadge, num, pct } from "../components/ui";
import { useChart } from "../theme";

const TICK_MS = 400;          // one screen update every 0.4 s
const PAGE = 2000;            // rows fetched from the API at a time
const HISTORY = 60;           // ticks kept in the chart

type Status = "caught" | "false alarm" | "missed";
type Counts = { seen: number; tp: number; fn: number; fp: number; tn: number };
type FeedItem = { key: string; r: ReplayRow; v: Verdict; status: Status };
type Tick = { t: number; known: number; unknown: number; missed: number };

const ZERO: Counts = { seen: 0, tp: 0, fn: 0, fp: 0, tn: 0 };
const STATUS_STYLE: Record<Status, string> = {
  caught: "bg-sky-500/15 text-sky-300 ring-sky-500/30",
  "false alarm": "bg-slate-500/15 text-slate-300 ring-slate-500/30",
  missed: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
};

export const fileLabel = (f: string) =>
  f.replace(".pcap_ISCX.csv", "").replace("-WorkingHours", "").replace("-workingHours", "").replace(/-/g, " ");

export default function Replay() {
  const c = useChart();
  const [model, setModel] = useState<ModelId>("B");
  const [mode, setMode] = useState<Mode>("parallel");
  const [files, setFiles] = useState<string[]>([]);
  const [file, setFile] = useState("");
  const [total, setTotal] = useState(0);
  const [start, setStart] = useState(0);
  const [speed, setSpeed] = useState("50");
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState<Counts>(ZERO);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [ticks, setTicks] = useState<Tick[]>([]);
  const [pos, setPos] = useState(0);

  const buffer = useRef<ReplayRow[]>([]);
  const nextFetch = useRef(0);
  const fetching = useRef(false);
  const tickNo = useRef(0);
  const posRef = useRef(0);
  const generation = useRef(0);   // bumps on every reset, so late answers from an old replay are ignored

  const fetchPage = useCallback(async (from: number, gen = generation.current) => {
    fetching.current = true;
    try {
      const r = await api.replay(model, from, PAGE, file || undefined);
      if (gen !== generation.current) return;          // a newer reset happened meanwhile
      setFiles(r.source_files);
      setTotal(r.total);
      buffer.current.push(...r.rows);
      nextFetch.current = from + r.rows.length;
    } catch (e) {
      setError((e as Error).message);
      setPlaying(false);
    } finally {
      fetching.current = false;
    }
  }, [model, file]);

  const reset = useCallback(async (from: number) => {
    setPlaying(false);
    buffer.current = [];
    posRef.current = from;
    tickNo.current = 0;
    setPos(from); setCounts(ZERO); setFeed([]); setTicks([]); setError(null);
    generation.current += 1;
    await fetchPage(from, generation.current);
  }, [fetchPage]);

  // New model or capture file: start again from the beginning.
  useEffect(() => { setStart(0); reset(0); }, [reset]);

  // The replay clock.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const batch = buffer.current.splice(0, Number(speed));
      if (buffer.current.length < PAGE / 2 && !fetching.current && nextFetch.current < total) {
        fetchPage(nextFetch.current);
      }
      if (batch.length === 0) {
        if (!fetching.current && nextFetch.current >= total) setPlaying(false);   // end of data
        return;
      }
      const c = { ...ZERO };
      let known = 0, unknown = 0, missed = 0;
      const items: FeedItem[] = [];
      for (const r of batch) {
        const v = mode === "parallel" ? r.verdict_parallel : r.verdict_cascade;
        const attack = r.Family !== "BENIGN";
        const alarm = v !== "BENIGN";
        c.seen++;
        if (v === "KNOWN_ATTACK") known++;
        if (v === "UNKNOWN_ANOMALY") unknown++;
        if (attack && alarm) { c.tp++; items.push({ key: `${r.row}`, r, v, status: "caught" }); }
        else if (attack && !alarm) { c.fn++; missed++; items.push({ key: `${r.row}`, r, v, status: "missed" }); }
        else if (!attack && alarm) { c.fp++; items.push({ key: `${r.row}`, r, v, status: "false alarm" }); }
        else c.tn++;
      }
      setCounts((p) => ({ seen: p.seen + c.seen, tp: p.tp + c.tp, fn: p.fn + c.fn, fp: p.fp + c.fp, tn: p.tn + c.tn }));
      setFeed((p) => [...items.slice(-40).reverse(), ...p].slice(0, 40));
      tickNo.current += 1;
      setTicks((p) => [...p, { t: tickNo.current, known, unknown, missed }].slice(-HISTORY));
      posRef.current += batch.length;
      setPos(posRef.current);
    }, TICK_MS);
    return () => clearInterval(id);
  }, [playing, speed, mode, total, fetchPage]);

  const attacks = counts.tp + counts.fn;
  const benign = counts.fp + counts.tn;
  const alerts = counts.tp + counts.fp;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-white">Live replay</h1>
        <p className="mt-1 text-sm text-slate-400">
          The test-set flows streamed in capture order, as an analyst's alert console would see them.
        </p>
        <p className="mt-2 inline-block rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-400">
          Replays the verdicts computed once by <span className="font-mono">scripts/06_evaluate_test.py</span>; nothing is re-scored here.
          For live inference use <span className="text-slate-200">Analyse</span>.
        </p>
      </header>

      <Card>
        <div className="flex flex-wrap items-end gap-6">
          <div>
            <Label>Model</Label>
            <Segmented<ModelId> value={model} onChange={setModel}
              options={[{ value: "A", label: "Known" }, { value: "B", label: "Zero-Day" }]} />
          </div>
          <div>
            <Label>Decision logic</Label>
            <Segmented<Mode> value={mode} onChange={(m) => { setMode(m); reset(start); }}
              options={[{ value: "parallel", label: "Parallel" }, { value: "cascade", label: "Cascade" }]} />
          </div>
          <div>
            <Label>Capture</Label>
            <select value={file} onChange={(e) => setFile(e.target.value)}
                    className="rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200">
              <option value="">All test captures</option>
              {files.map((f) => <option key={f} value={f}>{fileLabel(f)}</option>)}
            </select>
          </div>
          <div>
            <Label>Speed (flows / update)</Label>
            <Segmented value={speed} onChange={setSpeed}
              options={["10", "50", "250", "1000"].map((s) => ({ value: s, label: s }))} />
          </div>
          <div className="ml-auto flex gap-2">
            <button onClick={() => setPlaying(!playing)}
                    className="inline-flex items-center gap-2 rounded-xl bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-sky-400">
              {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {playing ? "Pause" : "Play"}
            </button>
            <button onClick={() => reset(start)} title="Restart from the chosen position"
                    className="inline-flex items-center gap-2 rounded-xl border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:border-slate-500">
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div className="mt-5">
          <div className="mb-1 flex justify-between text-[11px] text-slate-500">
            <span>Start position (drag, then release)</span>
            <span className="font-mono">{num(pos)} / {num(total)}</span>
          </div>
          <input type="range" min={0} max={Math.max(total - 1, 0)} step={1000} value={start}
                 onChange={(e) => setStart(Number(e.target.value))}
                 onPointerUp={() => reset(start)} onKeyUp={() => reset(start)}
                 className="w-full accent-sky-400" />
        </div>
      </Card>

      {error && <ErrorBox message={error} />}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Flows seen" value={num(counts.seen)} />
        <Stat label="Alerts" value={num(alerts)} accent="#e879f9" />
        <Stat label="Attacks caught" value={attacks ? pct(counts.tp / attacks) : "–"} hint={`${num(counts.tp)} of ${num(attacks)}`} accent="#38bdf8" />
        <Stat label="Attacks missed" value={num(counts.fn)} accent="#fb7185" />
        <Stat label="False-alarm rate" value={benign ? pct(counts.fp / benign, 2) : "–"} hint={`${num(counts.fp)} benign flows alerted`} />
        <Stat label="Alert precision" value={alerts ? pct(counts.tp / alerts) : "–"} hint="share of alerts that were real" />
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card title="Alerts per update" subtitle="Last 60 updates" className="lg:col-span-3">
          <div className="h-64">
            <ResponsiveContainer>
              <AreaChart data={ticks}>
                <CartesianGrid stroke={c.grid} />
                <XAxis dataKey="t" tick={{ fill: c.muted, fontSize: 10 }} />
                <YAxis tick={{ fill: c.muted, fontSize: 10 }} />
                <Tooltip {...c.tooltip} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area type="monotone" dataKey="known" name="Known attack" stackId="a" stroke={VERDICT_STYLE.KNOWN_ATTACK.hex} fill={VERDICT_STYLE.KNOWN_ATTACK.hex} fillOpacity={0.35} isAnimationActive={false} />
                <Area type="monotone" dataKey="unknown" name="Unknown anomaly" stackId="a" stroke={VERDICT_STYLE.UNKNOWN_ANOMALY.hex} fill={VERDICT_STYLE.UNKNOWN_ANOMALY.hex} fillOpacity={0.35} isAnimationActive={false} />
                <Area type="monotone" dataKey="missed" name="Missed attack" stroke="#fb7185" fill="#fb7185" fillOpacity={0.15} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Event feed" subtitle="Alerts and missed attacks, newest first" className="lg:col-span-2">
          <div className="h-64 space-y-1.5 overflow-y-auto pr-1">
            {feed.length === 0 && <p className="text-xs text-slate-500">Press Play.</p>}
            {feed.map((f) => (
              <div key={f.key} className="flex items-center gap-2 rounded-lg border border-slate-800/80 bg-slate-950/60 px-2.5 py-1.5 text-xs">
                <span className={`w-20 shrink-0 rounded-full px-2 py-0.5 text-center text-[10px] font-semibold ring-1 ${STATUS_STYLE[f.status]}`}>{f.status}</span>
                <VerdictBadge v={f.v} />
                <span className="truncate text-slate-300">{f.r.Label}</span>
                <span className="ml-auto shrink-0 font-mono text-[10px] text-slate-500">#{f.r.row}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function Label({ children }: { children: string }) {
  return <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-slate-500">{children}</div>;
}

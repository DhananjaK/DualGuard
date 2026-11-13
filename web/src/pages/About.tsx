// About: how DualGuard works, what each model is, and its known limits.
import { ArrowDown, ArrowRight, Brain, GitBranch, Scale, TreePine } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { api, type ModelInfo } from "../api";
import { Card, ErrorBox, VERDICTS, VERDICT_STYLE, VerdictBadge } from "../components/ui";

const VERDICT_TEXT = {
  BENIGN: "Nothing suspicious: the autoencoder rebuilds the flow well and LightGBM does not name an attack.",
  KNOWN_ATTACK: "LightGBM names an attack family it was trained on with at least 50% probability.",
  UNKNOWN_ANOMALY: "The autoencoder finds the flow abnormal, but LightGBM cannot confidently name it: a possible new attack.",
} as const;

const LIMITS = [
  "Thresholds fixed on Monday–Wednesday traffic gave 10.7% benign false alarms on Thursday–Friday; ~90% of them come from the DDoS and Infiltration captures (disturbed or possibly mislabelled benign traffic).",
  "Flow statistics only: payload-level attacks (SQL injection, XSS) are invisible to the autoencoder; LightGBM detects known web attacks but names unseen ones 'DoS'.",
  "Low-and-slow attacks whose single flows look normal (e.g. Bot beacons) are mostly missed.",
  "Unseen attacks can never be named correctly; they are flagged as unknown or as the nearest known family.",
  "Very small families (Heartbleed, Infiltration) give unstable percentages; results are reported with counts.",
  "CIC-IDS2017 is a 2017 laboratory capture; behaviour on modern production traffic is not demonstrated.",
  "Hyperparameters were fixed in advance, not tuned. Prototype for coursework, not a production IDS.",
];

export default function About() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.models().then(setModels).catch((e) => setError((e as Error).message)); }, []);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-white">How DualGuard works</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Two complementary detectors look at every network flow. One has learned what <i>normal</i> traffic looks like and
          needs no attack labels; the other has learned the <i>names</i> of attacks from labelled examples.
        </p>
      </header>

      <Card title="Pipeline">
        <div className="flex flex-col items-center gap-3 text-xs">
          <Box title="Network flow" text="77 CICFlowMeter statistics (packet sizes, timings, flags)" />
          <ArrowDown className="h-4 w-4 text-slate-600" />
          <Box title="Preprocessing" text="Drop Destination Port and constant columns · signed log · standardise · clip ±10 (fitted on training data only)" />
          <ArrowDown className="h-4 w-4 text-slate-600" />
          <div className="grid w-full max-w-3xl grid-cols-2 gap-4">
            <Box icon={<Brain className="h-4 w-4 text-fuchsia-300" />} title="Stage 1 – Autoencoder"
                 text="Trained on benign flows only. Compresses 66–68 features to 16 and rebuilds them. High rebuild error = abnormal." />
            <Box icon={<TreePine className="h-4 w-4 text-amber-300" />} title="Stage 2 – LightGBM"
                 text="Gradient-boosted trees trained on labelled flows. Outputs a probability for each known family." />
          </div>
          <ArrowDown className="h-4 w-4 text-slate-600" />
          <Box icon={<GitBranch className="h-4 w-4 text-sky-300" />} title="Decision logic"
               text="Parallel (default): LightGBM confident attack → known; else AE abnormal → unknown; else benign. Cascade: AE first, LightGBM only for abnormal flows." />
          <ArrowDown className="h-4 w-4 text-slate-600" />
          <div className="flex flex-wrap justify-center gap-2">{VERDICTS.map((v) => <VerdictBadge key={v} v={v} />)}</div>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        {VERDICTS.map((v) => (
          <Card key={v}>
            <div className="mb-2"><VerdictBadge v={v} /></div>
            <p className="text-sm text-slate-300">{VERDICT_TEXT[v]}</p>
            <div className="mt-3 h-1 rounded-full" style={{ background: VERDICT_STYLE[v].hex }} />
          </Card>
        ))}
      </div>

      <Card title="The fair comparison rule" right={<Scale className="h-4 w-4 text-slate-500" />}>
        <p className="text-sm leading-relaxed text-slate-300">
          Every threshold is set on <b>validation</b> data so that at most <b>2% of benign flows</b> raise an alarm. In parallel mode,
          LightGBM's false alarms are counted first and the autoencoder only gets the rest of the budget. Every method is compared at
          this same operating point; otherwise any detector could reach 100% detection by alarming on everything. The test data was
          opened once, after the expected results had been written down and committed.
        </p>
      </Card>

      {error && <ErrorBox message={error} />}
      <div className="grid gap-6 lg:grid-cols-2">
        {models.map((m) => (
          <Card key={m.scenario} title={`${m.name} (model ${m.scenario})`} subtitle={`Trained on ${m.trained_on}`}>
            <p className="mb-3 text-sm text-slate-300">{m.use}.</p>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <dt className="text-slate-500">Families known to LightGBM</dt>
              <dd className="font-mono text-slate-300">{m.classes.join(", ")}</dd>
              <dt className="text-slate-500">Features used</dt><dd className="font-mono text-slate-300">{m.n_features}</dd>
              <dt className="text-slate-500">AE threshold (cascade / parallel)</dt>
              <dd className="font-mono text-slate-300">{m.thresholds.ae.toFixed(4)} / {m.thresholds.ae_parallel.toFixed(4)}</dd>
              <dt className="text-slate-500">τ (LightGBM confidence)</dt><dd className="font-mono text-slate-300">{m.thresholds.tau}</dd>
              <dt className="text-slate-500">Default mode</dt><dd className="font-mono text-slate-300">{m.default_mode}</dd>
            </dl>
            <details className="mt-3 text-xs">
              <summary className="cursor-pointer text-sky-300 hover:text-sky-200">Show all {m.n_features} features</summary>
              <p className="mt-2 font-mono leading-relaxed text-slate-400">{m.features.join(" · ")}</p>
            </details>
          </Card>
        ))}
      </div>

      <Card title="Known limitations">
        <ul className="space-y-2 text-sm text-slate-300">
          {LIMITS.map((l) => (
            <li key={l} className="flex gap-2"><ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-slate-600" />{l}</li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

function Box({ title, text, icon }: { title: string; text: string; icon?: ReactNode }) {
  return (
    <div className="w-full max-w-3xl rounded-xl border border-slate-800 bg-slate-950/70 px-4 py-3">
      <div className="flex items-center gap-2 font-semibold text-slate-100">{icon}{title}</div>
      <div className="mt-1 text-slate-400">{text}</div>
    </div>
  );
}

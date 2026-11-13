// Typed client for the DualGuard FastAPI service. Every call goes through /api (Vite proxy).

export type Verdict = "BENIGN" | "KNOWN_ATTACK" | "UNKNOWN_ANOMALY";
export type ModelId = "A" | "B";
export type Mode = "parallel" | "cascade";

export interface ResultRow {
  input_row: number;
  true_label?: string;
  true_family?: string;
  verdict: Verdict;
  family: string;
  ae_error: number;
  ae_threshold: number;
  ae_ratio: number;
  lgbm_family: string;
  lgbm_confidence: number;
  p_benign: number;
}

export interface PredictResponse {
  model: ModelId;
  mode: Mode;
  n_received: number;
  n_scored: number;
  n_invalid: number;
  elapsed_ms: number;
  verdict_counts: Partial<Record<Verdict, number>>;
  family_counts: Record<string, number>;
  truncated: boolean;
  results: ResultRow[];
}

export interface ModelInfo {
  scenario: ModelId;
  name: string;
  trained_on: string;
  use: string;
  classes: string[];
  n_features: number;
  features: string[];
  default_mode: Mode;
  modes: Mode[];
  thresholds: { budget: number; tau: number; ae: number; ae_parallel: number };
}

export interface ReplayRow {
  row: number;
  Day: string;
  Label: string;
  Family: string;
  ae_error: number;
  lgbm_family: string;
  lgbm_conf: number;
  verdict_parallel: Verdict;
  verdict_cascade: Verdict;
  SourceFile: string;
}

export interface ReplayResponse {
  model: ModelId;
  total: number;
  start: number;
  source_files: string[];
  rows: ReplayRow[];
}

export type TableRow = Record<string, number | string | null>;
export type Evidence = Record<string, TableRow[]>;

const BASE = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* body was not JSON */
    }
    throw new Error(`${res.status} – ${msg}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () =>
    fetch(`${BASE}/health`).then((r) => json<{ status: string; models_loaded: ModelId[] }>(r)),

  models: () => fetch(`${BASE}/models`).then((r) => json<ModelInfo[]>(r)),

  samples: () => fetch(`${BASE}/samples`).then((r) => json<string[]>(r)),

  sampleFile: async (name: string): Promise<Blob> => {
    const r = await fetch(`${BASE}/samples/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error(`${r.status} – could not load sample`);
    return r.blob();
  },

  replay: (model: ModelId, start: number, limit: number, sourceFile?: string) => {
    const q = new URLSearchParams({ model, start: String(start), limit: String(limit) });
    if (sourceFile) q.set("source_file", sourceFile);
    return fetch(`${BASE}/replay?${q}`).then((r) => json<ReplayResponse>(r));
  },

  evidence: (model: ModelId) => fetch(`${BASE}/evidence/${model}`).then((r) => json<Evidence>(r)),

  predictCsv: (file: Blob, fileName: string, model: ModelId, mode: Mode) => {
    const form = new FormData();
    form.append("file", file, fileName);
    form.append("model", model);
    form.append("mode", mode);
    return fetch(`${BASE}/predict_csv`, { method: "POST", body: form }).then((r) =>
      json<PredictResponse>(r),
    );
  },
};

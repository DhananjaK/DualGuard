import { Activity, BarChart3, Info, Moon, Radio, ShieldCheck, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api } from "./api";
import About from "./pages/About";
import Analyse from "./pages/Analyse";
import Evidence from "./pages/Evidence";
import Replay from "./pages/Replay";
import { useTheme } from "./theme";

const NAV = [
  { to: "/analyse", label: "Analyse", icon: Activity, hint: "Upload flows, get verdicts" },
  { to: "/replay", label: "Live replay", icon: Radio, hint: "Stream of test-set verdicts" },
  { to: "/evidence", label: "Evidence", icon: BarChart3, hint: "Evaluation results" },
  { to: "/about", label: "About", icon: Info, hint: "How DualGuard works" },
];

function ApiStatus() {
  const [state, setState] = useState<"checking" | "up" | "down">("checking");
  const [models, setModels] = useState<string[]>([]);
  useEffect(() => {
    const check = () =>
      api.health()
        .then((h) => { setState("up"); setModels(h.models_loaded); })
        .catch(() => setState("down"));
    check();
    const id = setInterval(check, 10_000);   // re-check every 10 s
    return () => clearInterval(id);
  }, []);
  const colour = state === "up" ? "bg-emerald-400" : state === "down" ? "bg-rose-500" : "bg-slate-500";
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${colour} ${state === "up" ? "animate-pulse" : ""}`} />
        <span className="font-medium text-slate-300">API {state === "up" ? "online" : state === "down" ? "offline" : "…"}</span>
      </div>
      {state === "up" && <div className="mt-1 text-slate-500">models: {models.join(", ")}</div>}
      {state === "down" && <div className="mt-1 text-slate-500">start: uvicorn api.main:app</div>}
    </div>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button onClick={toggle} title={`Switch to ${next} mode`} aria-label={`Switch to ${next} mode`}
            className="flex w-full items-center justify-between rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs text-slate-300 hover:border-slate-600">
      <span>{theme === "dark" ? "Dark" : "Light"} mode</span>
      {theme === "dark" ? <Moon className="h-4 w-4 text-sky-300" /> : <Sun className="h-4 w-4 text-amber-300" />}
    </button>
  );
}

export default function App() {
  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-950/80 p-4 transition-colors">
        <div className="mb-8 flex items-center gap-2 px-2">
          <ShieldCheck className="h-7 w-7 text-sky-400" />
          <div>
            <div className="text-lg font-bold tracking-tight text-white">DualGuard</div>
            <div className="text-[11px] text-slate-500">Hybrid intrusion detection</div>
          </div>
        </div>
        <nav className="flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon, hint }) => (
            <NavLink
              key={to}
              to={to}
              title={hint}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition ${
                  isActive ? "bg-sky-500/15 text-sky-200 ring-1 ring-sky-500/30" : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto space-y-3">
          <ThemeToggle />
          <ApiStatus />
          <p className="px-2 text-[10px] leading-relaxed text-slate-600">
            Autoencoder + LightGBM on CIC-IDS2017.<br />CS7003NU A2 prototype – not for production use.
          </p>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl p-8">
          <Routes>
            <Route path="/" element={<Navigate to="/analyse" replace />} />
            <Route path="/analyse" element={<Analyse />} />
            <Route path="/replay" element={<Replay />} />
            <Route path="/evidence" element={<Evidence />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

// Light / dark theme: one class on <html>, remembered in localStorage.
//
// How it works: every neutral colour in the UI is a Tailwind "slate" shade, and Tailwind v4
// turns each shade into a CSS variable (--color-slate-950 ...). In light mode index.css
// re-points those variables (dark shades <-> light shades), so the whole UI flips without
// touching each component. Charts draw with explicit colours, so they read them from useChart().
import { createContext, type ReactNode, useContext, useEffect, useState } from "react";

export type Theme = "dark" | "light";
const KEY = "dualguard-theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage unavailable: fall through */
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({ theme: "dark", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("light", theme === "light");
    root.style.colorScheme = theme;            // native controls (select, slider, scrollbars)
    try { localStorage.setItem(KEY, theme); } catch { /* ignore */ }
  }, [theme]);
  return (
    <ThemeContext.Provider value={{ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);

// Bright accents chosen for a dark background are too pale as TEXT on white: darker twins.
const LIGHT_TEXT: Record<string, string> = {
  "#34d399": "#059669", "#fbbf24": "#b45309", "#e879f9": "#a21caf", "#38bdf8": "#0369a1",
  "#fb7185": "#be123c", "#f472b6": "#be185d", "#60a5fa": "#1d4ed8", "#a78bfa": "#6d28d9",
  "#64748b": "#475569", "#e2e8f0": "#0f172a",
};

/** Colours for Recharts and inline styles, matching the current theme. */
export function useChart() {
  const light = useTheme().theme === "light";
  const tick = light ? "#475569" : "#94a3b8";
  return {
    light,
    tone: (hex: string) => (light ? LIGHT_TEXT[hex.toLowerCase()] ?? hex : hex),
    grid: light ? "#e2e8f0" : "#1e293b",
    tick,
    muted: "#64748b",
    text: light ? "#1e293b" : "#cbd5e1",
    strong: light ? "#0f172a" : "#e2e8f0",
    faint: light ? "#94a3b8" : "#475569",
    axis: { fill: tick, fontSize: 11 },
    tooltip: {
      contentStyle: {
        background: light ? "#ffffff" : "#0f172a",
        border: `1px solid ${light ? "#cbd5e1" : "#334155"}`,
        borderRadius: 8,
        fontSize: 12,
        color: light ? "#0f172a" : "#e2e8f0",
      },
    },
  };
}

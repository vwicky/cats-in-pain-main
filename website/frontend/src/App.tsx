import { useEffect, useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import JobPage from "./pages/JobPage";
import ResultsPage from "./pages/ResultsPage";

export default function App() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (typeof window === "undefined") return "dark";
    const saved = window.localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    const isDark = theme === "dark";
    document.documentElement.classList.toggle("dark", isDark);
    document.body.classList.toggle("dark", isDark);
    window.localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="border-b border-slate-200 dark:border-slate-800 px-6 py-4 flex gap-4 items-center">
        <Link to="/" className="font-semibold text-emerald-600 dark:text-emerald-400">
          Cat Pain · MVP
        </Link>
        <span className="text-slate-600 dark:text-slate-500 text-sm">local-first · interpretability</span>
        <button
          type="button"
          onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          className="ml-auto text-xs px-3 py-1.5 rounded-full border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          {theme === "dark" ? "Light theme" : "Dark theme"}
        </button>
      </header>
      <main className="max-w-6xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/job/:id" element={<JobPage />} />
          <Route path="/job/:id/results" element={<ResultsPage />} />
        </Routes>
      </main>
    </div>
  );
}

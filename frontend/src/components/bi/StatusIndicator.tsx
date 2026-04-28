import { useEffect, useState } from "react";
import { Activity, CheckCircle2, AlertTriangle } from "lucide-react";
import { fetchHealth, type HealthResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

export function StatusIndicator() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const tick = () =>
      fetchHealth()
        .then((h) => active && (setHealth(h), setError(null)))
        .catch((e) => active && setError(e.message));
    tick();
    const id = setInterval(tick, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const dbRaw = (health?.db ?? health?.database ?? "").toString().toLowerCase();
  const dbReady = dbRaw === "ready" || dbRaw === "ok" || dbRaw === "connected" || dbRaw === "up";
  const geminiRaw = (health?.gemini ?? health?.gemini_key ?? "").toString().toLowerCase();
  const geminiReady = geminiRaw === "ok" || geminiRaw === "ready" || geminiRaw === "true";

  const Pill = ({
    label,
    ok,
    loading,
  }: {
    label: string;
    ok: boolean;
    loading?: boolean;
  }) => (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
        loading
          ? "border-border bg-muted text-muted-foreground"
          : ok
            ? "border-success/40 bg-success/10 text-success"
            : "border-warning/40 bg-warning/10 text-warning",
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          loading ? "bg-muted-foreground" : ok ? "bg-success animate-pulse-dot" : "bg-warning",
        )}
      />
      {label}
    </div>
  );

  if (error) {
    return (
      <div className="flex items-center gap-1.5 rounded-full border border-destructive/40 bg-destructive/10 px-2.5 py-1 text-xs text-destructive">
        <AlertTriangle className="h-3 w-3" />
        Backend offline
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Pill label={`DB ${dbReady ? "ready" : dbRaw || "…"}`} ok={dbReady} loading={!health} />
      <Pill
        label={`Gemini ${geminiReady ? "ok" : geminiRaw || "…"}`}
        ok={geminiReady}
        loading={!health}
      />
    </div>
  );
}

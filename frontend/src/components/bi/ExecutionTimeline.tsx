import { CheckCircle2, Loader2, XCircle, Wand2, Database, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

export type StepStatus = "pending" | "running" | "complete" | "error";
export interface PipelineStep {
  id: "nl_sql" | "sql_exec" | "chart";
  label: string;
  status: StepStatus;
  durationMs?: number;
  detail?: string;
}

const ICONS = {
  nl_sql: Wand2,
  sql_exec: Database,
  chart: BarChart3,
};

export function ExecutionTimeline({ steps }: { steps: PipelineStep[] }) {
  if (!steps.length) return null;
  return (
    <div className="animate-fade-up rounded-2xl border border-border bg-card p-4 shadow-card">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-display text-sm font-semibold tracking-tight">Execution Pipeline</h3>
        <span className="text-xs text-muted-foreground">
          {steps.filter((s) => s.status === "complete").length}/{steps.length} complete
        </span>
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        {steps.map((step, i) => {
          const Icon = ICONS[step.id];
          return (
            <div
              key={step.id}
              className={cn(
                "relative overflow-hidden rounded-xl border p-3 transition-all",
                step.status === "running" &&
                  "border-primary/50 bg-primary/5 shadow-glow",
                step.status === "complete" && "border-success/30 bg-success/5",
                step.status === "error" && "border-destructive/40 bg-destructive/10",
                step.status === "pending" && "border-border bg-muted/30 opacity-60",
              )}
            >
              <div className="flex items-start gap-2">
                <div
                  className={cn(
                    "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg",
                    step.status === "running" && "bg-primary/15 text-primary",
                    step.status === "complete" && "bg-success/15 text-success",
                    step.status === "error" && "bg-destructive/15 text-destructive",
                    step.status === "pending" && "bg-muted text-muted-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-xs font-semibold text-foreground">
                      {i + 1}. {step.label}
                    </div>
                    {step.status === "running" && (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                    )}
                    {step.status === "complete" && (
                      <CheckCircle2 className="h-3.5 w-3.5 text-success" />
                    )}
                    {step.status === "error" && (
                      <XCircle className="h-3.5 w-3.5 text-destructive" />
                    )}
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-[11px] text-muted-foreground">
                    <span className="capitalize">{step.status}</span>
                    {step.durationMs !== undefined && (
                      <span className="font-mono">· {step.durationMs}ms</span>
                    )}
                  </div>
                  {step.detail && (
                    <div className="mt-1 truncate text-[11px] text-muted-foreground">
                      {step.detail}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

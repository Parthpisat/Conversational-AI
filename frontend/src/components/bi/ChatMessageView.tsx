import { Sparkles, User, AlertTriangle, RefreshCw } from "lucide-react";
import type { ChatMessage } from "@/hooks/useChat";
import { ExecutionTimeline } from "./ExecutionTimeline";
import { SQLPanel } from "./SQLPanel";
import { ChartView } from "./ChartView";
import { DataTable } from "./DataTable";
import { cn } from "@/lib/utils";

export function ChatMessageView({
  message,
  onRegenerate,
}: {
  message: ChatMessage;
  onRegenerate?: () => void;
}) {
  if (message.role === "user") {
    return (
      <div className="flex animate-fade-up justify-end">
        <div className="flex max-w-[85%] items-start gap-2">
          <div className="rounded-2xl rounded-tr-sm border border-primary/30 bg-primary/10 px-4 py-2.5 text-sm text-foreground shadow-card">
            {message.mode === "sql" && (
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-primary">
                SQL Mode
              </div>
            )}
            <div className={cn(message.mode === "sql" && "font-mono text-xs")}>
              {message.content}
            </div>
          </div>
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-secondary text-secondary-foreground">
            <User className="h-4 w-4" />
          </div>
        </div>
      </div>
    );
  }

  const p = message.payload;
  return (
    <div className="flex animate-fade-up gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-primary shadow-glow">
        <Sparkles className="h-4 w-4 text-primary-foreground" />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {p?.steps && <ExecutionTimeline steps={p.steps} />}

        {(p?.sql || p?.explanation) && (
          <SQLPanel sql={p.sql} explanation={p.explanation} reasoning={p.reasoning} />
        )}

        <ChartView chartType={p?.chartType} chartJson={p?.chartJson} />

        {p?.rows && p.rows.length > 0 && <DataTable rows={p.rows} />}

        {p?.message && (
          <div className="rounded-2xl border border-border bg-card px-4 py-3 text-sm text-foreground shadow-card">
            {p.message}
          </div>
        )}

        {p?.error && (
          <div className="flex items-start gap-2 rounded-2xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="flex-1">
              <div className="font-semibold">Something went wrong</div>
              <div className="mt-0.5 text-xs opacity-90">{p.error}</div>
            </div>
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                className="inline-flex items-center gap-1 rounded-md border border-destructive/40 bg-destructive/20 px-2 py-1 text-xs font-medium hover:bg-destructive/30"
              >
                <RefreshCw className="h-3 w-3" /> Retry
              </button>
            )}
          </div>
        )}

        {message.streaming && !p?.error && (
          <div className="flex items-center gap-1 px-2 text-xs text-muted-foreground">
            <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-primary" />
            <span
              className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-primary"
              style={{ animationDelay: "0.2s" }}
            />
            <span
              className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-primary"
              style={{ animationDelay: "0.4s" }}
            />
            <span className="ml-2">Streaming…</span>
          </div>
        )}
      </div>
    </div>
  );
}

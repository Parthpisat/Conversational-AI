import { useEffect, useRef, useState } from "react";
import { Send, Square, Sparkles, Code2, RefreshCw, Trash2, Menu, X } from "lucide-react";
import { useChat } from "@/hooks/useChat";
import { ChatMessageView } from "@/components/bi/ChatMessageView";
import { SchemaExplorer } from "@/components/bi/SchemaExplorer";
import { StatusIndicator } from "@/components/bi/StatusIndicator";
import { cn } from "@/lib/utils";

const SUGGESTIONS = [
  "Show top 10 customers by revenue",
  "Monthly sales trend for last 12 months",
  "Distribution of orders by category",
  "Average order value per region",
];

export function ChatApp() {
  const { messages, isStreaming, send, stop, regenerate, clear } = useChat();
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"nl" | "sql">("nl");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (taRef.current) {
      taRef.current.style.height = "auto";
      taRef.current.style.height = Math.min(taRef.current.scrollHeight, 160) + "px";
    }
  }, [input]);

  const onSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isStreaming) return;
    send(input, mode);
    setInput("");
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-72 border-r border-sidebar-border bg-sidebar transition-transform lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between border-b border-sidebar-border px-4 py-3 lg:hidden">
            <span className="font-display text-sm font-semibold">Schema</span>
            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-md p-1 hover:bg-sidebar-accent"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            <SchemaExplorer />
          </div>
        </div>
      </aside>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main */}
      <main className="flex min-w-0 flex-1 flex-col">
        {/* Header */}
        <header className="flex items-center gap-3 border-b border-border bg-card/40 px-4 py-3 backdrop-blur">
          <button
            onClick={() => setSidebarOpen(true)}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground lg:hidden"
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-primary shadow-glow">
              <Sparkles className="h-4 w-4 text-primary-foreground" />
            </div>
            <div>
              <h1 className="font-display text-base font-bold leading-tight tracking-tight">
                <span className="text-primary">Chatlytics</span>
              </h1>
              <p className="text-[11px] text-muted-foreground">Conversational analytics</p>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div className="hidden sm:block">
              <StatusIndicator />
            </div>
            {messages.length > 0 && (
              <>
                <button
                  onClick={regenerate}
                  disabled={isStreaming}
                  title="Regenerate last query"
                  className="hidden items-center gap-1.5 rounded-md border border-border bg-secondary px-2.5 py-1.5 text-xs text-secondary-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-50 sm:inline-flex"
                >
                  <RefreshCw className="h-3 w-3" />
                  Regenerate
                </button>
                <button
                  onClick={clear}
                  className="inline-flex items-center gap-1.5 rounded-md border border-border bg-secondary px-2.5 py-1.5 text-xs text-secondary-foreground hover:bg-accent hover:text-accent-foreground"
                  title="Clear conversation"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </>
            )}
          </div>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="scrollbar-thin flex-1 overflow-y-auto bg-gradient-glow">
          <div className="mx-auto w-full max-w-4xl space-y-6 px-4 py-6">
            {messages.length === 0 && <EmptyState onPick={(q) => send(q, "nl")} />}
            {messages.map((m) => (
              <ChatMessageView key={m.id} message={m} onRegenerate={regenerate} />
            ))}
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-border bg-card/60 px-4 py-3 backdrop-blur">
          <form onSubmit={onSubmit} className="mx-auto max-w-4xl">
            <div className="flex items-center justify-between pb-2">
              <ModeToggle mode={mode} onChange={setMode} />
              <div className="text-[11px] text-muted-foreground">
                {isStreaming ? "Streaming response…" : "Press ⌘/Ctrl + Enter to send"}
              </div>
            </div>
            <div className="flex items-end gap-2 rounded-2xl border border-border bg-input/50 p-2 shadow-card focus-within:border-primary/60 focus-within:shadow-glow">
              <textarea
                ref={taRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    onSubmit();
                  } else if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
                rows={1}
                placeholder={
                  mode === "nl"
                    ? "Ask anything about your data…"
                    : "SELECT * FROM your_table WHERE …"
                }
                className={cn(
                  "min-h-[40px] flex-1 resize-none bg-transparent px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none",
                  mode === "sql" && "font-mono text-xs",
                )}
              />
              {isStreaming ? (
                <button
                  type="button"
                  onClick={stop}
                  className="inline-flex h-10 items-center gap-1.5 rounded-xl border border-destructive/40 bg-destructive/15 px-3 text-sm font-medium text-destructive transition-colors hover:bg-destructive/25"
                >
                  <Square className="h-3.5 w-3.5 fill-current" />
                  Stop
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="inline-flex h-10 items-center gap-1.5 rounded-xl bg-gradient-primary px-4 text-sm font-semibold text-primary-foreground shadow-glow transition-all hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
                >
                  <Send className="h-3.5 w-3.5" />
                  Send
                </button>
              )}
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

function ModeToggle({
  mode,
  onChange,
}: {
  mode: "nl" | "sql";
  onChange: (m: "nl" | "sql") => void;
}) {
  return (
    <div className="inline-flex items-center rounded-full border border-border bg-muted/50 p-0.5 text-xs font-medium">
      <button
        type="button"
        onClick={() => onChange("nl")}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 transition-colors",
          mode === "nl"
            ? "bg-gradient-primary text-primary-foreground shadow-glow"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <Sparkles className="h-3 w-3" />
        Natural Language
      </button>
      <button
        type="button"
        onClick={() => onChange("sql")}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 transition-colors",
          mode === "sql"
            ? "bg-gradient-primary text-primary-foreground shadow-glow"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <Code2 className="h-3 w-3" />
        Raw SQL
      </button>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-primary shadow-glow">
        <Sparkles className="h-7 w-7 text-primary-foreground" />
      </div>
      <h2 className="font-display text-2xl font-bold tracking-tight">
        Ask your data anything
      </h2>
      <p className="mt-2 max-w-md text-sm text-muted-foreground">
        Type a question in plain English. Watch as your query is translated to SQL, executed, and
        visualized — all in real time.
      </p>
      <div className="mt-8 grid w-full max-w-2xl gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="group rounded-xl border border-border bg-card p-3 text-left text-sm text-foreground/90 shadow-card transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-glow"
          >
            <div className="flex items-center gap-2">
              <Sparkles className="h-3.5 w-3.5 text-primary opacity-60 transition-opacity group-hover:opacity-100" />
              {s}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

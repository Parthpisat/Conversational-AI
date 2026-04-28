import { useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, Code2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SQLPanelProps {
  sql?: string;
  explanation?: string;
  reasoning?: string[] | string;
  onRegenerate?: () => void;
}

export function SQLPanel({ sql, explanation, reasoning }: SQLPanelProps) {
  const [copied, setCopied] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);

  if (!sql && !explanation) return null;

  const reasoningArr = Array.isArray(reasoning)
    ? reasoning
    : typeof reasoning === "string" && reasoning
      ? [reasoning]
      : [];

  const copy = async () => {
    if (!sql) return;
    await navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="animate-fade-up rounded-2xl border border-border bg-card shadow-card">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Code2 className="h-4 w-4 text-primary" />
          <span className="font-display text-sm font-semibold">Generated SQL</span>
        </div>
        {sql && (
          <button
            onClick={copy}
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-secondary px-2 py-1 text-xs text-secondary-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
            {copied ? "Copied" : "Copy"}
          </button>
        )}
      </div>

      {sql && (
        <pre className="scrollbar-thin max-h-72 overflow-auto px-4 py-3 font-mono text-[12.5px] leading-relaxed text-foreground/90">
          <code>{highlightSQL(sql)}</code>
        </pre>
      )}

      {explanation && (
        <div className="border-t border-border bg-muted/30 px-4 py-3">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Explanation
          </div>
          <p className="text-sm text-foreground/90">{explanation}</p>
        </div>
      )}

      {reasoningArr.length > 0 && (
        <div className="border-t border-border">
          <button
            onClick={() => setShowReasoning((v) => !v)}
            className="flex w-full items-center justify-between px-4 py-2 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <span>Reasoning steps ({reasoningArr.length})</span>
            {showReasoning ? (
              <ChevronUp className="h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </button>
          {showReasoning && (
            <ol className="space-y-1.5 px-4 pb-3 text-xs text-foreground/80">
              {reasoningArr.map((r, i) => (
                <li key={i} className="flex gap-2">
                  <span className="font-mono text-primary">{i + 1}.</span>
                  <span>{r}</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}

const KEYWORDS =
  /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|AS|AND|OR|NOT|IN|IS|NULL|DISTINCT|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|WITH|UNION|ALL|INSERT|UPDATE|DELETE|VALUES|SET|INTO|CREATE|TABLE|DROP|ALTER|DESC|ASC)\b/gi;

function highlightSQL(sql: string) {
  // Render with simple span tokenization
  const parts: Array<{ text: string; cls?: string }> = [];
  let last = 0;
  const re = new RegExp(KEYWORDS.source, "gi");
  let m: RegExpExecArray | null;
  while ((m = re.exec(sql))) {
    if (m.index > last) parts.push({ text: sql.slice(last, m.index) });
    parts.push({ text: m[0], cls: "text-primary font-semibold" });
    last = m.index + m[0].length;
  }
  if (last < sql.length) parts.push({ text: sql.slice(last) });
  return parts.map((p, i) =>
    p.cls ? (
      <span key={i} className={p.cls}>
        {p.text}
      </span>
    ) : (
      <span key={i}>{p.text}</span>
    ),
  );
}

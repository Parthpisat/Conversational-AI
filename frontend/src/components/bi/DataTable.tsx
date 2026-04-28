import { useMemo, useState } from "react";
import { Download, ChevronLeft, ChevronRight, Table2 } from "lucide-react";
import { downloadCSV } from "@/lib/csv";

const PAGE_SIZE = 25;

export function DataTable({ rows }: { rows: Array<Record<string, any>> }) {
  const [page, setPage] = useState(0);
  const headers = useMemo(() => (rows[0] ? Object.keys(rows[0]) : []), [rows]);
  if (!rows?.length) return null;
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="animate-fade-up overflow-hidden rounded-2xl border border-border bg-card shadow-card">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Table2 className="h-4 w-4 text-primary" />
          <span className="font-display text-sm font-semibold">Results</span>
          <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-mono text-muted-foreground">
            {rows.length} rows
          </span>
        </div>
        <button
          onClick={() => downloadCSV("query-results.csv", rows)}
          className="inline-flex items-center gap-1.5 rounded-md border border-border bg-secondary px-2 py-1 text-xs text-secondary-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          <Download className="h-3 w-3" />
          CSV
        </button>
      </div>

      <div className="scrollbar-thin max-h-[420px] overflow-auto">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-card/95 backdrop-blur">
            <tr>
              {headers.map((h) => (
                <th
                  key={h}
                  className="border-b border-border px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-border/50 transition-colors hover:bg-muted/40"
              >
                {headers.map((h) => (
                  <td key={h} className="px-3 py-2 font-mono text-xs text-foreground/90">
                    {formatCell(row[h])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-border px-4 py-2 text-xs text-muted-foreground">
          <span>
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 disabled:opacity-40 hover:bg-accent hover:text-accent-foreground"
            >
              <ChevronLeft className="h-3 w-3" /> Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 disabled:opacity-40 hover:bg-accent hover:text-accent-foreground"
            >
              Next <ChevronRight className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function formatCell(v: any) {
  if (v === null || v === undefined) return <span className="text-muted-foreground">—</span>;
  if (typeof v === "object") return JSON.stringify(v);
  if (typeof v === "number") return v.toLocaleString();
  return String(v);
}

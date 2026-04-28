import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Database, Link2, Table as TableIcon } from "lucide-react";
import { fetchSchema, type SchemaResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

function normalizeTables(schema: SchemaResponse | null) {
  if (!schema) return [];
  const t = schema.tables;
  if (!t) return [];
  const normalizeCols = (cols: any): { name: string; type?: string }[] => {
    if (!cols) return [];
    if (Array.isArray(cols)) {
      return cols.map((c: any) =>
        typeof c === "string" ? { name: c } : { name: c.name, type: c.type },
      );
    }
    // object form: { col_name: "TYPE — description" }
    return Object.entries(cols).map(([name, type]) => ({
      name,
      type: typeof type === "string" ? type : undefined,
    }));
  };
  if (Array.isArray(t)) {
    return t.map((tbl: any) => ({
      name: tbl.name ?? "table",
      columns: normalizeCols(tbl.columns),
    }));
  }
  // object form { tableName: { columns: {...} | [...] } | [cols] }
  return Object.entries(t).map(([name, val]: [string, any]) => {
    const cols = Array.isArray(val) ? val : (val?.columns ?? val);
    return { name, columns: normalizeCols(cols) };
  });
}

export function SchemaExplorer() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    fetchSchema()
      .then((s) => {
        if (active) setSchema(s);
      })
      .catch((e) => active && setError(e.message))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  const tables = normalizeTables(schema);
  const relationships = schema?.relationships ?? [];

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <div className="flex items-center gap-2 text-sidebar-foreground">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-primary">
          <Database className="h-4 w-4 text-primary-foreground" />
        </div>
        <div>
          <div className="font-display text-sm font-semibold">Schema Explorer</div>
          <div className="text-xs text-muted-foreground">
            {loading ? "Loading…" : `${tables.length} tables`}
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive-foreground">
          Could not load schema: {error}
        </div>
      )}

      <div className="scrollbar-thin flex-1 space-y-1 overflow-y-auto pr-1">
        {tables.map((tbl) => {
          const isOpen = !!open[tbl.name];
          return (
            <div key={tbl.name} className="rounded-lg">
              <button
                onClick={() => setOpen((o) => ({ ...o, [tbl.name]: !isOpen }))}
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm text-sidebar-foreground transition-colors hover:bg-sidebar-accent"
              >
                {isOpen ? (
                  <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                )}
                <TableIcon className="h-3.5 w-3.5 text-primary" />
                <span className="font-mono text-xs">{tbl.name}</span>
                <span className="ml-auto text-[10px] text-muted-foreground">
                  {tbl.columns.length}
                </span>
              </button>
              {isOpen && (
                <div className="ml-6 mt-1 space-y-0.5 border-l border-sidebar-border pl-3">
                  {tbl.columns.map((c: any) => (
                    <div
                      key={c.name}
                      className="flex items-center justify-between py-0.5 text-xs"
                    >
                      <span className="font-mono text-sidebar-foreground/80">{c.name}</span>
                      {c.type && (
                        <span className="text-[10px] uppercase text-muted-foreground">
                          {c.type}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
        {!loading && !tables.length && !error && (
          <div className="rounded-lg border border-dashed border-sidebar-border p-3 text-xs text-muted-foreground">
            No tables returned.
          </div>
        )}
      </div>

      {relationships.length > 0 && (
        <div className="border-t border-sidebar-border pt-3">
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-sidebar-foreground">
            <Link2 className="h-3.5 w-3.5" />
            Relationships
          </div>
          <div className="space-y-1">
            {relationships.map((r: any, i: number) => {
              const text =
                typeof r === "string" ? r : `${r.from ?? "?"} → ${r.to ?? "?"}`;
              return (
                <div key={i} className="font-mono text-[11px] text-muted-foreground">
                  {text}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

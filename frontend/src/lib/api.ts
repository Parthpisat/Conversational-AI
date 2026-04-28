export const API_BASE = "http://localhost:8000";

export interface SchemaColumn {
  name: string;
  type?: string;
  nullable?: boolean;
}
export interface SchemaTable {
  name: string;
  columns: SchemaColumn[] | string[];
}
export interface SchemaResponse {
  tables?: SchemaTable[] | Record<string, any>;
  relationships?: Array<{ from: string; to: string; type?: string }>;
  [k: string]: any;
}

export interface HealthResponse {
  status?: string;
  db?: string;
  database?: string;
  gemini?: string;
  gemini_key?: string;
  [k: string]: any;
}

export async function fetchSchema(): Promise<SchemaResponse> {
  const res = await fetch(`${API_BASE}/schema`);
  if (!res.ok) throw new Error(`Schema fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export type SSEEvent = {
  event: string;
  data: any;
};

/**
 * Stream POST /query (or /query/sql) with SSE-style chunked response.
 * Backend may return either text/event-stream OR newline-delimited JSON.
 * We parse both.
 */
export async function streamQuery(
  payload: { query: string; mode: "nl" | "sql" },
  onEvent: (e: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const endpoint = payload.mode === "sql" ? "/query/sql" : "/query";
  const body = payload.mode === "sql" ? { sql: payload.query } : { question: payload.query };

  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Query request failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flushSSEBlock = (block: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const raw of block.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;
    const dataStr = dataLines.join("\n");
    let parsed: any = dataStr;
    try {
      parsed = JSON.parse(dataStr);
    } catch {
      /* keep as string */
    }
    // Some backends embed event name in data
    if (parsed && typeof parsed === "object" && parsed.event && !block.includes("event:")) {
      onEvent({ event: parsed.event, data: parsed.data ?? parsed });
    } else {
      onEvent({ event, data: parsed });
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Try SSE delimiter first (\n\n)
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      if (block.trim()) flushSSEBlock(block);
    }
  }

  // Flush remaining
  if (buffer.trim()) {
    // Could be NDJSON if no SSE delimiters arrived
    const lines = buffer.split("\n").filter((l) => l.trim());
    for (const l of lines) {
      try {
        const parsed = JSON.parse(l);
        onEvent({ event: parsed.event ?? "message", data: parsed.data ?? parsed });
      } catch {
        flushSSEBlock(buffer);
        break;
      }
    }
  }
}

import { useCallback, useRef, useState } from "react";
import { streamQuery, type SSEEvent } from "@/lib/api";
import type { PipelineStep } from "@/components/bi/ExecutionTimeline";
import type { ChartJson } from "@/components/bi/ChartView";

export interface AssistantPayload {
  steps: PipelineStep[];
  sql?: string;
  explanation?: string;
  reasoning?: string[] | string;
  chartType?: string;
  chartJson?: ChartJson;
  rows?: Array<Record<string, any>>;
  error?: string;
  message?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: "nl" | "sql";
  payload?: AssistantPayload;
  streaming?: boolean;
  createdAt: number;
}

const STEP_TEMPLATE: PipelineStep[] = [
  { id: "nl_sql", label: "NL → SQL Translation", status: "pending" },
  { id: "sql_exec", label: "SQL Execution", status: "pending" },
  { id: "chart", label: "Chart Generation", status: "pending" },
];

function newId() {
  return Math.random().toString(36).slice(2, 10);
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const updateAssistant = useCallback(
    (id: string, updater: (a: AssistantPayload) => AssistantPayload, opts?: { done?: boolean }) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                payload: updater(m.payload ?? { steps: [...STEP_TEMPLATE] }),
                streaming: opts?.done ? false : m.streaming,
              }
            : m,
        ),
      );
    },
    [],
  );

  const setStep = useCallback(
    (id: string, stepId: PipelineStep["id"], patch: Partial<PipelineStep>) => {
      updateAssistant(id, (a) => ({
        ...a,
        steps: a.steps.map((s) => (s.id === stepId ? { ...s, ...patch } : s)),
      }));
    },
    [updateAssistant],
  );

  const send = useCallback(
    async (query: string, mode: "nl" | "sql") => {
      const trimmed = query.trim();
      if (!trimmed || isStreaming) return;

      const userMsg: ChatMessage = {
        id: newId(),
        role: "user",
        content: trimmed,
        mode,
        createdAt: Date.now(),
      };
      const assistantId = newId();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        payload: {
          steps:
            mode === "sql"
              ? STEP_TEMPLATE.filter((s) => s.id !== "nl_sql").map((s) => ({ ...s }))
              : STEP_TEMPLATE.map((s) => ({ ...s })),
        },
        createdAt: Date.now(),
      };
      setMessages((p) => [...p, userMsg, assistantMsg]);
      setIsStreaming(true);

      const startedAt: Partial<Record<PipelineStep["id"], number>> = {};
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      // For NL mode, mark step 1 running immediately
      if (mode === "nl") {
        startedAt.nl_sql = performance.now();
        setStep(assistantId, "nl_sql", { status: "running" });
      } else {
        startedAt.sql_exec = performance.now();
        setStep(assistantId, "sql_exec", { status: "running" });
      }

      try {
        await streamQuery({ query: trimmed, mode }, (e: SSEEvent) => {
          handleEvent(assistantId, e, startedAt, setStep, updateAssistant);
        }, ctrl.signal);

        // Finalize: any non-pending non-error step → complete
        updateAssistant(
          assistantId,
          (a) => ({
            ...a,
            steps: a.steps.map((s) =>
              s.status === "running"
                ? {
                    ...s,
                    status: "complete",
                    durationMs:
                      startedAt[s.id] !== undefined
                        ? Math.round(performance.now() - startedAt[s.id]!)
                        : s.durationMs,
                  }
                : s,
            ),
          }),
          { done: true },
        );
      } catch (err: any) {
        if (err?.name === "AbortError") {
          updateAssistant(
            assistantId,
            (a) => ({ ...a, error: "Stopped." }),
            { done: true },
          );
        } else {
          updateAssistant(
            assistantId,
            (a) => ({
              ...a,
              error: err?.message ?? "Request failed. Is the backend running on :8000?",
              steps: a.steps.map((s) =>
                s.status === "running" ? { ...s, status: "error" } : s,
              ),
            }),
            { done: true },
          );
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming, setStep, updateAssistant],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const regenerate = useCallback(() => {
    // Find the last user message and resend
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (lastUser) send(lastUser.content, lastUser.mode ?? "nl");
  }, [messages, send]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, isStreaming, send, stop, regenerate, clear };
}

function handleEvent(
  id: string,
  e: SSEEvent,
  startedAt: Partial<Record<PipelineStep["id"], number>>,
  setStep: (id: string, stepId: PipelineStep["id"], patch: Partial<PipelineStep>) => void,
  updateAssistant: (id: string, u: (a: AssistantPayload) => AssistantPayload) => void,
) {
  const evt = (e.event ?? "").toString();
  const d = e.data ?? {};

  // NL → SQL translation
  if (evt === "nlp_translation_start" || evt === "nl_sql_start") {
    startedAt.nl_sql = performance.now();
    setStep(id, "nl_sql", { status: "running" });
    return;
  }
  if (evt === "nlp_translation_complete" || evt === "nl_sql_complete") {
    const ms =
      startedAt.nl_sql !== undefined ? Math.round(performance.now() - startedAt.nl_sql) : undefined;
    setStep(id, "nl_sql", { status: "complete", durationMs: ms });
    updateAssistant(id, (a) => ({
      ...a,
      sql: d.sql ?? d.query ?? a.sql,
      explanation: d.explanation ?? d.description ?? a.explanation,
      reasoning: d.reasoning ?? d.steps ?? a.reasoning,
    }));
    startedAt.sql_exec = performance.now();
    setStep(id, "sql_exec", { status: "running" });
    return;
  }

  // SQL execution
  if (evt === "sql_execution_start" || evt === "sql_exec_start") {
    startedAt.sql_exec = performance.now();
    setStep(id, "sql_exec", { status: "running" });
    return;
  }
  if (evt === "sql_execution_complete" || evt === "sql_exec_complete") {
    const ms =
      startedAt.sql_exec !== undefined
        ? Math.round(performance.now() - startedAt.sql_exec)
        : undefined;
    const rows = d.rows ?? d.data ?? d.results;
    setStep(id, "sql_exec", {
      status: "complete",
      durationMs: ms,
      detail: rows ? `${rows.length} rows` : undefined,
    });
    if (rows) updateAssistant(id, (a) => ({ ...a, rows }));
    startedAt.chart = performance.now();
    setStep(id, "chart", { status: "running" });
    return;
  }

  // Chart generation
  if (evt === "chart_generation_start" || evt === "chart_start") {
    startedAt.chart = performance.now();
    setStep(id, "chart", { status: "running" });
    return;
  }
  if (evt === "chart_generation_complete" || evt === "chart_complete") {
    const ms =
      startedAt.chart !== undefined ? Math.round(performance.now() - startedAt.chart) : undefined;
    setStep(id, "chart", {
      status: "complete",
      durationMs: ms,
      detail: d.chart_type,
    });
    updateAssistant(id, (a) => ({
      ...a,
      chartType: d.chart_type ?? a.chartType,
      chartJson: d.chart_json ?? a.chartJson,
    }));
    return;
  }

  // Final response
  if (evt === "response_complete" || evt === "complete" || evt === "done") {
    const details = d.details ?? d;
    updateAssistant(id, (a) => ({
      ...a,
      sql: details.sql ?? a.sql,
      explanation: details.explanation ?? a.explanation,
      reasoning: details.reasoning_steps ?? details.reasoning ?? a.reasoning,
      chartType: details.chart_type ?? a.chartType,
      chartJson: details.chart_json ?? a.chartJson,
      rows: details.table ?? details.rows ?? details.data ?? a.rows,
      message: details.message ?? details.summary ?? a.message,
    }));
    return;
  }

  if (evt === "error") {
    updateAssistant(id, (a) => ({
      ...a,
      error: d.message ?? d.error ?? "An error occurred.",
      steps: a.steps.map((s) => (s.status === "running" ? { ...s, status: "error" } : s)),
    }));
    return;
  }

  // Unknown but maybe useful — try to absorb common fields
  if (d && typeof d === "object") {
    updateAssistant(id, (a) => ({
      ...a,
      sql: d.sql ?? a.sql,
      explanation: d.explanation ?? a.explanation,
      chartType: d.chart_type ?? a.chartType,
      chartJson: d.chart_json ?? a.chartJson,
      rows: d.rows ?? d.data ?? a.rows,
    }));
  }
}

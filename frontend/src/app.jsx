import { useState, useEffect, useRef, useCallback } from "react";
import { BarChart, Bar, LineChart, Line, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API = "http://localhost:8000";

const SUGGESTIONS = [
  "Top 10 most ordered products",
  "Which departments have the highest reorder rate?",
  "Show order volume by hour of day",
  "Top 5 aisles by reorder rate and average basket position",
  "What % of products in each department are organic?",
  "Average days between orders per user",
];

// ── Recharts chart renderer ──
function RechartsChart({ data, config, type }) {
  if (!data || !config) return null;

  const randomColor = () => `#${Math.floor(Math.random()*16777215).toString(16).padStart(6, '0')}`;

  const renderChart = () => {
    switch (type) {
      case 'bar':
        return (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={config.x} />
            <YAxis />
            <Tooltip />
            <Legend />
            {config.y.map(key => <Bar key={key} dataKey={key} fill={randomColor()} />)}
          </BarChart>
        );
      case 'line':
        return (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={config.x} />
            <YAxis />
            <Tooltip />
            <Legend />
            {config.y.map(key => <Line key={key} type="monotone" dataKey={key} stroke={randomColor()} />)}
          </LineChart>
        );
      case 'pie':
        return (
          <PieChart>
            <Pie data={data} dataKey={config.values} nameKey={config.names} cx="50%" cy="50%" outerRadius={100} label>
              {data.map((entry, index) => <Cell key={`cell-${index}`} fill={randomColor()} />)}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        );
      default:
        return null;
    }
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      {renderChart()}
    </ResponsiveContainer>
  );
}

// ── Data table renderer ──
function DataTable({ rows, columns }) {
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  if (!rows || rows.length === 0)
    return (
      <div style={{ color: "var(--text-muted)", padding: 16 }}>No data</div>
    );

  const paginatedRows = rows.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paginatedRows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c}>
                  {row[c] === null || row[c] === undefined ? (
                    <span style={{ color: "var(--text-muted)" }}>—</span>
                  ) : typeof row[c] === "number" ? (
                    Number.isInteger(row[c])
                      ? row[c].toLocaleString()
                      : row[c].toFixed(4)
                  ) : (
                    String(row[c])
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="pagination">
        <button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage === 1}>Previous</button>
        <span>Page {currentPage} of {Math.ceil(rows.length / rowsPerPage)}</span>
        <button onClick={() => setCurrentPage(p => Math.min(Math.ceil(rows.length / rowsPerPage), p + 1))} disabled={currentPage * rowsPerPage >= rows.length}>Next</button>
      </div>
    </div>
  );
}

// ── SQL block with copy ──
function SQLBlock({ sql }) {
  const [copied, setCopied] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!collapsed && ref.current && window.hljs) {
      window.hljs.highlightElement(ref.current);
    }
  }, [sql, collapsed]);

  const copy = () => {
    navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="sql-block">
      <div className="sql-header">
        <span className="sql-label" onClick={() => setCollapsed(c => !c)} style={{ cursor: 'pointer' }}>
          Generated SQL {collapsed ? '▶' : '▼'}
        </span>
        <button className="copy-btn" onClick={copy}>
          {copied ? "✓ Copied" : "Copy"}
        </button>
      </div>
      {!collapsed && (
        <pre>
          <code ref={ref} className="language-sql">
            {sql}
          </code>
        </pre>
      )}
    </div>
  );
}

// ── Schema panel ──
function SchemaPanel({ schema }) {
  if (!schema)
    return (
      <div className="empty-state">
        <div className="empty-icon">📋</div>
        <div className="empty-title">Loading schema…</div>
      </div>
    );
  return (
    <div>
      {Object.entries(schema.tables).map(([name, info]) => (
        <div key={name} className="schema-table-section">
          <div className="schema-table-name">{name}</div>
          <div className="schema-desc">{info.description}</div>
          {Object.entries(info.columns).map(([col, dtype]) => (
            <div key={col} className="schema-col">
              <span className="schema-col-name">{col}</span>
              <span className="schema-col-type">{dtype}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ── Visualization panel ──
function VizPanel({ result, schema }) {
  const [tab, setTab] = useState("chart");

  useEffect(() => {
    if (!result) return;
    if (result.chart_json && result.chart_type !== "table") setTab("chart");
    else setTab("table");
  }, [result]);

  if (!result) {
    return (
      <div className="viz-panel">
        <div className="viz-content">
          <div
            className="empty-state"
            style={{ height: "calc(100vh - 120px)" }}
          >
            <div className="empty-icon">📊</div>
            <div className="empty-title">Ask a question</div>
            <div className="empty-sub">
              Type a question in plain English on the left. Charts and tables
              will appear here.
            </div>
          </div>
        </div>
      </div>
    );
  }

  const hasChart = result.chart_json && result.chart_type !== "table";

  return (
    <div className="viz-panel">
      <div className="viz-tabs">
        {hasChart && (
          <button
            className={`viz-tab ${tab === "chart" ? "active" : ""}`}
            onClick={() => setTab("chart")}
          >
            📈 Chart
          </button>
        )}
        <button
          className={`viz-tab ${tab === "table" ? "active" : ""}`}
          onClick={() => setTab("table")}
        >
          📋 Table
        </button>
        <button
          className={`viz-tab ${tab === "sql" ? "active" : ""}`}
          onClick={() => setTab("sql")}
        >
          🔍 SQL
        </button>
        <button
          className={`viz-tab ${tab === "schema" ? "active" : ""}`}
          onClick={() => setTab("schema")}
        >
          🗂 Schema
        </button>
      </div>

      <div className="viz-content">
        {result.error && (
          <div className="error-banner">⚠️ {result.error}</div>
        )}

        {result.explanation && tab !== "schema" && (
          <div className="explanation-box">💡 {result.explanation}</div>
        )}

        {tab === "chart" && hasChart && (
          <>
            <div className="info-cards">
              <div className="info-card">
                <div className="info-card-label">Rows returned</div>
                <div className="info-card-value">
                  {result.row_count?.toLocaleString() ?? "—"}
                </div>
              </div>
              <div className="info-card">
                <div className="info-card-label">Chart type</div>
                <div
                  className="info-card-value"
                  style={{ textTransform: "capitalize" }}
                >
                  {result.chart_type}
                </div>
              </div>
            </div>
            <div className="chart-container">
              <RechartsChart data={result.chart_json.data} config={result.chart_json.config} type={result.chart_type} />
            </div>

          </>
        )}

        {tab === "table" && (
          <>
            <div className="row-count">
              Showing {result.table?.length?.toLocaleString()} of{" "}
              {result.row_count?.toLocaleString()} rows
            </div>
            <DataTable rows={result.table} columns={result.columns} />
          </>
        )}

        {tab === "sql" && <SQLBlock sql={result.sql} />}

        {tab === "schema" && <SchemaPanel schema={schema} />}
      </div>
    </div>
  );
}

// ── Main App ──
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeResult, setActiveResult] = useState(null);
  const [dbStatus, setDbStatus] = useState("loading");
  const [schema, setSchema] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef(null);
  const conversationHistoryRef = useRef([]);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`);
        const d = await r.json();
        setDbStatus(d.status === "ok" ? "ok" : "loading");
      } catch {
        setDbStatus("err");
      }
    };
    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetch(`${API}/schema`)
      .then((r) => r.json())
      .then(setSchema)
      .catch(() => {});
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(
    async (questionOverride) => {
      const question = (questionOverride || input).trim();
      if (!question || loading) return;

      setInput("");
      setLoading(true);
      setShowSuggestions(false);

      const userMsg = { role: "user", content: question, id: Date.now() };
      setMessages((prev) => [...prev, userMsg]);
      conversationHistoryRef.current.push({ role: "user", content: question });

      try {
        const resp = await fetch(`${API}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            conversation_history: conversationHistoryRef.current.slice(-10),
          }),
        });

        const data = await resp.json();

        const assistantMsg = {
          role: "assistant",
          content: data.explanation || "Query complete.",
          id: Date.now() + 1,
          result: data,
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setActiveResult(data);

        conversationHistoryRef.current.push({
          role: "assistant",
          content: `SQL: ${data.sql}\nResult: ${data.row_count} rows. ${data.explanation}`,
        });
      } catch (e) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${e.message}`,
            id: Date.now() + 1,
            error: true,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, loading]
  );



  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <div className="topbar">
        <span className="topbar-logo">BI Agent</span>
        <span className="topbar-sub">Instacart Dataset</span>
        <span
          className={`status-dot ${dbStatus}`}
          style={{ marginLeft: "auto" }}
        />
        <span className="status-label">
          {dbStatus === "ok"
            ? "DB Ready"
            : dbStatus === "loading"
            ? "Loading DB…"
            : "DB Error"}
        </span>
      </div>

      <div className="main">
        {/* Chat panel */}
        <div className="chat-panel">
          <div className="messages">
            {messages.length === 0 && (
              <div
                style={{
                  textAlign: "center",
                  padding: "40px 16px",
                  color: "var(--text-muted)",
                }}
              >
                <div style={{ fontSize: 36, marginBottom: 8 }}>🛒</div>
                <div
                  style={{
                    fontSize: 15,
                    fontWeight: 600,
                    color: "var(--text)",
                  }}
                >
                  Instacart BI Agent
                </div>
                <div style={{ fontSize: 13, marginTop: 8 }}>
                  Ask anything about 3.4M orders across 6 tables.
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div key={msg.id} className={`msg ${msg.role}`}>
                <div className="msg-role">
                  {msg.role === "user" ? "You" : "Agent"}
                </div>
                <div
                  className="msg-bubble"
                  style={
                    msg.error
                      ? {
                          borderColor: "rgba(239,68,68,0.4)",
                          color: "#f87171",
                        }
                      : {}
                  }
                >
                  {msg.content}
                </div>
                {msg.result && (
                  <div
                    className="msg-meta"
                    onClick={() => setActiveResult(msg.result)}
                  >
                    📊 {msg.result.row_count?.toLocaleString()} rows ·{" "}
                    {msg.result.chart_type} chart — view →
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="msg assistant">
                <div className="msg-role">Agent</div>
                <div
                  className="msg-bubble"
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <span className="spinner" /> Thinking…
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {showSuggestions && (
            <div className="suggestions">
              <div className="suggestions-label">Try asking…</div>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="sugg-chip"
                  onClick={() => sendMessage(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          <div className="input-area">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask a question in plain English…"
              disabled={loading || dbStatus !== "ok"}
              rows={2}
            />
            <button
              className="send-btn"
              onClick={() => sendMessage()}
              disabled={loading || !input.trim() || dbStatus !== "ok"}
            >
              ↑
            </button>
          </div>
        </div>

        {/* Viz panel */}
        <VizPanel result={activeResult} schema={schema} />
      </div>
    </>
  );
}
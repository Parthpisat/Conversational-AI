import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3 } from "lucide-react";

export interface ChartConfig {
  x?: string;
  y?: string | string[];
  title?: string;
}
export interface ChartJson {
  data: Array<Record<string, any>>;
  config?: ChartConfig;
}

const COLORS = [
  "var(--color-chart-1)",
  "var(--color-chart-2)",
  "var(--color-chart-3)",
  "var(--color-chart-4)",
  "var(--color-chart-5)",
];

export function ChartView({
  chartType,
  chartJson,
}: {
  chartType?: string;
  chartJson?: ChartJson;
}) {
  if (!chartJson || !chartJson.data?.length) return null;
  const type = (chartType ?? "").toLowerCase();
  if (type === "table" || !type) return null;

  const { data, config = {} } = chartJson;
  const xKey = config.x ?? Object.keys(data[0])[0];
  const yKeys = Array.isArray(config.y)
    ? config.y
    : config.y
      ? [config.y]
      : Object.keys(data[0])
          .filter((k) => k !== xKey && typeof data[0][k] === "number")
          .slice(0, 5);

  return (
    <div className="animate-fade-up rounded-2xl border border-border bg-card p-4 shadow-card">
      <div className="mb-3 flex items-center gap-2">
        <BarChart3 className="h-4 w-4 text-primary" />
        <h3 className="font-display text-sm font-semibold">
          {config.title ?? "Visualization"}
        </h3>
        <span className="ml-auto text-[10px] uppercase tracking-wider text-muted-foreground">
          {type}
        </span>
      </div>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {renderChart(type, data, xKey, yKeys)}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function renderChart(
  type: string,
  data: any[],
  xKey: string,
  yKeys: string[],
): React.ReactElement {
  const tooltipStyle = {
    backgroundColor: "var(--color-popover)",
    border: "1px solid var(--color-border)",
    borderRadius: 8,
    fontSize: 12,
    color: "var(--color-popover-foreground)",
  };
  const axisStyle = { fill: "var(--color-muted-foreground)", fontSize: 11 };

  if (type === "pie") {
    const valueKey = yKeys[0] ?? "value";
    return (
      <PieChart>
        <Pie
          data={data}
          dataKey={valueKey}
          nameKey={xKey}
          outerRadius={100}
          innerRadius={50}
          paddingAngle={2}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11, color: "var(--color-muted-foreground)" }} />
      </PieChart>
    );
  }

  if (type === "line") {
    return (
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis dataKey={xKey} tick={axisStyle} stroke="var(--color-border)" />
        <YAxis tick={axisStyle} stroke="var(--color-border)" />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {yKeys.map((k, i) => (
          <Line
            key={k}
            type="monotone"
            dataKey={k}
            stroke={COLORS[i % COLORS.length]}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        ))}
      </LineChart>
    );
  }

  // default bar
  return (
    <BarChart data={data}>
      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
      <XAxis dataKey={xKey} tick={axisStyle} stroke="var(--color-border)" />
      <YAxis tick={axisStyle} stroke="var(--color-border)" />
      <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--color-muted)", opacity: 0.3 }} />
      <Legend wrapperStyle={{ fontSize: 11 }} />
      {yKeys.map((k, i) => (
        <Bar key={k} dataKey={k} fill={COLORS[i % COLORS.length]} radius={[6, 6, 0, 0]} />
      ))}
    </BarChart>
  );
}

import { createFileRoute } from "@tanstack/react-router";
import { ChatApp } from "@/components/bi/ChatApp";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "Chatlytics — Conversational Business Intelligence" },
      {
        name: "description",
        content:
          "Ask your database anything in plain English. Real-time SQL generation, execution and visualization powered by AI.",
      },
      { property: "og:title", content: "Chatlytics — Conversational BI" },
      {
        property: "og:description",
        content: "Conversational analytics with streaming SQL pipelines and live charts.",
      },
    ],
  }),
});

function Index() {
  return <ChatApp />;
}

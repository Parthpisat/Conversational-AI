import logging
import os
import time
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from database import db
from models import QueryRequest, QueryResponse, SQLRequest, SchemaResponse
from nlp_to_sql import NLToSQL
from chart_builder import build_chart
from streaming_events import streaming_emitter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# App lifecycle: load data on startup
# ---------------------------------------------------------------------------

nl_engine: NLToSQL = None  # initialized after DB loads (needs schema text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nl_engine
    logger.info("🚀 Starting BI Agent backend...")

    # Load all CSVs into DuckDB
    load_results = db.load_data()
    logger.info(f"Load results: {load_results}")

    # Initialize NL→SQL engine with live schema text
    schema_text = db.get_schema_for_prompt()
    sample_rows_text = db.get_sample_rows_for_prompt()
    nl_engine = NLToSQL(schema_text=schema_text, sample_rows_text=sample_rows_text)
    logger.info("✅ NL→SQL engine ready")

    yield  # App is running

    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Conversational BI Agent",
    description="Ask questions in plain English about Instacart shopping data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — returns DB load status."""
    status = db.status()
    return {
        "status": "ok" if status.get("loaded") else "loading",
        "db": status,
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.get("/schema", response_model=SchemaResponse)
def get_schema():
    """Returns the full schema for the frontend schema explorer panel."""
    schema = db.get_schema()
    return SchemaResponse(**schema)


@app.post("/query", response_class=StreamingResponse)
def run_query_stream(req: QueryRequest):
    """
    Main endpoint: accepts a natural language question, streams pipeline events,
    then returns SQL, chart JSON, and table data.
    
    Streams events in Server-Sent Events (SSE) format for real-time progress display.
    """
    def event_generator():
        """Generator that yields SSE formatted events."""
        start_time = time.time()
        last_event_index = [0]  # Track last emitted event index
        
        # Clear previous events from emitter
        streaming_emitter.events = []
        streaming_emitter.step_timings = {}
        
        try:
            if not db._loaded:
                error_response = {
                    "step_number": 0,
                    "event_name": "error",
                    "name": "Database Loading",
                    "status": "error",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "error": "Database still loading, try again shortly."
                }
                yield f"event: error\ndata: {json.dumps(error_response)}\n\n"
                return

            question = req.question.strip()
            if not question:
                error_response = {
                    "step_number": 0,
                    "event_name": "error",
                    "name": "Invalid Input",
                    "status": "error",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "error": "Question cannot be empty."
                }
                yield f"event: error\ndata: {json.dumps(error_response)}\n\n"
                return

            # --- Step 1: NL → SQL ---
            try:
                result = nl_engine.translate(
                    question=question,
                    conversation_history=req.conversation_history,
                )
                
                # Yield any new events from translation
                for event in streaming_emitter.events[last_event_index[0]:]:
                    yield f"event: {event.event_name}\ndata: {event.to_json()}\n\n"
                    last_event_index[0] += 1
                    
            except Exception as e:
                logger.error(f"NL→SQL API error: {e}")
                # Yield error event
                for event in streaming_emitter.events[last_event_index[0]:]:
                    yield f"event: {event.event_name}\ndata: {event.to_json()}\n\n"
                    last_event_index[0] += 1
                return

            sql = result["sql"]
            chart_type = result.get("chart_type", "table")
            explanation = result.get("explanation", "")
            reasoning_steps = result.get("reasoning_steps", [])

            # --- Step 2: Execute SQL (with 1 retry on failure) ---
            error_msg = None
            rows, columns, total_rows = [], [], 0

            try:
                rows, columns, total_rows = db.execute_query(sql)
                
                # Yield any new events from SQL execution
                for event in streaming_emitter.events[last_event_index[0]:]:
                    yield f"event: {event.event_name}\ndata: {event.to_json()}\n\n"
                    last_event_index[0] += 1
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"SQL failed: {error_msg}. Attempting self-correction...")
                
                # Yield error event
                for event in streaming_emitter.events[last_event_index[0]:]:
                    yield f"event: {event.event_name}\ndata: {event.to_json()}\n\n"
                    last_event_index[0] += 1

                # Self-correction: re-ask LLM with the error
                try:
                    retry_result = nl_engine.retry_with_error(
                        original_question=question,
                        failed_sql=sql,
                        error_message=error_msg,
                        conversation_history=req.conversation_history,
                    )
                    sql = retry_result["sql"]
                    chart_type = retry_result.get("chart_type", chart_type)
                    explanation = retry_result.get("explanation", explanation)
                    reasoning_steps = retry_result.get("reasoning_steps", reasoning_steps)
                    rows, columns, total_rows = db.execute_query(sql)
                    error_msg = None  # Cleared — retry succeeded
                    logger.info("Self-correction succeeded ✓")
                    
                    # Yield retry events
                    for event in streaming_emitter.events[last_event_index[0]:]:
                        yield f"event: {event.event_name}\ndata: {event.to_json()}\n\n"
                        last_event_index[0] += 1
                        
                except Exception as retry_err:
                    logger.error(f"Self-correction also failed: {retry_err}")
                    
                    # Yield final result anyway
                    final_result = {
                        "question": question,
                        "sql": sql,
                        "explanation": explanation,
                        "chart_type": "table",
                        "chart_json": None,
                        "table": rows,
                        "columns": columns,
                        "row_count": total_rows,
                        "error": str(retry_err),
                        "reasoning_steps": reasoning_steps,
                        "response_time_ms": (time.time() - start_time) * 1000
                    }
                    final_event = {
                        "step_number": 999,
                        "event_name": "response_complete",
                        "name": "Response Complete",
                        "status": "complete",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "details": final_result
                    }
                    yield f"event: response_complete\ndata: {json.dumps(final_event)}\n\n"
                    return

            # --- Step 3: Build chart ---
            chart_json = None
            if rows and chart_type != "table":
                try:
                    df = db.execute_query_df(sql)
                    chart_json = build_chart(df, chart_type, question)
                    
                    # Yield any new events from chart building
                    for event in streaming_emitter.events[last_event_index[0]:]:
                        yield f"event: {event.event_name}\ndata: {event.to_json()}\n\n"
                        last_event_index[0] += 1
                        
                except Exception as e:
                    logger.warning(f"Chart build failed: {e}")
                    # Non-fatal: return table data without chart

            # --- Final Response ---
            response_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"📊 Response Data - Time: {response_time_ms:.2f}ms, Reasoning Steps: {len(reasoning_steps) if reasoning_steps else 0}")
            if reasoning_steps:
                logger.info(f"📝 Reasoning steps included: {reasoning_steps}")
            
            final_result = {
                "question": question,
                "sql": sql,
                "explanation": explanation,
                "chart_type": chart_type,
                "chart_json": chart_json,
                "table": rows,
                "columns": columns,
                "row_count": total_rows,
                "error": error_msg,
                "reasoning_steps": reasoning_steps,
                "response_time_ms": response_time_ms
            }
            
            # Emit final result event
            final_event = {
                "step_number": 999,
                "event_name": "response_complete",
                "name": "Response Complete",
                "status": "complete",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "details": final_result
            }
            yield f"event: response_complete\ndata: {json.dumps(final_event)}\n\n"
        
        except Exception as e:
            logger.error(f"Unexpected error in stream: {e}")
            error_response = {
                "step_number": 0,
                "event_name": "error",
                "name": "Unexpected Error",
                "status": "error",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(e)
            }
            yield f"event: error\ndata: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/query/sql", response_model=QueryResponse)
def run_raw_sql(req: SQLRequest):
    """
    Run arbitrary SQL directly — useful for debugging or power users.
    No NL translation, no retry logic.
    """
    if not db._loaded:
        raise HTTPException(status_code=503, detail="Database still loading.")

    try:
        rows, columns, total_rows = db.execute_query(req.sql)
        df = db.execute_query_df(req.sql)
        chart_json = build_chart(df, "bar")  # Default to bar for raw SQL
    except Exception as e:
        return QueryResponse(
            question="[Raw SQL]",
            sql=req.sql,
            explanation="Direct SQL execution",
            chart_type="table",
            error=str(e),
        )

    return QueryResponse(
        question="[Raw SQL]",
        sql=req.sql,
        explanation="Direct SQL execution",
        chart_type="bar",
        chart_json=chart_json,
        table=rows,
        columns=columns,
        row_count=total_rows,
    )


# @app.post("/explain")
# def explain_chart(req: ExplainRequest):
#     """
#     Explain the chart data using the Gemini API.
#     """
#     if not req.result:
#         raise HTTPException(status_code=400, detail="Result not provided.")

#     try:
#         explanation = nl_engine.explain(req.result)
#         return {"explanation": explanation}
#     except Exception as e:
#         logger.error(f"Gemini API error: {e}")
#         raise HTTPException(status_code=502, detail=f"Explanation failed: {e}")
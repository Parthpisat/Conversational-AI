import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import db
from models import QueryRequest, QueryResponse, SQLRequest, SchemaResponse
from nlp_to_sql import NLToSQL
from chart_builder import build_chart

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


@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """
    Main endpoint: accepts a natural language question, returns SQL,
    Plotly chart JSON, and table data.

    Includes automatic error recovery: if the generated SQL fails,
    Claude is re-prompted with the error to self-correct (1 retry).
    """
    if not db._loaded:
        raise HTTPException(status_code=503, detail="Database still loading, try again shortly.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # --- Step 1: NL → SQL ---
    try:
        result = nl_engine.translate(
            question=question,
            conversation_history=req.conversation_history,
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=502, detail=f"NL→SQL translation failed: {e}")

    sql = result["sql"]
    chart_type = result.get("chart_type", "table")
    explanation = result.get("explanation", "")

    # --- Step 2: Execute SQL (with 1 retry on failure) ---
    error_msg = None
    rows, columns, total_rows = [], [], 0

    try:
        rows, columns, total_rows = db.execute_query(sql)
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"SQL failed: {error_msg}. Attempting self-correction...")

        # Self-correction: re-ask Gemini with the error
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
            rows, columns, total_rows = db.execute_query(sql)
            error_msg = None  # Cleared — retry succeeded
            logger.info("Self-correction succeeded ✓")
        except Exception as retry_err:
            logger.error(f"Self-correction also failed: {retry_err}")
            return QueryResponse(
                question=question,
                sql=sql,
                explanation=explanation,
                chart_type="table",
                error=f"Query failed after retry: {retry_err}",
            )

    # --- Step 3: Build chart ---
    chart_json = None
    if rows and chart_type != "table":
        try:
            df = db.execute_query_df(sql)
            chart_json = build_chart(df, chart_type, question)
        except Exception as e:
            logger.warning(f"Chart build failed: {e}")
            # Non-fatal: return table data without chart

    return QueryResponse(
        question=question,
        sql=sql,
        explanation=explanation,
        chart_type=chart_type,
        chart_json=chart_json,
        table=rows,
        columns=columns,
        row_count=total_rows,
        error=error_msg,
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
from pydantic import BaseModel
from typing import Optional, Any


class QueryRequest(BaseModel):
    question: str
    conversation_history: list[dict] = []  # [{role, content}] for memory


class SQLRequest(BaseModel):
    sql: str


class QueryResponse(BaseModel):
    question: str
    sql: str
    explanation: str
    chart_type: str
    chart_json: Optional[dict] = None   # Plotly figure JSON
    table: Optional[list[dict]] = None  # Row data for table view
    columns: Optional[list[str]] = None
    row_count: int = 0
    error: Optional[str] = None
    reasoning_steps: Optional[list[dict]] = None  # Chain of Thought steps: [{step: int, description: str}]
    response_time_ms: Optional[float] = None  # Total response time in milliseconds


class SchemaResponse(BaseModel):
    tables: dict[str, Any]
    relationships: list[str]
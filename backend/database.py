"""
database.py — DuckDB-based data layer for the Instacart BI Agent.

Design decisions:
- DuckDB is used instead of pandas for query execution. It handles the
  32M-row order_products__prior table via columnar storage without loading
  everything into RAM.
- CSVs are registered as DuckDB views so queries run directly on the files
  (lazy loading). For repeated queries, they are materialized as tables.
- order_products__prior and order_products__train are unioned into a single
  `order_products_all` view so analysts don't need to think about eval_set.
- NaN in days_since_prior_order (first orders) is preserved as SQL NULL.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Adjust this path to wherever your CSVs live
# Get the absolute path to the directory containing this file (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Resolve DATA_DIR relative to the project root
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# Table definitions: logical name → CSV filename
CSV_FILES = {
    "orders": "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
    "products": "products.csv",
    "aisles": "aisles.csv",
    "departments": "departments.csv",
}

# Column schemas for the system prompt (used by NL→SQL module)
SCHEMA_INFO = {
    "orders": {
        "description": "Core order table (~3.4M rows)",
        "columns": {
            "order_id": "INTEGER PRIMARY KEY",
            "user_id": "INTEGER",
            "eval_set": "VARCHAR — 'prior' or 'train' or 'test'",
            "order_number": "INTEGER — nth order for this user (starts at 1)",
            "order_dow": "INTEGER — day of week (0=Saturday, 1=Sunday, ...)",
            "order_hour_of_day": "INTEGER — hour 0–23",
            "days_since_prior_order": "FLOAT — NULL for a user's first order",
        },
    },
    "order_products_prior": {
        "description": "Products in prior orders (~32M rows)",
        "columns": {
            "order_id": "INTEGER — FK → orders.order_id",
            "product_id": "INTEGER — FK → products.product_id",
            "add_to_cart_order": "INTEGER — position added to cart",
            "reordered": "INTEGER — 1 if reordered, 0 if first time",
        },
    },
    "order_products_train": {
        "description": "Products in train orders (~1.4M rows, same schema as prior)",
        "columns": {
            "order_id": "INTEGER — FK → orders.order_id",
            "product_id": "INTEGER — FK → products.product_id",
            "add_to_cart_order": "INTEGER",
            "reordered": "INTEGER",
        },
    },
    "products": {
        "description": "Product catalog (~50K products)",
        "columns": {
            "product_id": "INTEGER PRIMARY KEY",
            "product_name": "VARCHAR",
            "aisle_id": "INTEGER — FK → aisles.aisle_id",
            "department_id": "INTEGER — FK → departments.department_id",
        },
    },
    "aisles": {
        "description": "Aisle lookup (134 aisles)",
        "columns": {
            "aisle_id": "INTEGER PRIMARY KEY",
            "aisle": "VARCHAR — aisle name",
        },
    },
    "departments": {
        "description": "Department lookup (21 departments)",
        "columns": {
            "department_id": "INTEGER PRIMARY KEY",
            "department": "VARCHAR — department name",
        },
    },
    "order_products_all": {
        "description": "UNION of prior + train order products (virtual view, ~33.4M rows). Use this for full coverage.",
        "columns": {
            "order_id": "INTEGER",
            "product_id": "INTEGER",
            "add_to_cart_order": "INTEGER",
            "reordered": "INTEGER",
            "source": "VARCHAR — 'prior' or 'train'",
        },
    },
}

RELATIONSHIPS = [
    "orders.order_id → order_products_prior.order_id (1:many)",
    "orders.order_id → order_products_train.order_id (1:many)",
    "order_products_all is a UNION VIEW of prior + train",
    "order_products_prior.product_id → products.product_id",
    "order_products_train.product_id → products.product_id",
    "products.aisle_id → aisles.aisle_id",
    "products.department_id → departments.department_id",
]


class Database:
    """Singleton DuckDB connection with lazy CSV loading."""

    def __init__(self):
        # In-memory DuckDB — fast, no disk I/O for intermediate results
        self.conn = duckdb.connect(database=":memory:")
        self._loaded = False
        self._load_errors: list[str] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def load_data(self) -> dict:
        """
        Register all CSVs as DuckDB views, then materialize the small
        lookup tables (aisles, departments, products) as real tables for
        speed. The two large order_products files stay as views (lazy).
        """
        if self._loaded:
            return self.status()

        logger.info(f"Loading data from {DATA_DIR}")
        start = time.time()
        results = {}

        for table_name, filename in CSV_FILES.items():
            path = DATA_DIR / filename
            if not path.exists():
                msg = f"File not found: {path}"
                logger.warning(msg)
                self._load_errors.append(msg)
                results[table_name] = {"status": "missing", "path": str(path)}
                continue

            try:
                # Register as a DuckDB view — reads are lazy/columnar
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name}
                    AS SELECT * FROM read_csv_auto('{path}', header=true, nullstr='')
                """)

                # Materialize small tables for join performance
                if table_name in ("aisles", "departments", "products"):
                    self.conn.execute(f"""
                        CREATE OR REPLACE TABLE {table_name}_mat AS
                        SELECT * FROM {table_name}
                    """)
                    self.conn.execute(f"DROP VIEW {table_name}")
                    self.conn.execute(f"ALTER TABLE {table_name}_mat RENAME TO {table_name}")

                row_count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]

                results[table_name] = {"status": "loaded", "rows": row_count}
                logger.info(f"  ✓ {table_name}: {row_count:,} rows")

            except Exception as e:
                msg = f"Error loading {table_name}: {e}"
                logger.error(msg)
                self._load_errors.append(msg)
                results[table_name] = {"status": "error", "error": str(e)}

        # Create the unified order_products_all view
        try:
            self.conn.execute("""
                CREATE OR REPLACE VIEW order_products_all AS
                SELECT order_id, product_id, add_to_cart_order, reordered,
                       'prior' AS source
                FROM order_products_prior
                UNION ALL
                SELECT order_id, product_id, add_to_cart_order, reordered,
                       'train' AS source
                FROM order_products_train
            """)
            results["order_products_all"] = {"status": "view_created"}
            logger.info("  ✓ order_products_all view created (prior ∪ train)")
        except Exception as e:
            logger.warning(f"Could not create union view: {e}")

        elapsed = time.time() - start
        logger.info(f"Data loading complete in {elapsed:.1f}s")
        self._loaded = True
        return results

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute_query(
        self, sql: str, max_rows: int = 500
    ) -> tuple[list[dict], list[str], int]:
        """
        Execute SQL and return (rows, columns, total_row_count).
        Caps result at max_rows for the API response; full count is
        always returned so the UI can show "Showing N of M rows".
        """
        if not self._loaded:
            raise RuntimeError("Database not loaded. Call load_data() first.")

        # Run the query
        result = self.conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        all_rows = result.fetchall()
        total = len(all_rows)

        # Convert to list of dicts, truncate for response
        rows = [dict(zip(columns, row)) for row in all_rows[:max_rows]]

        # Sanitize: convert non-JSON-serializable types
        for row in rows:
            for k, v in row.items():
                if isinstance(v, float) and (v != v):  # NaN check
                    row[k] = None
                elif isinstance(v, (np.integer, np.floating)):  # numpy scalar
                    row[k] = v.item()
                elif isinstance(v, np.ndarray):  # numpy array
                    row[k] = v.tolist()
                elif isinstance(v, (np.bool_)):
                    row[k] = bool(v)
                elif hasattr(v, "item"):
                    row[k] = v.item()

        return rows, columns, total

    def execute_query_df(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a pandas DataFrame (for chart building)."""
        if not self._loaded:
            raise RuntimeError("Database not loaded.")
        return self.conn.execute(sql).df()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        if not self._loaded:
            return {"loaded": False, "errors": self._load_errors}

        tables = {}
        for name in list(CSV_FILES.keys()) + ["order_products_all"]:
            try:
                count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {name}"
                ).fetchone()[0]
                tables[name] = count
            except Exception:
                tables[name] = "unavailable"

        return {
            "loaded": True,
            "tables": tables,
            "errors": self._load_errors,
        }

    def get_schema(self) -> dict:
        return {
            "tables": SCHEMA_INFO,
            "relationships": RELATIONSHIPS,
        }

    def get_schema_for_prompt(self) -> str:
        """
        Returns a compact schema string to inject into the Claude system
        prompt so it can generate accurate SQL.
        """
        lines = ["=== DATABASE SCHEMA (DuckDB SQL dialect) ===\n"]

        for table, info in SCHEMA_INFO.items():
            lines.append(f"TABLE: {table}")
            lines.append(f"  {info['description']}")
            for col, dtype in info["columns"].items():
                lines.append(f"  - {col}: {dtype}")
            lines.append("")

        lines.append("=== RELATIONSHIPS ===")
        for rel in RELATIONSHIPS:
            lines.append(f"  • {rel}")

        lines.append("""
=== IMPORTANT NOTES ===
- Use order_products_all (the UNION view) for full product coverage unless
  the question specifically asks about prior/train split.
- days_since_prior_order is NULL for a user's first order — always use
  COALESCE or WHERE ... IS NOT NULL when filtering/aggregating this column.
- order_dow: 0=Saturday, 1=Sunday, 2=Monday, ..., 6=Friday
- reordered: 1 means the product was reordered, 0 means first time in cart
- Always add LIMIT clauses — tables can be very large.
- For reorder rate: AVG(reordered) works because the column is 0/1.
""")
        return "\n".join(lines)

    def get_sample_rows_for_prompt(self) -> str:
        """
        Returns a formatted string of sample rows for key tables to give the model
        a concrete sense of the data.
        """
        lines = ["=== SAMPLE ROWS ==="]
        for table in ["orders", "products", "order_products_all"]:
            try:
                df = self.execute_query_df(f"SELECT * FROM {table} LIMIT 3")
                lines.append(f"-- {table} --")
                lines.append(df.to_string(index=False))
                lines.append("")
            except Exception as e:
                logger.warning(f"Could not get sample rows for {table}: {e}")
        return "\n".join(lines)


# Global singleton instance
db = Database()
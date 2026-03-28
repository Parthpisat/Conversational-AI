import os
import time
import logging
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Resolve DATA_DIR relative to this file, not cwd.
# database.py lives in .../backend/ so parent is the project root
# and data/ is its sibling folder.
_THIS_DIR = Path(__file__).parent
_DEFAULT_DATA = _THIS_DIR.parent / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA)))

# Table definitions: logical name → CSV filename
CSV_FILES = {
    "orders": "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
    "products": "products.csv",
    "aisles": "aisles.csv",
    "departments": "departments.csv",
}

# Explicit pandas dtypes per table.
# Using "Int64" (capital I) for nullable integers — preserves NaN without
# converting the column to float, which would corrupt join keys.
# days_since_prior_order is intentionally left out so pandas keeps it as
# float64, which naturally represents NaN for first orders.
DTYPE_MAP = {
    "orders": {
        "order_id":            "Int64",
        "user_id":             "Int64",
        "eval_set":            "str",
        "order_number":        "Int64",
        "order_dow":           "Int64",
        "order_hour_of_day":   "Int64",
    },
    "order_products_prior": {
        "order_id":          "Int64",
        "product_id":        "Int64",
        "add_to_cart_order": "Int64",
        "reordered":         "Int64",
    },
    "order_products_train": {
        "order_id":          "Int64",
        "product_id":        "Int64",
        "add_to_cart_order": "Int64",
        "reordered":         "Int64",
    },
    "products": {
        "product_id":    "Int64",
        "product_name":  "str",
        "aisle_id":      "Int64",
        "department_id": "Int64",
    },
    "aisles": {
        "aisle_id": "Int64",
        "aisle":    "str",
    },
    "departments": {
        "department_id": "Int64",
        "department":    "str",
    },
}

# Column schemas for the system prompt (used by NL->SQL module)
SCHEMA_INFO = {
    "orders": {
        "description": "Core order table (~3.4M rows)",
        "columns": {
            "order_id":               "INTEGER PRIMARY KEY",
            "user_id":                "INTEGER",
            "eval_set":               "VARCHAR — 'prior' or 'train' or 'test'",
            "order_number":           "INTEGER — nth order for this user (starts at 1)",
            "order_dow":              "INTEGER — day of week (0=Saturday, 1=Sunday, ...)",
            "order_hour_of_day":      "INTEGER — hour 0-23",
            "days_since_prior_order": "FLOAT — NULL for a user's first order",
        },
    },
    "order_products_prior": {
        "description": "Products in prior orders (~32M rows)",
        "columns": {
            "order_id":          "INTEGER — FK -> orders.order_id",
            "product_id":        "INTEGER — FK -> products.product_id",
            "add_to_cart_order": "INTEGER — position added to cart",
            "reordered":         "INTEGER — 1 if reordered, 0 if first time",
        },
    },
    "order_products_train": {
        "description": "Products in train orders (~1.4M rows, same schema as prior)",
        "columns": {
            "order_id":          "INTEGER — FK -> orders.order_id",
            "product_id":        "INTEGER — FK -> products.product_id",
            "add_to_cart_order": "INTEGER",
            "reordered":         "INTEGER",
        },
    },
    "products": {
        "description": "Product catalog (~50K products)",
        "columns": {
            "product_id":    "INTEGER PRIMARY KEY",
            "product_name":  "VARCHAR",
            "aisle_id":      "INTEGER — FK -> aisles.aisle_id",
            "department_id": "INTEGER — FK -> departments.department_id",
        },
    },
    "aisles": {
        "description": "Aisle lookup (134 aisles)",
        "columns": {
            "aisle_id": "INTEGER PRIMARY KEY",
            "aisle":    "VARCHAR — aisle name",
        },
    },
    "departments": {
        "description": "Department lookup (21 departments)",
        "columns": {
            "department_id": "INTEGER PRIMARY KEY",
            "department":    "VARCHAR — department name",
        },
    },
    "order_products_all": {
        "description": "UNION of prior + train order products (virtual view, ~33.4M rows). Use this for full coverage.",
        "columns": {
            "order_id":          "INTEGER",
            "product_id":        "INTEGER",
            "add_to_cart_order": "INTEGER",
            "reordered":         "INTEGER",
            "source":            "VARCHAR — 'prior' or 'train'",
        },
    },
}

RELATIONSHIPS = [
    "orders.order_id -> order_products_prior.order_id (1:many)",
    "orders.order_id -> order_products_train.order_id (1:many)",
    "order_products_all is a UNION VIEW of prior + train",
    "order_products_prior.product_id -> products.product_id",
    "order_products_train.product_id -> products.product_id",
    "products.aisle_id -> aisles.aisle_id",
    "products.department_id -> departments.department_id",
]


class Database:
    """Singleton DuckDB connection with CSV loading via pandas."""

    def __init__(self):
        self.conn = duckdb.connect(database=":memory:")
        self._loaded = False
        self._load_errors: list[str] = []

    def load_data(self) -> dict:
        """
        Load all 6 CSVs into DuckDB as materialized tables.

        Strategy:
        - pandas reads each CSV with explicit dtypes (avoids read_csv_auto
          parameter syntax which breaks on older DuckDB versions).
        - Each DataFrame is registered then immediately materialized as a
          real DuckDB table so all subsequent SQL runs at DuckDB speed.
        - order_products_all is created as a VIEW unioning prior + train.
        - Errors are logged but never crash the server.
        """
        if self._loaded:
            return self.status()

        data_dir = DATA_DIR.resolve()
        logger.info(f"Loading CSVs from: {data_dir}")

        if not data_dir.exists():
            msg = (
                f"DATA_DIR not found: {data_dir} | "
                f"Set DATA_DIR env var or place CSVs in {_DEFAULT_DATA}"
            )
            logger.error(msg)
            self._load_errors.append(msg)
            self._loaded = True
            return {"loaded": False, "errors": self._load_errors}

        start = time.time()
        results = {}

        for table_name, filename in CSV_FILES.items():
            path = data_dir / filename

            if not path.exists():
                msg = f"CSV not found: {path}"
                logger.error(msg)
                self._load_errors.append(msg)
                results[table_name] = {"status": "missing", "path": str(path)}
                continue

            try:
                logger.info(f"  Reading {filename} ...")
                dtypes = DTYPE_MAP.get(table_name, {})
                df = pd.read_csv(path, dtype=dtypes, low_memory=False)
                row_count = len(df)

                if row_count == 0:
                    msg = f"{table_name} loaded 0 rows — check {path}"
                    logger.warning(msg)
                    self._load_errors.append(msg)

                # Register the DataFrame, materialise as DuckDB table, unregister.
                # This works on ALL DuckDB versions — no read_csv_auto quirks.
                tmp_name = f"_tmp_{table_name}"
                self.conn.register(tmp_name, df)
                self.conn.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {tmp_name}"
                )
                self.conn.unregister(tmp_name)

                results[table_name] = {"status": "loaded", "rows": row_count}
                logger.info(f"  ✓ {table_name}: {row_count:,} rows")

            except Exception as e:
                msg = f"Error loading {table_name}: {e}"
                logger.error(msg)
                self._load_errors.append(msg)
                results[table_name] = {"status": "error", "error": str(e)}

        # Unified view: prior UNION ALL train
        prior_ok = results.get("order_products_prior", {}).get("status") == "loaded"
        train_ok = results.get("order_products_train", {}).get("status") == "loaded"

        if prior_ok and train_ok:
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
                union_count = self.conn.execute(
                    "SELECT COUNT(*) FROM order_products_all"
                ).fetchone()[0]
                results["order_products_all"] = {
                    "status": "view_created", "rows": union_count
                }
                logger.info(f"  ✓ order_products_all (prior + train): {union_count:,} rows")
            except Exception as e:
                msg = f"Could not create order_products_all view: {e}"
                logger.error(msg)
                self._load_errors.append(msg)
        else:
            missing = [t for t, ok in [
                ("order_products_prior", prior_ok),
                ("order_products_train", train_ok),
            ] if not ok]
            logger.warning(f"Skipping order_products_all — missing tables: {missing}")

        elapsed = time.time() - start
        logger.info(
            f"Load complete in {elapsed:.1f}s | errors: {len(self._load_errors)}"
        )
        self._loaded = True
        return results

    def execute_query(
        self, sql: str, max_rows: int = 500
    ) -> tuple[list[dict], list[str], int]:
        """
        Execute SQL and return (rows, columns, total_row_count).
        Caps result at max_rows for the API; full count always returned.
        All non-JSON-serializable numpy types are sanitized.
        """
        if not self._loaded:
            raise RuntimeError("Database not loaded. Call load_data() first.")

        result = self.conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        all_rows = result.fetchall()
        total = len(all_rows)

        rows = [dict(zip(columns, row)) for row in all_rows[:max_rows]]

        for row in rows:
            for k, v in row.items():
                if v is None:
                    pass
                elif isinstance(v, float) and v != v:      # NaN check
                    row[k] = None
                elif isinstance(v, np.integer):
                    row[k] = int(v)
                elif isinstance(v, np.floating):
                    row[k] = None if np.isnan(v) else float(v)
                elif isinstance(v, np.bool_):
                    row[k] = bool(v)
                elif isinstance(v, np.ndarray):
                    row[k] = v.tolist()
                elif hasattr(v, "item"):
                    row[k] = v.item()

        return rows, columns, total

    def execute_query_df(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a pandas DataFrame (used by chart builder)."""
        if not self._loaded:
            raise RuntimeError("Database not loaded.")
        return self.conn.execute(sql).df()

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
        """Compact schema string injected into the Claude system prompt."""
        lines = ["=== DATABASE SCHEMA (DuckDB SQL dialect) ===\n"]

        for table, info in SCHEMA_INFO.items():
            lines.append(f"TABLE: {table}")
            lines.append(f"  {info['description']}")
            for col, dtype in info["columns"].items():
                lines.append(f"  - {col}: {dtype}")
            lines.append("")

        lines.append("=== RELATIONSHIPS ===")
        for rel in RELATIONSHIPS:
            lines.append(f"  * {rel}")

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
        """Sample rows for key tables — gives Claude concrete data intuition."""
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
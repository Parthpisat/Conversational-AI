import json
import logging
import os
import time
from typing import Optional
from dotenv import load_dotenv
from ollama_client import ollama_client
from streaming_events import streaming_emitter

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "llama3.2:latest"

# ---------------------------------------------------------------------------
# Few-shot examples injected into the system prompt
# These cover the hardest join patterns so Claude learns the schema deeply.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
=== FEW-SHOT SQL EXAMPLES ===

Q: What are the top 10 most ordered products?

{
  "sql": "SELECT p.product_name, COUNT(*) AS order_count FROM order_products_all op JOIN products p ON op.product_id = p.product_id GROUP BY p.product_name ORDER BY order_count DESC LIMIT 10",
  "chart_type": "bar",
  "explanation": "Counts total appearances across all orders (prior + train) for each product, returns top 10."
}

Q: Which departments have the highest reorder rate?

{
  "sql": "SELECT d.department, ROUND(AVG(op.reordered) * 100, 2) AS reorder_rate_pct FROM order_products_all op JOIN products p ON op.product_id = p.product_id JOIN departments d ON p.department_id = d.department_id GROUP BY d.department ORDER BY reorder_rate_pct DESC",
  "chart_type": "bar",
  "explanation": "Joins order_products → products → departments. AVG(reordered) on a 0/1 column gives the reorder rate. Multiplied by 100 for percentage."
}

Q: What is the distribution of orders by hour of day?

{
  "sql": "SELECT order_hour_of_day AS hour, COUNT(*) AS num_orders FROM orders GROUP BY hour ORDER BY hour",
  "chart_type": "line",
  "explanation": "Groups orders by hour. Line chart shows the intraday shopping pattern."
}

Q: Show me the top 5 aisles by reorder rate and their average basket position

{
  "sql": "SELECT a.aisle, ROUND(AVG(op.reordered) * 100, 2) AS reorder_rate_pct, ROUND(AVG(op.add_to_cart_order), 2) AS avg_basket_position FROM order_products_all op JOIN products p ON op.product_id = p.product_id JOIN aisles a ON p.aisle_id = a.aisle_id GROUP BY a.aisle ORDER BY reorder_rate_pct DESC LIMIT 5",
  "chart_type": "bar",
  "explanation": "Three-table join: order_products → products → aisles. Returns reorder rate and average add-to-cart position per aisle."
}

Q: How many orders does the average user place?

{
  "sql": "SELECT ROUND(AVG(order_count), 2) AS avg_orders_per_user FROM (SELECT user_id, COUNT(*) AS order_count FROM orders GROUP BY user_id)",
  "chart_type": "table",
  "explanation": "Subquery counts orders per user, outer query averages that count."
}

Q: What percentage of products in each department are organic?

{
  "sql": "SELECT d.department, COUNT(*) AS total_products, SUM(CASE WHEN LOWER(p.product_name) LIKE '%organic%' THEN 1 ELSE 0 END) AS organic_count, ROUND(100.0 * SUM(CASE WHEN LOWER(p.product_name) LIKE '%organic%' THEN 1 ELSE 0 END) / COUNT(*), 2) AS organic_pct FROM products p JOIN departments d ON p.department_id = d.department_id GROUP BY d.department ORDER BY organic_pct DESC",
  "chart_type": "bar",
  "explanation": "Uses CASE WHEN to flag organic products by name pattern, then computes percentage per department."
}

Q: Show order volume by day of week

{
  "sql": "SELECT CASE order_dow WHEN 0 THEN 'Saturday' WHEN 1 THEN 'Sunday' WHEN 2 THEN 'Monday' WHEN 3 THEN 'Tuesday' WHEN 4 THEN 'Wednesday' WHEN 5 THEN 'Thursday' WHEN 6 THEN 'Friday' END AS day_name, order_dow, COUNT(*) AS num_orders FROM orders GROUP BY order_dow, day_name ORDER BY order_dow",
  "chart_type": "bar",
  "explanation": "Maps numeric order_dow (0=Saturday) to day names and counts orders per day."
}

Q: What is the average days between orders, excluding first-time orders?

{
  "sql": "SELECT ROUND(AVG(days_since_prior_order), 2) AS avg_days_between_orders, COUNT(*) AS order_count FROM orders WHERE days_since_prior_order IS NOT NULL",
  "chart_type": "table",
  "explanation": "Filters out NULLs (first orders) before averaging days_since_prior_order."
}
"""

# ---------------------------------------------------------------------------
# Chart type selection guide (also in the system prompt)
# ---------------------------------------------------------------------------

CHART_GUIDE = """
=== CHART TYPE SELECTION RULES ===
- "bar": Use for comparing a metric across discrete categories.
    - Q: "Top 10 products by order count" → bar chart (product vs. count)
    - Q: "Reorder rate by department" → bar chart (department vs. rate)
- "line": Use for showing a trend over a continuous or ordered sequence.
    - Q: "Order volume by hour" → line chart (hour vs. volume)
- "pie": Use for showing part-of-whole composition. Only for a few categories (≤8).
    - Q: "What percentage of orders are reorders?" → pie chart (reordered vs. not)
- "histogram": Use for showing the distribution of a single continuous variable.
    - Q: "What is the distribution of days between orders?" → histogram (days vs. frequency)
- "table": Use for single aggregated values, multi-column results without a clear visual mapping, or when the user explicitly asks for a list/table.
"""


def build_system_prompt(schema_text: str, sample_rows_text: str) -> str:
    return f"""You are an expert SQL analyst working with a DuckDB database containing
Instacart grocery shopping data. Your job is to convert natural language questions
into correct DuckDB SQL queries.

{schema_text}

{sample_rows_text}

{FEW_SHOT_EXAMPLES}

{CHART_GUIDE}

=== OUTPUT FORMAT ===
First, show your reasoning process using numbered steps. Then respond with ONLY a valid JSON object:

Step-by-step reasoning:
1. [Identify what tables and columns are needed]
2. [Determine what joins are required]
3. [Explain the aggregation/filtering logic]
4. [Specify the chart type and why]

Then provide the JSON (no markdown fences):
{{
  "sql": "<DuckDB SQL query>",
  "chart_type": "<bar|line|pie|scatter|histogram|table>",
  "explanation": "<one or two sentences explaining what the query computes>"
}}

Rules:
- Always use DuckDB SQL syntax.
- Always add LIMIT (≤ 100) unless the user asks for all data or a count.
- Never use pandas or Python — only SQL.
- Use order_products_all for combined prior+train data.
- Handle NULLs in days_since_prior_order explicitly.
- Prefer readable column aliases.
"""


def build_retry_prompt(original_question: str, failed_sql: str, error: str) -> str:
    return f"""The previous SQL query failed with an error. Please fix it.

Original question: {original_question}

Failed SQL:
{failed_sql}

Error message:
{error}

Respond with the corrected JSON object only (no markdown, no explanation outside JSON):
{{
  "sql": "<corrected SQL>",
  "chart_type": "<chart type>",
  "explanation": "<explanation>"
}}
"""


class NLToSQL:
    def __init__(self, schema_text: str, sample_rows_text: str):
        self.system_prompt = build_system_prompt(schema_text, sample_rows_text)

    def _call_ollama(self, messages: list[dict]) -> dict:
        """
        Call the Ollama model and parse the JSON response.
        Returns dict with keys: sql, chart_type, explanation, reasoning_steps
        """
        # Prepare messages for Ollama - include system prompt and conversation
        ollama_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        logger.info(f"📨 System prompt being sent to LLM (first 300 chars):\n{self.system_prompt[:300]}")
        logger.info(f"📨 System prompt includes 'Step-by-step reasoning': {'Step-by-step reasoning' in self.system_prompt}")
        
        # Add conversation history
        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else "user"
            content = msg.get("content") if isinstance(msg, dict) else str(msg)
            ollama_messages.append({"role": role, "content": content})
            if role == "user":
                logger.info(f"👤 User message: {content[:100]}")

        # Retry logic with exponential backoff for temporary errors
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = ollama_client.chat(
                    ollama_messages,
                    temperature=0.1,  # Low temperature for deterministic SQL
                    top_p=0.9
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Ollama API failed after {max_retries} attempts: {e}")
                    raise
                else:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Ollama API temporary error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue

        raw_response = response.get("response", "")
        
        logger.info(f"🔍 RAW RESPONSE FROM LLM:")
        logger.info(f"   Length: {len(raw_response)} chars")
        logger.info(f"   First 500 chars:\n{raw_response[:500]}")
        logger.info(f"   Last 500 chars:\n{raw_response[-500:]}")
        
        # Extract reasoning steps from the response
        reasoning_steps = self._extract_reasoning_steps(raw_response)
        
        # Clean the response - SQLCoder might return HTML-like tags or special tokens
        cleaned_response = raw_response.strip()
        
        # Remove common LLM artifacts and special tokens
        cleaned_response = cleaned_response.replace("<s>", "").replace("</s>", "")
        cleaned_response = cleaned_response.replace("[INST]", "").replace("[/INST]", "")
        cleaned_response = cleaned_response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        
        # SQLCoder often returns complete JSON with sql, chart_type, explanation
        try:
            # First, try to parse the entire response as JSON
            try:
                result = json.loads(cleaned_response)
                # Validate that we have the required SQL field
                if "sql" not in result:
                    raise json.JSONDecodeError("Missing sql field", "", 0)
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try to extract JSON from markdown
                if "```json" in cleaned_response:
                    json_part = cleaned_response.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_part)
                elif "```" in cleaned_response:
                    # Might be just SQL code without JSON wrapper
                    sql_code = cleaned_response.split("```")[1].strip()
                    if sql_code.lower().startswith("sql"):
                        sql_code = sql_code[3:].strip()
                    result = {"sql": sql_code, "chart_type": "table", "explanation": ""}
                else:
                    # Try to find JSON object in the response
                    json_start = cleaned_response.find("{")
                    json_end = cleaned_response.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        json_part = cleaned_response[json_start:json_end]
                        result = json.loads(json_part)
                    else:
                        # Assume it's raw SQL
                        sql_code = cleaned_response
                        if "SELECT" in sql_code.upper():
                            select_pos = sql_code.upper().find("SELECT")
                            sql_code = sql_code[select_pos:]
                        elif "WITH" in sql_code.upper():
                            with_pos = sql_code.upper().find("WITH")
                            sql_code = sql_code[with_pos:]
                        result = {"sql": sql_code.strip(), "chart_type": "table", "explanation": ""}
        
        except (json.JSONDecodeError, IndexError):
            # If JSON parsing fails, treat the cleaned response as SQL
            logger.warning(f"Failed to parse Ollama response as JSON, treating as raw SQL: {cleaned_response}")
            # Extract just the SQL part if possible
            sql_code = cleaned_response
            if "SELECT" in sql_code.upper():
                select_pos = sql_code.upper().find("SELECT")
                sql_code = sql_code[select_pos:]
            result = {"sql": sql_code.strip(), "chart_type": "table", "explanation": ""}

        # Validate required fields
        if "sql" not in result:
            raise ValueError(f"Ollama response missing 'sql' field: {result}")

        result.setdefault("chart_type", "table")
        result.setdefault("explanation", "")
        result["reasoning_steps"] = reasoning_steps  # Add reasoning steps
        
        # Clean the SQL to fix any formatting issues from the model
        if "sql" in result:
            result["sql"] = self._clean_sql(result["sql"])
        
        return result
    
    def _clean_sql(self, sql_code: str) -> str:
        """
        Clean malformed SQL from the model response.
        Removes trailing commas, extra quotes, and other common issues.
        """
        # Remove trailing commas that cause syntax errors
        sql_code = sql_code.rstrip(",")
        sql_code = sql_code.rstrip(", ")
        
        # Remove any trailing semicolons followed by garbage
        if ";" in sql_code:
            # Take only the part before the first semicolon
            sql_code = sql_code.split(";")[0]
        
        # Remove any extra quotes or malformed endings
        sql_code = sql_code.rstrip('"')
        sql_code = sql_code.rstrip("'")
        sql_code = sql_code.rstrip()
        
        # Fix common pattern: "... LIMIT 10", " -> remove the trailing quote and comma
        if sql_code.endswith('",'):
            sql_code = sql_code[:-2]
        elif sql_code.endswith('"'):
            sql_code = sql_code[:-1]
        
        # Remove any non-SQL content at the end
        lines = sql_code.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('#'):
                cleaned_lines.append(line)
        
        sql_code = ' '.join(cleaned_lines)
        
        # Final trim
        return sql_code.strip()

    def _extract_reasoning_steps(self, response: str) -> list[dict]:
        """
        Extract reasoning steps from the LLM response.
        Looks for numbered steps (1., 2., 3., etc.) before the JSON.
        Returns list of dicts: [{step: 1, description: "..."}, ...]
        """
        reasoning_steps = []
        
        # Find the JSON part (it starts with { and ends with })
        json_start = response.find("{")
        if json_start == -1:
            logger.warning("❌ No JSON found in response for reasoning extraction")
            logger.warning(f"Full response: {response[:500]}")
            return reasoning_steps  # No JSON found, no reasoning to extract
        
        # Extract text before JSON
        pre_json_text = response[:json_start].strip()
        logger.info(f"📄 Full pre-JSON text for reasoning extraction:\n{pre_json_text}")
        logger.info(f"📄 Pre-JSON text length: {len(pre_json_text)} chars")
        
        # Look for numbered steps
        lines = pre_json_text.split('\n')
        logger.info(f"📄 Total lines found: {len(lines)}")
        
        for line_num, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            if not line:
                logger.debug(f"   Line {line_num}: [EMPTY]")
                continue
            
            logger.debug(f"   Line {line_num}: '{line}'")
            
            # Check if line starts with a number followed by a period or parenthesis
            if len(line) > 2 and line[0].isdigit():
                logger.debug(f"      ✓ Line starts with digit: {line[0]}")
                # Extract step number and description
                for i, char in enumerate(line):
                    if char in '.):' and i > 0 and line[:i].isdigit():
                        try:
                            step_num = int(line[:i])
                            description = line[i+1:].strip()
                            logger.debug(f"      ✓ Matched step {step_num}: {description[:50]}")
                            if description:
                                reasoning_steps.append({
                                    "step": step_num,
                                    "description": description
                                })
                                logger.info(f"✅ Added step {step_num}")
                            break
                        except ValueError as e:
                            logger.debug(f"      ✗ ValueError parsing: {e}")
            elif "step" in line.lower():
                logger.debug(f"      ✓ Contains 'step' keyword")
                # Handle "Step 1: ..." format
                for i, char in enumerate(line):
                    if char == ':' and i > 0:
                        try:
                            step_num = int(''.join(filter(str.isdigit, line[:i])))
                            description = line[i+1:].strip()
                            logger.debug(f"      ✓ Matched step {step_num}: {description[:50]}")
                            if description:
                                reasoning_steps.append({
                                    "step": step_num,
                                    "description": description
                                })
                                logger.info(f"✅ Added step {step_num}")
                            break
                        except ValueError as e:
                            logger.debug(f"      ✗ ValueError parsing: {e}")
        
        logger.info(f"🎯 FINAL RESULT: Extracted {len(reasoning_steps)} reasoning steps")
        if reasoning_steps:
            logger.info(f"   Steps: {reasoning_steps}")
        else:
            logger.warning(f"❌ NO REASONING STEPS EXTRACTED")
        
        return reasoning_steps

    def translate(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Translate a natural language question to SQL.

        conversation_history: list of {role: "user"|"assistant", content: str}
        This enables follow-up questions like "now filter that to organics".
        """
        # Emit start event
        streaming_emitter.start_step(
            step_number=1,
            step_name="nlp_translation",
            description="Translating Natural Language to SQL"
        )
        
        try:
            # Build message list: history + current question
            messages = list(conversation_history or [])
            messages.append({"role": "user", "content": question})

            logger.info(f"Translating: {question!r}")
            result = self._call_ollama(messages)
            logger.info(f"Generated SQL: {result['sql'][:200]}...")
            
            # Emit completion event with SQL details
            streaming_emitter.complete_step(
                step_number=1,
                step_name="nlp_translation",
                description="Translating Natural Language to SQL",
                details={
                    "sql": result["sql"][:100] + "..." if len(result["sql"]) > 100 else result["sql"],
                    "chart_type": result.get("chart_type", "table"),
                    "explanation": result.get("explanation", ""),
                    "reasoning_steps": len(result.get("reasoning_steps", []))
                }
            )
            
            return result
        except Exception as e:
            # Emit error event
            streaming_emitter.error_step(
                step_number=1,
                step_name="nlp_translation",
                description="Translating Natural Language to SQL",
                error_msg=str(e)
            )
            raise

    def retry_with_error(
        self,
        original_question: str,
        failed_sql: str,
        error_message: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Self-correction: re-prompt Gemini with the error message so it
        can fix the SQL.
        """
        # Emit retry start event
        streaming_emitter.start_step(
            step_number=1,
            step_name="nlp_translation",
            description="Re-translating with Error Correction"
        )
        
        try:
            logger.info(f"Retrying after error: {error_message[:100]}")

            retry_content = build_retry_prompt(original_question, failed_sql, error_message)
            messages = list(conversation_history or [])
            messages.append({"role": "user", "content": retry_content})

            result = self._call_ollama(messages)
            
            # Emit completion event for retry
            streaming_emitter.complete_step(
                step_number=1,
                step_name="nlp_translation",
                description="Re-translating with Error Correction",
                details={
                    "sql": result["sql"][:100] + "..." if len(result["sql"]) > 100 else result["sql"],
                    "chart_type": result.get("chart_type", "table"),
                    "status": "recovered"
                }
            )
            
            return result
        except Exception as e:
            # Emit error event for retry
            streaming_emitter.error_step(
                step_number=1,
                step_name="nlp_translation",
                description="Re-translating with Error Correction",
                error_msg=str(e)
            )
            raise

"""Agent 2: Text-to-SQL & Dataset Profiler.

Takes the retrieved schema/document context from Agent 1 and asks a Groq-hosted
LLM to write a single, safe, read-only SQLite SELECT query. The query is
executed against the local database and basic Pandas profiling is run on the
result.

Design notes (why this file looks the way it does):

* Groq periodically retires/renames models (this is the #1 cause of the
  "model not found" error this project was hitting). Instead of hard-coding
  one model name, we try a small ordered fallback chain and only give up if
  every candidate fails.
* Groq's `openai/gpt-oss-*` and `qwen/qwen3.8-27b` models support strict
  JSON-schema structured outputs, which removes the need for fragile
  string/regex parsing of the model's response. For any other model we still
  ask for JSON object mode and fall back to a defensive text extractor.
* All SQL is validated as a read-only SELECT/CTE before it ever touches
  sqlite3 - the model is never trusted to self-police.
"""

import os
import re
import json
import time
import sqlite3
import pandas as pd
from pydantic import BaseModel, ValidationError
from groq import Groq, APIStatusError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()


class SQLResponse(BaseModel):
    """Expected shape of the LLM's JSON response. Used both as the strict
    JSON-schema contract sent to Groq and to validate whatever comes back,
    regardless of which model in the fallback chain actually answered.
    """
    sql: str

# Tried in order; GROQ_MODEL (if set in the environment/.env) is always tried
# first. Keeping this a list (not a single hardcoded string) is what makes
# the agent resilient to Groq deprecating/renaming a model.
_DEFAULT_FALLBACK_CHAIN = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "groq/compound",
]

# Models known to support strict JSON-schema structured outputs on Groq as of
# this writing. Everything else falls back to plain JSON-object mode.
_JSON_SCHEMA_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}

_SQL_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sql_response",
        "schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single valid, executable, read-only SQLite SELECT statement.",
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_BLOCKED_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter",
    "attach", "detach", "pragma", "vacuum", "replace", "create",
)


class SQLSafetyError(RuntimeError):
    """Raised when the model returns something that isn't a safe SELECT query."""


def _model_chain():
    chain = []
    override = os.getenv("GROQ_MODEL")
    if override:
        chain.append(override)
    for m in _DEFAULT_FALLBACK_CHAIN:
        if m not in chain:
            chain.append(m)
    return chain


def _extract_sql(raw_content: str) -> str:
    """Best-effort extraction of a SQL string out of raw LLM text.

    Handles fenced code blocks, ``<think>...</think>`` reasoning traces that
    some Groq models emit, a JSON payload, or bare SQL text.
    """
    if not raw_content:
        return ""

    text = raw_content.strip()

    # Strip <think>...</think> reasoning traces some models emit even when
    # not asked to (this was previously attempted with a broken empty regex).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Strip markdown code fences (```json ... ``` or ```sql ... ```).
    fence_match = re.search(r"```(?:json|sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to parse a JSON object containing a "sql" key, validated through a
    # Pydantic model so a malformed/partial payload is rejected cleanly
    # rather than silently reading a wrong field.
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            parsed = SQLResponse.model_validate(payload)
            return parsed.sql.strip()
        except (json.JSONDecodeError, TypeError, ValidationError):
            pass

    # Fall back to treating the remaining text as raw SQL.
    return text.strip()


def _validate_select_only(sql_query: str) -> None:
    """Guard against anything other than a read-only SELECT/CTE statement."""
    if not sql_query or not sql_query.strip():
        raise SQLSafetyError("The model did not return a SQL query.")

    stripped = sql_query.strip().lower().lstrip("(")
    if not (stripped.startswith("select") or stripped.startswith("with")):
        raise SQLSafetyError(
            f"Refusing to execute a non-SELECT statement for safety: {sql_query[:120]!r}"
        )
    for keyword in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", stripped):
            raise SQLSafetyError(
                f"Refusing to execute a query containing the blocked keyword '{keyword}': {sql_query[:120]!r}"
            )
    if ";" in stripped.rstrip(";"):
        raise SQLSafetyError("Refusing to execute multiple statements in a single query.")


def _call_groq(client: Groq, model: str, system_prompt: str, user_query: str):
    use_json_schema = model in _JSON_SCHEMA_MODELS
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0,
    )
    kwargs["response_format"] = _SQL_JSON_SCHEMA if use_json_schema else {"type": "json_object"}

    start = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response, elapsed_ms


def generate_sql_and_profile(user_query: str, context: dict, db_path: str = "data/chinook.db"):
    """Runs Agent 2 end to end.

    Returns ``(sql_query, dataframe, profile, meta)`` where ``meta`` carries an
    execution trace (which model answered, per-attempt latency, and any
    fallbacks that were needed) so the UI can show what actually happened
    instead of a single opaque error.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file in the project root containing:\n"
            "GROQ_API_KEY=gsk_..."
        )
    client = Groq(api_key=groq_api_key)

    system_prompt = f"""You are a Text-to-SQL AI Agent for a SQLite database. Given the natural language prompt and the retrieved schema/document context, return ONLY a JSON object with a single key "sql" containing one valid, executable, read-only SQLite SELECT query.

Rules:
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, or PRAGMA statements.
- Only reference tables and columns that appear in the schema context below.
- Return exactly one statement, no explanations, no markdown.

Schema Context: {context.get('schema_context', [])}
Document Context: {context.get('document_context', [])}
Relationships (table -> table via foreign key): {context.get('graph_edges', [])}
"""

    steps = list(context.get("trace", []))
    errors = []
    response = None
    used_model = None

    for model in _model_chain():
        try:
            response, elapsed_ms = _call_groq(client, model, system_prompt, user_query)
            used_model = model
            steps.append({"agent": "text_to_sql", "model": model, "status": "ok", "latency_ms": round(elapsed_ms, 1)})
            break
        except APIStatusError as e:
            # Covers NotFoundError (model decommissioned/typo'd) and
            # BadRequestError (e.g. this model rejects json_schema mode) -
            # both mean "try the next model in the chain", not "give up".
            steps.append({"agent": "text_to_sql", "model": model, "status": "failed", "error": str(e)[:200]})
            errors.append(f"{model}: {e}")
            continue
        except APIConnectionError as e:
            steps.append({"agent": "text_to_sql", "model": model, "status": "connection_error", "error": str(e)[:200]})
            errors.append(f"{model}: connection error - {e}")
            continue

    if response is None:
        raise RuntimeError(
            "Every configured Groq model failed to answer. Tried: "
            + ", ".join(_model_chain())
            + ". Run `python scripts/test_groq.py` to debug which models your API key can access. "
            + "Errors: " + " | ".join(errors)
        )

    message = response.choices[0].message
    raw_content = message.content or ""
    sql_query = _extract_sql(raw_content)
    _validate_select_only(sql_query)

    exec_start = time.perf_counter()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()
    exec_ms = (time.perf_counter() - exec_start) * 1000
    steps.append({"agent": "sql_execution", "status": "ok", "latency_ms": round(exec_ms, 1), "rows": len(df)})

    profile = {
        "row_count": len(df),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
    }

    meta = {"model_used": used_model, "steps": steps}
    return sql_query, df, profile, meta

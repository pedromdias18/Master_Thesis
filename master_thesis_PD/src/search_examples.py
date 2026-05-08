"""
search_examples.py — Capture illustrative search results for Chapter 4.

Runs a fixed list of queries against the three search modes (Semantic, Keyword,
AI-Parsed) and writes the full results (with scores and metadata) to a JSON
file for use in the thesis.

Usage (from project root):
    python src/search_examples.py

Outputs:
    data/db/search_examples.json

Prerequisites:
    The database and FAISS index must already exist (run build_search_db.py
    first). ANTHROPIC_API_KEY must be set in .env (only needed for the
    AI-Parsed mode — the other two modes run locally).
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ── Paths (match chatbot.py) ─────────────────────────────────────────────────

DB_DIR = Path('data/db')
DB_PATH = DB_DIR / 'companies.db'
FAISS_PATH = DB_DIR / 'company_search.faiss'
IDS_PATH = DB_DIR / 'company_ids.json'
OUTPUT_PATH = DB_DIR / 'search_examples.json'
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Query sets ───────────────────────────────────────────────────────────────

# Each query is (label, text). Label is a short identifier used in the JSON.
# Semantic queries mix within-language and cross-language cases to showcase
# that the multilingual embedding model maps different surface forms of the
# same concept into nearby regions of vector space.
SEMANTIC_QUERIES = [
    ("shipping",            "shipping company"),
    ("construction",        "construction"),
    ("hotel",               "hotel and accommodation"),
    ("car_wash",            "car wash"),
    ("bakery",              "bakery"),
    ("legal_services",      "legal services"),
    ("software",            "software development"),
    ("agriculture",         "agriculture and farming"),
    ("renewable_energy",    "renewable energy"),
    ("restaurant",          "restaurant"),
]

# Keyword queries hit FTS5 on company-name columns and registration-number
# columns only (see search_fts below). The registration number case is the
# precise-identifier lookup requested for the thesis.
KEYWORD_QUERIES = [
    ("name_hotel",              "hotel"),
    ("name_invest",             "invest"),
    ("name_banco",              "banco"),
    ("regnum_946158070",        "946158070"),
]

# AI-Parsed queries are restricted to list-returning requests: filtering and
# ordering over the full 150k records. Aggregate/count queries are excluded
# because the system always returns companies, not scalar answers.
AI_PARSED_QUERIES = [
    ("top_norway_capital",      "top 20 Norwegian companies with most share capital"),
    ("top_norway_employees",    "Norwegian companies with the most employees"),
    ("oldest_honduras",         "10 oldest registered companies in Honduras"),
    ("newest_myanmar",          "20 newest companies in Myanmar"),
    ("honduras_by_capital",     "top 15 Honduran companies by share capital"),
]

# How many results to keep per query.
SEMANTIC_TOP_K       = 5
KEYWORD_TOP_K        = 5
AI_PARSED_DEFAULT_K  = 20   # upper bound; the LLM's own limit usually wins


# ── LLM call (copied from chatbot.py) ────────────────────────────────────────

CLAUDE_MODEL = "claude-opus-4-5-20251101"
CLAUDE_MAX_TOKENS = 1024

_claude_client = None

def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        _claude_client = anthropic.Anthropic(api_key=api_key)
    return _claude_client


def call_llm(prompt):
    try:
        response = _get_claude_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return None


# ── Query parser (copied from chatbot.py) ────────────────────────────────────

def parse_query_with_llm(user_query):
    prompt = f"""You are a search query parser for a company database with 150,000 companies from Myanmar (MM), Norway (NO), and Honduras (HN).

The database has these columns:
- company_name_legal_name, company_name_trade_name, company_name_short_name
- unique_identifier_registration_number
- company_type, status (ACT=Active, INA=Inactive, BAN=Bankrupt, LIQ=Liquidation, CEA=Ceased)
- country_registration (MM, NO, HN)
- business_address_city, business_address_street
- num_employees (integer, only Norway has data)
- share_capital_amount (numeric), share_capital_currency
- registration_date (YYYY-MM-DD)
- description (text about company activities)

Parse the user query and determine the best search strategy.

User query: "{user_query}"

Two strategies:
1. "semantic" — for finding companies by what they do. Use when the user searches by topic/activity/meaning.
2. "sql" — for analytical queries with sorting, ranking, counting, or filtering by numeric fields.
   Use when the user asks for "top N", "most", "largest", "oldest", "newest", ordering, etc.

Return ONLY valid JSON:
{{
  "strategy": "semantic" or "sql",
  "search_text": "terms for semantic search (only if strategy=semantic)",
  "filters": {{
    "country_registration": "MM/NO/HN (if mentioned)",
    "status": "ACT/INA/BAN/LIQ/CEA (if mentioned)"
  }},
  "order_by": "column_name (only if strategy=sql)",
  "order_dir": "DESC or ASC (only if strategy=sql)",
  "limit": number (default 10),
  "where_extra": "additional SQL WHERE conditions if needed, e.g. share_capital_amount > 0 AND share_capital_amount != ''"
}}

Only include fields that are relevant. No markdown, no explanation."""

    result = call_llm(prompt)
    if result:
        try:
            if result.startswith('```'):
                result = result.split('\n', 1)[1]
                result = result.rsplit('```', 1)[0]
                result = result.strip()
            if '<think>' in result:
                result = result.split('</think>')[-1].strip()
            return json.loads(result)
        except json.JSONDecodeError:
            print(f"  [parser] JSON decode failed, raw response: {result[:200]}")
            return None
    return None


# ── Resource loaders (module-level singletons) ───────────────────────────────

_faiss_index = None
_faiss_ids = None
_embed_model = None


def load_faiss_index():
    global _faiss_index, _faiss_ids
    if _faiss_index is None:
        import faiss
        _faiss_index = faiss.read_index(str(FAISS_PATH))
        with open(IDS_PATH, 'r', encoding='utf-8') as f:
            _faiss_ids = json.load(f)
    return _faiss_index, _faiss_ids


def load_embedding_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


# ── Search functions (logic matches chatbot.py) ──────────────────────────────

def search_semantic(query_text, filters=None, top_k=10):
    index, ids = load_faiss_index()
    model = load_embedding_model()

    query_embedding = model.encode([query_text], normalize_embeddings=True).astype('float32')
    search_k = top_k * 5 if filters else top_k
    scores, indices = index.search(query_embedding, min(search_k, index.ntotal))

    matching_ids = [ids[i] for i in indices[0] if i < len(ids)]
    matching_scores = scores[0][:len(matching_ids)]

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    results = []
    for company_id, score in zip(matching_ids, matching_scores):
        sql = "SELECT * FROM companies WHERE id = ?"
        params = [company_id]
        if filters:
            for col, val in filters.items():
                sql += f" AND {col} = ?"
                params.append(val)
        row = conn.execute(sql, params).fetchone()
        if row:
            result = dict(row)
            result['_score'] = float(score)
            results.append(result)
            if len(results) >= top_k:
                break
    conn.close()
    return results


def search_fts(text_query, filters=None, limit=10):
    """FTS5 keyword search restricted to name columns and registration numbers."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT c.* FROM companies c
        JOIN companies_fts ON c.id = companies_fts.id
        WHERE companies_fts MATCH ?
    """
    fts_query = (
        "{company_name_legal_name company_name_trade_name company_name_short_name "
        "company_name_local_language_name unique_identifier_registration_number "
        f"unique_identifier_prior_registration_number}}: {text_query}"
    )
    params = [fts_query]

    if filters:
        for col, val in filters.items():
            sql += f" AND c.{col} = ?"
            params.append(val)
    sql += " ORDER BY companies_fts.rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def search_sql(filters=None, order_by=None, order_dir="DESC", limit=10, where_extra=None):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    sql = "SELECT * FROM companies WHERE 1=1"
    params = []

    if filters:
        for col, val in filters.items():
            sql += f" AND {col} = ?"
            params.append(val)

    if where_extra:
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_. ><=!''\"AND OR NOT "
        )
        if all(c in allowed_chars for c in where_extra):
            sql += f" AND {where_extra}"

    if order_by:
        valid_columns = [
            'share_capital_amount', 'num_employees', 'registration_date',
            'company_name_legal_name', 'business_address_city', 'status',
            'company_type', 'country_registration',
        ]
        if order_by in valid_columns:
            if order_by in ('share_capital_amount', 'num_employees'):
                sql += f" ORDER BY CAST({order_by} AS REAL) {order_dir}"
            else:
                sql += f" ORDER BY {order_by} {order_dir}"

    sql += " LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


# ── Row compacting ───────────────────────────────────────────────────────────

# Keep only the fields relevant for the thesis tables. The full row has ~24
# columns; most are noise for illustration.
COMPACT_FIELDS = [
    'id',
    'company_name_legal_name',
    'company_name_local_language_name',
    'unique_identifier_registration_number',
    'country_registration',
    'business_address_city',
    'company_type',
    'status',
    'num_employees',
    'share_capital_amount',
    'share_capital_currency',
    'registration_date',
    'description',
]


def compact(row, rank):
    out = {'rank': rank}
    for k in COMPACT_FIELDS:
        if k in row and row[k] not in (None, ''):
            out[k] = row[k]
    if '_score' in row:
        out['score'] = row['_score']
    # Fallback display name: legal → short → trade
    out['display_name'] = (
        row.get('company_name_legal_name')
        or row.get('company_name_short_name')
        or row.get('company_name_trade_name')
        or None
    )
    return out


# ── Main driver ──────────────────────────────────────────────────────────────

def run_semantic_suite():
    print("\n=== Semantic search ===")
    out = []
    for label, text in SEMANTIC_QUERIES:
        print(f"  {label}: {text!r}")
        results = search_semantic(text, top_k=SEMANTIC_TOP_K)
        out.append({
            "label": label,
            "query": text,
            "num_results": len(results),
            "results": [compact(r, i + 1) for i, r in enumerate(results)],
        })
    return out


def run_keyword_suite():
    print("\n=== Keyword search ===")
    out = []
    for label, text in KEYWORD_QUERIES:
        print(f"  {label}: {text!r}")
        results = search_fts(text, limit=KEYWORD_TOP_K)
        out.append({
            "label": label,
            "query": text,
            "num_results": len(results),
            "results": [compact(r, i + 1) for i, r in enumerate(results)],
        })
    return out


def run_ai_parsed_suite():
    print("\n=== AI-Parsed search ===")
    out = []
    for label, text in AI_PARSED_QUERIES:
        print(f"  {label}: {text!r}")
        parsed = parse_query_with_llm(text)
        if parsed is None:
            out.append({
                "label": label,
                "query": text,
                "error": "LLM parse failed",
                "results": [],
            })
            continue

        strategy = parsed.get("strategy", "semantic")
        filters = {k: v for k, v in parsed.get("filters", {}).items() if v}

        if strategy == "sql":
            order_by  = parsed.get("order_by", "")
            order_dir = parsed.get("order_dir", "DESC")
            limit     = min(parsed.get("limit", AI_PARSED_DEFAULT_K), 200)
            where_extra = parsed.get("where_extra", "")
            results = search_sql(
                filters=filters if filters else None,
                order_by=order_by,
                order_dir=order_dir,
                limit=limit,
                where_extra=where_extra if where_extra else None,
            )
        else:
            search_text = parsed.get("search_text", text)
            results = search_semantic(
                search_text,
                filters=filters if filters else None,
                top_k=min(parsed.get("limit", AI_PARSED_DEFAULT_K), 20),
            )

        out.append({
            "label": label,
            "query": text,
            "llm_parsed": parsed,
            "num_results": len(results),
            "results": [compact(r, i + 1) for i, r in enumerate(results)],
        })
    return out


def main():
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        print("Run `python src/build_search_db.py` first.")
        sys.exit(1)
    if not FAISS_PATH.exists():
        print(f"ERROR: FAISS index not found at {FAISS_PATH}")
        sys.exit(1)

    report = {
        "generated_at": datetime.now().isoformat(timespec='seconds'),
        "db_path":      str(DB_PATH),
        "faiss_path":   str(FAISS_PATH),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "semantic":   run_semantic_suite(),
        "keyword":    run_keyword_suite(),
        "ai_parsed":  run_ai_parsed_suite(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote {OUTPUT_PATH}")
    print(f"  Semantic  queries: {len(report['semantic'])}")
    print(f"  Keyword   queries: {len(report['keyword'])}")
    print(f"  AI-parsed queries: {len(report['ai_parsed'])}")


if __name__ == "__main__":
    main()

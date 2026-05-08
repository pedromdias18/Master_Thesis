"""
Search Database Builder: Create the unified search infrastructure.

This script:
1. Merges all normalized country CSVs into one unified table
2. Creates a SQLite database for structured queries (filter, sort)
3. Builds a FAISS vector index on text fields for semantic search
4. Provides a search API combining both

Usage (from project root):
    python src/build_search_db.py

Prerequisites:
    pip install sentence-transformers faiss-cpu
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

# All possible columns across all countries (union of all CSVs)
# When a country doesn't have a column, it gets empty string
ALL_COLUMNS = [
    "company_name.legal_name",
    "company_name.trade_name",
    "company_name.short_name",
    "company_name.local_language_name",
    "unique_identifier.registration_number",
    "unique_identifier.prior_registration_number",
    "parent_company.registration_number",
    "parent_company.company_name",
    "company_type",
    "status",
    "country.registration",
    "country.business_address",
    "country.postal_address",
    "business_address.street",
    "business_address.city",
    "postal_address.street",
    "postal_address.city",
    "num_employees",
    "share_capital.amount",
    "share_capital.currency",
    "share_capital.fully_paid",
    "registration_date",
    "description",
]

# Text fields used for building the semantic search index
# These are concatenated into a single "search_text" for each record
TEXT_FIELDS_FOR_SEARCH = [
    "company_name.legal_name",
    "company_name.trade_name",
    "company_name.short_name",
    "company_name.local_language_name",
    "description",
    "business_address.city",
]

# Embedding model — small, fast, multilingual
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ── Step 1: Merge CSVs ──────────────────────────────────────────────────────

def merge_csvs(normalized_dir):
    """
    Load all normalized CSVs and merge into one DataFrame with uniform columns.
    """
    files = {
        "Myanmar": "myanmar_normalized.csv",
        "Norway": "norway_normalized.csv",
        "Honduras": "honduras_normalized.csv",
    }

    dfs = []
    for country_name, filename in files.items():
        path = normalized_dir / filename
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {country_name}")
            continue

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        print(f"  Loaded {country_name}: {len(df):,} records, {len(df.columns)} columns")

        # Add missing columns with empty strings
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        # Keep only the canonical columns, in order
        df = df[ALL_COLUMNS]
        dfs.append(df)

    if not dfs:
        print("ERROR: No CSV files found!")
        return None

    merged = pd.concat(dfs, ignore_index=True)
    # Add a unique ID column
    merged.insert(0, 'id', range(1, len(merged) + 1))

    print(f"\n  Merged: {len(merged):,} total records")
    return merged


# ── Step 2: Create SQLite Database ───────────────────────────────────────────

def create_sqlite_db(df, db_path):
    """
    Create a SQLite database from the merged DataFrame.
    Includes indexes on commonly filtered columns.
    """
    conn = sqlite3.connect(db_path)

    # Replace dots in column names with underscores for SQL compatibility
    col_map = {col: col.replace('.', '_') for col in df.columns}
    df_sql = df.rename(columns=col_map)

    # Write to SQLite
    df_sql.to_sql('companies', conn, if_exists='replace', index=False)

    # Create indexes for common filters
    cursor = conn.cursor()
    index_columns = [
        'country_registration',
        'status',
        'company_type',
        'business_address_city',
        'company_name_legal_name',
        'registration_date',
    ]
    for col in index_columns:
        try:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{col} ON companies ({col})')
        except Exception as e:
            print(f"  Warning: Could not create index on {col}: {e}")

    # Create FTS5 virtual table for full-text search
    cursor.execute('DROP TABLE IF EXISTS companies_fts')
    cursor.execute('''
        CREATE VIRTUAL TABLE companies_fts USING fts5(
            id,
            company_name_legal_name,
            company_name_trade_name,
            company_name_short_name,
            company_name_local_language_name,
            description,
            business_address_city,
            unique_identifier_registration_number,
            unique_identifier_prior_registration_number,
            content='companies',
            content_rowid='id'
        )
    ''')

    # Populate FTS table
    cursor.execute('''
        INSERT INTO companies_fts(id, company_name_legal_name, company_name_trade_name,
            company_name_short_name, company_name_local_language_name, description, business_address_city,
            unique_identifier_registration_number, unique_identifier_prior_registration_number)
        SELECT id, company_name_legal_name, company_name_trade_name,
            company_name_short_name, company_name_local_language_name, description, business_address_city,
            unique_identifier_registration_number, unique_identifier_prior_registration_number
        FROM companies
    ''')

    conn.commit()

    # Stats
    count = cursor.execute('SELECT COUNT(*) FROM companies').fetchone()[0]
    print(f"  SQLite database: {count:,} records")
    print(f"  FTS5 index: built on name + description + city + registration number fields")
    print(f"  Indexes: {len(index_columns)} column indexes created")

    conn.close()


# ── Step 3: Build FAISS Vector Index ─────────────────────────────────────────

def build_search_text(df):
    """
    Concatenate relevant text fields into a single search_text column.
    This is what gets embedded for semantic search.
    """
    def make_text(row):
        parts = []
        for field in TEXT_FIELDS_FOR_SEARCH:
            val = row.get(field, '')
            if val and val.strip():
                parts.append(val.strip())
        return ' | '.join(parts)

    df['search_text'] = df.apply(make_text, axis=1)

    # Stats
    non_empty = (df['search_text'] != '').sum()
    avg_len = df['search_text'].str.len().mean()
    print(f"  Search text: {non_empty:,} non-empty records, avg {avg_len:.0f} chars")

    return df


def build_faiss_index(df, output_dir):
    """
    Build FAISS index from search_text embeddings.
    Uses sentence-transformers for multilingual embedding.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Truncate texts — the model has a ~128 token limit anyway
    # Keeping first 256 chars captures the company name + start of description
    texts = [t[:256] for t in df['search_text'].tolist()]
    ids = df['id'].tolist()

    print(f"  Encoding {len(texts):,} texts (truncated to 256 chars)...")
    # Encode all at once with built-in progress bar — sentence-transformers
    # handles batching internally and it's faster than manual batching
    all_embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=512,
    )
    all_embeddings = np.array(all_embeddings).astype('float32')
    print(f"  Embeddings shape: {all_embeddings.shape}")

    # Build FAISS index (Inner Product since we normalized embeddings = cosine similarity)
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)
    print(f"  FAISS index: {index.ntotal:,} vectors, dimension {dimension}")

    # Save index and ID mapping
    faiss_path = output_dir / 'company_search.faiss'
    faiss.write_index(index, str(faiss_path))
    print(f"  Saved FAISS index to: {faiss_path}")

    ids_path = output_dir / 'company_ids.json'
    with open(ids_path, 'w') as f:
        json.dump(ids, f)
    print(f"  Saved ID mapping to: {ids_path}")

    return index, ids


# ── Step 4: Search Functions ─────────────────────────────────────────────────

def search_semantic(query, index, ids, model, db_path, top_k=10, filters=None):
    """
    Semantic search: encode query → find nearest vectors → return company records.
    
    Args:
        query: natural language search query
        index: FAISS index
        ids: list mapping FAISS position → company ID
        model: SentenceTransformer model
        db_path: path to SQLite database
        top_k: number of results to return
        filters: dict of column_name → value for SQL filtering
    
    Returns:
        list of company dicts
    """
    # Encode query
    query_embedding = model.encode([query], normalize_embeddings=True).astype('float32')

    # Search FAISS (get more results if we're filtering, since some will be filtered out)
    search_k = top_k * 5 if filters else top_k
    scores, indices = index.search(query_embedding, search_k)

    # Get matching company IDs
    matching_ids = [ids[i] for i in indices[0] if i < len(ids)]
    matching_scores = scores[0][:len(matching_ids)]

    # Fetch from SQLite
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    results = []
    for company_id, score in zip(matching_ids, matching_scores):
        # Build query with optional filters
        sql = "SELECT * FROM companies WHERE id = ?"
        params = [company_id]

        if filters:
            for col, val in filters.items():
                sql_col = col.replace('.', '_')
                sql += f" AND {sql_col} = ?"
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


def search_structured(db_path, filters=None, text_query=None, limit=20):
    """
    Structured search: SQL-based filtering with optional FTS5 text search.
    
    Args:
        db_path: path to SQLite database
        filters: dict of column_name → value
        text_query: text to search in FTS5 index
        limit: max results
    
    Returns:
        list of company dicts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if text_query:
        # Use FTS5 for text search
        sql = """
            SELECT c.*, companies_fts.rank AS _rank
            FROM companies c
            JOIN companies_fts ON c.id = companies_fts.id
            WHERE companies_fts MATCH ?
        """
        params = [text_query]
    else:
        sql = "SELECT * FROM companies WHERE 1=1"
        params = []

    if filters:
        for col, val in filters.items():
            sql_col = col.replace('.', '_')
            sql += f" AND c.{sql_col} = ?" if text_query else f" AND {sql_col} = ?"
            params.append(val)

    if text_query:
        sql += " ORDER BY _rank LIMIT ?"
    else:
        sql += " LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    normalized_dir = Path('data/normalized')
    db_dir = Path('data/db')
    db_dir.mkdir(exist_ok=True, parents=True)

    db_path = db_dir / 'companies.db'

    # Step 1: Merge
    print(f"\n{'='*60}")
    print(f"  Step 1: Merging normalized CSVs")
    print(f"{'='*60}")
    df = merge_csvs(normalized_dir)
    if df is None:
        return

    # Save merged CSV for reference
    merged_path = db_dir / 'companies_merged.csv'
    df.to_csv(merged_path, index=False, encoding='utf-8')
    print(f"  Saved merged CSV to: {merged_path}")

    # Step 2: SQLite
    print(f"\n{'='*60}")
    print(f"  Step 2: Creating SQLite database")
    print(f"{'='*60}")
    create_sqlite_db(df, str(db_path))

    # Step 3: FAISS
    print(f"\n{'='*60}")
    print(f"  Step 3: Building FAISS vector index")
    print(f"{'='*60}")
    df = build_search_text(df)
    index, ids = build_faiss_index(df, db_dir)

    # Step 4: Quick test
    print(f"\n{'='*60}")
    print(f"  Step 4: Testing search")
    print(f"{'='*60}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    test_queries = [
        ("renewable energy Honduras", {"country_registration": "HN"}),
        ("shipping logistics Myanmar", {"country_registration": "MM"}),
        ("museum Oslo", {"country_registration": "NO"}),
        ("construction company", None),
    ]

    for query, filters in test_queries:
        filter_str = f" (filters: {filters})" if filters else ""
        print(f"\n  Query: '{query}'{filter_str}")
        results = search_semantic(query, index, ids, model, str(db_path), top_k=3, filters=filters)
        for i, r in enumerate(results):
            name = r.get('company_name_legal_name', '?')
            country = r.get('country_registration', '?')
            score = r.get('_score', 0)
            desc = r.get('description', '')[:80]
            print(f"    {i+1}. [{score:.3f}] {name} ({country})")
            if desc:
                print(f"       {desc}")

    # FTS test
    print(f"\n  FTS5 text search test: 'energy'")
    fts_results = search_structured(str(db_path), text_query="energy", limit=3)
    for i, r in enumerate(fts_results):
        name = r.get('company_name_legal_name', '?')
        country = r.get('country_registration', '?')
        print(f"    {i+1}. {name} ({country})")

    print(f"\n{'='*60}")
    print(f"  Database built successfully!")
    print(f"{'='*60}")
    print(f"  SQLite: {db_path}")
    print(f"  FAISS:  {db_dir / 'company_search.faiss'}")
    print(f"  IDs:    {db_dir / 'company_ids.json'}")
    print(f"  Merged: {merged_path}")


if __name__ == "__main__":
    main()
"""
Company Search Chatbot — Streamlit UI

A conversational interface for searching 150,000 companies across
Myanmar, Norway, and Honduras.

Usage (from project root):
    streamlit run src/chatbot.py
"""

import os
import sys
import json
import sqlite3
import numpy as np
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────

DB_DIR = Path('data/db')
DB_PATH = DB_DIR / 'companies.db'
FAISS_PATH = DB_DIR / 'company_search.faiss'
IDS_PATH = DB_DIR / 'company_ids.json'
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ── Lookup Tables ────────────────────────────────────────────────────────────

COUNTRY_NAMES = {
    "MM": "Myanmar 🇲🇲", "NO": "Norway 🇳🇴", "HN": "Honduras 🇭🇳",
}

STATUS_LABELS = {
    "FOR": ("Formation", "pre-life", "Pre-registered or in process for registration; business not yet started"),
    "RER": ("Re-registered", "life", "Re-registered after struck-off, dormancy, or resolution; back in business"),
    "ACT": ("Active", "life", "Registered and active; in business"),
    "INA": ("Inactive", "life", "Dormant or suspended; no business activities"),
    "ADM": ("Administration", "life", "Expired, forfeited, or under sequester; in administration"),
    "LIP": ("Liquidation, provisional", "dying", "In provisional liquidation"),
    "LIQ": ("Liquidation", "dying", "In liquidation, winding-up, or striking-off"),
    "LII": ("Liquidation compulsory – insolvency", "dying", "Compulsory/judicial winding-up (insolvency)"),
    "LIS": ("Liquidation compulsory – solvency", "dying", "Compulsory/judicial winding-up (solvency)"),
    "LIM": ("Liquidation voluntary – members", "dying", "Voluntary winding-up (members/partners)"),
    "LIC": ("Liquidation voluntary – creditors", "dying", "Voluntary winding-up (creditors)"),
    "REC": ("Receivership", "dying", "In receivership"),
    "BAN": ("Bankruptcy", "dying", "In bankruptcy"),
    "DSL": ("Dissolution", "dying", "In dissolution"),
    "MER": ("Merger", "dying", "In merger"),
    "CON": ("Conversion", "dying", "In conversion"),
    "CEA": ("Ceased", "died", "No longer registered; out of business"),
}

COMPANY_TYPE_LABELS = {
    # Category A – Association
    "ASS": ("Association", "A", "National/International Associations, Body Corporate"),
    "COO": ("Cooperative Society", "A", "Cooperative society"),
    "NPO": ("Not-for-Profit Organization", "A", "Not-for-Profit Organization"),
    "INO": ("International Organization", "A", "International Organization"),
    # Category B – Branch
    "BRA": ("Domestic Branch", "B", "Branch of a local company"),
    # Category C – Local Company
    "LTD": ("Limited Company", "C", "Local incorporated company with limited liability"),
    "PLC": ("Public Limited Company", "C", "Public company with shares listed or available to public"),
    "PVT": ("Private Limited Company", "C", "Private company with shares not publicly listed"),
    "LLC": ("Limited Liability Company", "C", "Limited Liability Company"),
    "LTG": ("Company Limited by Guarantee", "C", "Company limited by guarantee, not share capital"),
    "PUC": ("Public Unlimited Company", "C", "Public company with unlimited liability"),
    "PVU": ("Private Unlimited Company", "C", "Private company with unlimited liability"),
    "OPC": ("One Person Limited Company", "C", "Limited company with a single shareholder"),
    "LLO": ("One Person LLC", "C", "One Person Limited Liability Company"),
    "SLC": ("Simplified Limited Company", "C", "Simplified Limited Company"),
    "SLO": ("One Person Simplified Limited Company", "C", "One Person Simplified Limited Company"),
    "LPS": ("Limited Partnership with Share Capital", "C", "Limited Partnership with Share Capital"),
    "RPC": ("Restricted Purpose Company", "C", "Restricted Purpose Company"),
    "EST": ("Establishment (Anstalt)", "C", "Establishment / Anstalt"),
    # Category F – Foreign Entity
    "FCO": ("Foreign Company", "F", "Foreign company registered in the country"),
    "FBR": ("Foreign Branch", "F", "Foreign branch, not registered in the country"),
    # Category G – Governmental Organization
    "GOA": ("Public Administration", "G", "Government — public administration"),
    "GOS": ("Public Service", "G", "Government — public service"),
    "GOE": ("Public Education", "G", "Government — education"),
    "GOD": ("Domestic Government Entity", "G", "Government — domestic production/services"),
    "GOF": ("Foreign Governmental Organization", "G", "Foreign governmental organization"),
    # Category P – Private Company (no share capital)
    "PPS": ("Sole Proprietorship", "P", "Sole proprietorship / trader (1 owner)"),
    "BNM": ("Business Name", "P", "Business name (1 or multiple owners)"),
    "SPS": ("Simple Partnership", "P", "Simple partnership / close corporation"),
    "LLP": ("Limited Liability Partnership", "P", "Limited liability partnership"),
    "ULT": ("General Partnership", "P", "Unlimited liability / general partnership"),
    # Category T – Trust / Collective Investments
    "TRU": ("Trust Fund", "T", "Trust fund"),
    "FOU": ("Foundation", "T", "Foundation"),
    "ICO": ("Collective Investments – other", "T", "Trust/Funds/Companies/Contracts etc."),
    "ICV": ("Collective Investments (variable capital)", "T", "Collective investments with variable capital"),
    "ICF": ("Collective Investments (fixed capital)", "T", "Collective investments with fixed capital"),
    "LCI": ("Limited Partnership for Collective Investments", "T", "LP for collective investments"),
    "SPC": ("Segregated Portfolio Company", "T", "Segregated portfolio company"),
    "PCC": ("Protected Cell Company", "T", "Protected cell company"),
    "TRC": ("Trust Company", "T", "Trust company"),
}

CATEGORY_LABELS = {
    "A": "Association", "B": "Branch", "C": "Local Company",
    "F": "Foreign Entity", "G": "Governmental Organization",
    "P": "Private Company (no share capital)", "T": "Trust / Collective Investments",
}


def format_status(code):
    """Return 'CODE - Label' with emoji, or raw code if unknown."""
    if not code:
        return ""
    info = STATUS_LABELS.get(code)
    if info:
        label, stage, _ = info
        emoji = "🟢" if stage == "life" else "🔴" if stage == "died" else "🟡" if stage == "dying" else "⚪"
        return f"{emoji} {code} — {label}"
    return code


def format_company_type(code):
    """Return 'CODE - Label (Category)' or raw code if unknown."""
    if not code:
        return ""
    info = COMPANY_TYPE_LABELS.get(code)
    if info:
        label, cat, _ = info
        cat_label = CATEGORY_LABELS.get(cat, cat)
        return f"{code} — {label} ({cat_label})"
    # Maybe it's a category code itself
    if code in CATEGORY_LABELS:
        return f"{code} — {CATEGORY_LABELS[code]}"
    return code


def format_country(code):
    """Return country display name."""
    if not code:
        return ""
    return COUNTRY_NAMES.get(code, code)


# ── LLM Setup (Claude Opus 4.5) ───────────────────────────────────────────────

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
    """Call Claude Opus 4.5. No fallback — fail visibly."""
    try:
        response = _get_claude_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def detect_and_translate(text):
    """
    Check if text is in English. If not, translate it using the LLM.
    Returns (translated_text, source_language) or (None, None) if already English.
    Caches results in session_state to avoid re-translating on rerun.
    """
    if not text or len(text.strip()) < 10:
        return None, None

    # Check cache
    cache_key = f"translation_{hash(text[:200])}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    prompt = f"""Analyze this text and:
1. Determine the language (return the language name in English, e.g., "Norwegian", "Spanish", "Burmese")
2. If it is NOT English, translate it to English.
3. If it IS English, just say so.

Text: "{text[:1000]}"

Return ONLY valid JSON, no markdown, no extra text:
{{
  "language": "the detected language",
  "is_english": true or false,
  "translation": "English translation (only if not English, otherwise empty string)"
}}"""

    result = call_llm(prompt)
    if result:
        try:
            if result.startswith('```'):
                result = result.split('\n', 1)[1]
                result = result.rsplit('```', 1)[0]
                result = result.strip()
            if '<think>' in result:
                result = result.split('</think>')[-1].strip()
            parsed = json.loads(result)
            if not parsed.get('is_english', True):
                translation = parsed.get('translation', '')
                language = parsed.get('language', 'Unknown')
                st.session_state[cache_key] = (translation, language)
                return translation, language
        except json.JSONDecodeError:
            pass

    st.session_state[cache_key] = (None, None)
    return None, None


# ── Load Resources (cached) ─────────────────────────────────────────────────

@st.cache_resource
def load_faiss_index():
    import faiss
    index = faiss.read_index(str(FAISS_PATH))
    with open(IDS_PATH, 'r') as f:
        ids = json.load(f)
    return index, ids


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ── Query Parsing ────────────────────────────────────────────────────────────

COUNTRY_MAP = {
    "myanmar": "MM", "burma": "MM",
    "norway": "NO", "norge": "NO", "norwegian": "NO",
    "honduras": "HN", "honduran": "HN",
}

STATUS_MAP = {
    "active": "ACT", "inactive": "INA", "dormant": "INA", "suspended": "INA",
    "bankrupt": "BAN", "bankruptcy": "BAN",
    "liquidation": "LIQ", "liquidated": "CEA",
    "ceased": "CEA", "dissolved": "CEA", "closed": "CEA", "cancelled": "CEA",
}


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
            pass

    return parse_query_simple(user_query)


def parse_query_simple(user_query):
    query_lower = user_query.lower()
    filters = {}
    search_words = user_query.split()
    remove_words = set()

    for keyword, code in COUNTRY_MAP.items():
        if keyword in query_lower:
            filters["country_registration"] = code
            remove_words.add(keyword)
            break

    for keyword, code in STATUS_MAP.items():
        if keyword in query_lower:
            filters["status"] = code
            remove_words.add(keyword)
            break

    search_text = ' '.join(w for w in search_words if w.lower() not in remove_words)
    for filler in ['find', 'show', 'search', 'list', 'me', 'all', 'the', 'in', 'from', 'with', 'companies', 'company']:
        search_text = search_text.replace(filler, '')
    search_text = ' '.join(search_text.split())

    return {"search_text": search_text, "filters": filters}


# ── Search Functions ─────────────────────────────────────────────────────────

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
    """Run FTS5 keyword search on company name fields only."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT c.* FROM companies c
        JOIN companies_fts ON c.id = companies_fts.id
        WHERE companies_fts MATCH ?
    """
    # Restrict search to name columns and registration number fields using FTS5 column filters
    fts_query = f"{{company_name_legal_name company_name_trade_name company_name_short_name company_name_local_language_name unique_identifier_registration_number unique_identifier_prior_registration_number}}: {text_query}"
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
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_. ><=!''\"AND OR NOT ")
        if all(c in allowed_chars for c in where_extra):
            sql += f" AND {where_extra}"

    if order_by:
        valid_columns = [
            'share_capital_amount', 'num_employees', 'registration_date',
            'company_name_legal_name', 'business_address_city', 'status',
            'company_type', 'country_registration',
        ]
        if order_by in valid_columns:
            # Always exclude rows where the sorted field is NULL or empty
            sql += f" AND {order_by} IS NOT NULL AND {order_by} != ''"
            if order_by in ('share_capital_amount', 'num_employees'):
                sql += f" AND CAST({order_by} AS REAL) != 0"
                sql += f" ORDER BY CAST({order_by} AS REAL) {order_dir}"
            else:
                sql += f" ORDER BY {order_by} {order_dir}"

    sql += " LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def get_company_by_id(company_id):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM companies WHERE id = ?", [company_id]).fetchone()
    conn.close()
    return dict(row) if row else None


def get_db_stats():
    conn = sqlite3.connect(str(DB_PATH))
    stats = {}
    stats['total'] = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    stats['by_country'] = {}
    for row in conn.execute("SELECT country_registration, COUNT(*) FROM companies GROUP BY country_registration"):
        stats['by_country'][row[0]] = row[1]
    conn.close()
    return stats


# ── UI Components ────────────────────────────────────────────────────────────

def render_company_card(company, index, show_score=True):
    """Render a compact company card in search results. Returns True if clicked."""
    name = (company.get('company_name_legal_name')
            or company.get('company_name_short_name')
            or company.get('company_name_trade_name')
            or 'Unknown')
    country = company.get('country_registration', '')
    status = company.get('status', '')
    company_type = company.get('company_type', '')
    city = company.get('business_address_city', '')
    score = company.get('_score', 0)
    company_id = company.get('id', 0)
    description = company.get('description', '')
    reg_number = company.get('unique_identifier_registration_number', '')

    with st.container(border=True):
        # Row 1: Name + button
        col_name, col_btn = st.columns([5, 1])
        with col_name:
            st.markdown(f"**{name}**")
        with col_btn:
            clicked = st.button("View ➜", key=f"view_{company_id}_{index}", use_container_width=True)

        # Row 2: Key info pills
        pills = []
        if country:
            pills.append(format_country(country))
        if reg_number:
            pills.append(f"🔖 {reg_number}")
        if status:
            pills.append(format_status(status))
        if company_type:
            pills.append(format_company_type(company_type))
        if city:
            pills.append(f"📍 {city}")
        if show_score and score > 0:
            pills.append(f"Match: {score:.1%}")

        st.caption(" · ".join(pills))

        # Row 3: Description preview
        if description:
            preview = description[:150] + "..." if len(description) > 150 else description
            st.caption(f"*{preview}*")

        return clicked


def render_company_detail(company):
    """Render the full detail page for a company."""
    name = (company.get('company_name_legal_name')
            or company.get('company_name_short_name')
            or company.get('company_name_trade_name')
            or 'Unknown')

    # Back button
    if st.button("← Back to results"):
        st.session_state.pop('viewing_company', None)
        st.rerun()

    st.title(name)

    # ── Names ────────────────────────────────────────────────────────
    st.header("Company Names")
    name_data = {}
    if company.get('company_name_legal_name'):
        name_data["Legal Name"] = company['company_name_legal_name']
    if company.get('company_name_trade_name'):
        name_data["Trade Name"] = company['company_name_trade_name']
    if company.get('company_name_short_name'):
        name_data["Short Name"] = company['company_name_short_name']
    if company.get('company_name_local_language_name'):
        name_data["Local Language Name"] = company['company_name_local_language_name']

    for label, value in name_data.items():
        st.markdown(f"**{label}:** {value}")

    # ── Classification ───────────────────────────────────────────────
    st.header("Classification")
    col1, col2 = st.columns(2)

    with col1:
        status_code = company.get('status', '')
        if status_code:
            status_info = STATUS_LABELS.get(status_code)
            st.markdown(f"**Status:** {format_status(status_code)}")
            if status_info:
                st.caption(f"Life-cycle stage: {status_info[1]} — {status_info[2]}")

    with col2:
        type_code = company.get('company_type', '')
        if type_code:
            type_info = COMPANY_TYPE_LABELS.get(type_code)
            st.markdown(f"**Company Type:** {format_company_type(type_code)}")
            if type_info:
                st.caption(type_info[2])

    # ── Identifiers ──────────────────────────────────────────────────
    st.header("Identifiers")
    id_data = {}
    if company.get('unique_identifier_registration_number'):
        id_data["Registration Number"] = company['unique_identifier_registration_number']
    if company.get('unique_identifier_prior_registration_number'):
        id_data["Prior Registration Number"] = company['unique_identifier_prior_registration_number']

    col1, col2 = st.columns(2)
    items = list(id_data.items())
    for i, (label, value) in enumerate(items):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"**{label}:** {value}")

    # ── Registration ─────────────────────────────────────────────────
    reg_date = company.get('registration_date', '')
    if reg_date:
        st.markdown(f"**Registration Date:** {reg_date}")

    # ── Country & Addresses ──────────────────────────────────────────
    st.header("Location")

    # Country
    col1, col2, col3 = st.columns(3)
    with col1:
        reg_country = company.get('country_registration', '')
        if reg_country:
            st.markdown(f"**Registration Country:** {format_country(reg_country)}")
    with col2:
        biz_country = company.get('country_business_address', '')
        if biz_country:
            st.markdown(f"**Business Address Country:** {format_country(biz_country)}")
    with col3:
        post_country = company.get('country_postal_address', '')
        if post_country:
            st.markdown(f"**Postal Address Country:** {format_country(post_country)}")

    # Addresses
    col1, col2 = st.columns(2)
    with col1:
        biz_street = company.get('business_address_street', '')
        biz_city = company.get('business_address_city', '')
        if biz_street or biz_city:
            st.markdown("**Business Address:**")
            if biz_street:
                st.markdown(f"📍 {biz_street}")
            if biz_city:
                st.markdown(f"🏙️ {biz_city}")

    with col2:
        post_street = company.get('postal_address_street', '')
        post_city = company.get('postal_address_city', '')
        if post_street or post_city:
            st.markdown("**Postal Address:**")
            if post_street:
                st.markdown(f"📍 {post_street}")
            if post_city:
                st.markdown(f"🏙️ {post_city}")

    # ── Parent Company ───────────────────────────────────────────────
    parent_name = company.get('parent_company_company_name', '')
    parent_reg = company.get('parent_company_registration_number', '')
    if parent_name or parent_reg:
        st.header("Parent Company")
        if parent_name:
            st.markdown(f"**Name:** {parent_name}")
        if parent_reg:
            st.markdown(f"**Registration Number:** {parent_reg}")

    # ── Financials ───────────────────────────────────────────────────
    capital = company.get('share_capital_amount', '')
    currency = company.get('share_capital_currency', '')
    employees = company.get('num_employees', '')
    fully_paid = company.get('share_capital_fully_paid', '')

    if capital or employees:
        st.header("Financial Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            if capital and currency:
                try:
                    capital_num = float(capital)
                    st.metric("Share Capital", f"{capital_num:,.2f} {currency}")
                except (ValueError, TypeError):
                    st.metric("Share Capital", f"{capital} {currency}")
            elif capital:
                st.metric("Share Capital", capital)
        with col2:
            if employees:
                st.metric("Employees", employees)
        with col3:
            if fully_paid:
                st.metric("Fully Paid", "Yes" if str(fully_paid).lower() in ('true', '1', 'yes') else "No")

    # ── Description ──────────────────────────────────────────────────
    description = company.get('description', '')
    if description:
        st.header("Description / Business Activities")

        company_id = company.get('id', 0)
        translation_key = f"translated_{company_id}"

        # Check if we already have a translation stored
        if translation_key in st.session_state:
            translated_text, language = st.session_state[translation_key]
            st.info(f"🌐 Translated from {language}")
            st.markdown(translated_text)
            if st.button("Show original", key=f"original_{company_id}"):
                del st.session_state[translation_key]
                st.rerun()
        else:
            st.markdown(description)
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🌐 Translate", key=f"translate_{company_id}", type="primary"):
                    with st.spinner("Translating..."):
                        translation, language = detect_and_translate(description)
                    if translation:
                        st.session_state[translation_key] = (translation, language)
                        st.rerun()
                    else:
                        st.toast("Already in English — no translation needed!", icon="✅")


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Company Search",
        page_icon="🔍",
        layout="wide",
    )

    # Check if viewing a specific company
    if 'viewing_company' in st.session_state:
        company = get_company_by_id(st.session_state['viewing_company'])
        if company:
            render_company_detail(company)
            return

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 Company Search")
        if DB_PATH.exists():
            stats = get_db_stats()
            st.metric("Total Companies", f"{stats['total']:,}")
            for country, count in stats['by_country'].items():
                display = COUNTRY_NAMES.get(country, country)
                st.caption(f"{display}: {count:,}")

            st.divider()
            st.subheader("Filters")
            country_filter = st.selectbox(
                "Country",
                ["All"] + list(COUNTRY_NAMES.keys()),
                format_func=lambda x: "All countries" if x == "All" else COUNTRY_NAMES.get(x, x)
            )
            status_filter = st.selectbox(
                "Status",
                ["All", "ACT", "INA", "CEA", "BAN", "LIQ"],
                format_func=lambda x: "All statuses" if x == "All" else format_status(x)
            )
            num_results = st.slider("Max results", 20, 200, 20, step=20)
            results_per_page = 20

            st.divider()
            st.subheader("Search Mode")
            search_mode = st.radio(
                "Method",
                ["Semantic (AI)", "Keyword (FTS5)", "AI-Parsed"],
                help="**Semantic:** understands meaning, free & local.\n\n**Keyword:** exact text match, free & local.\n\n**AI-Parsed:** LLM extracts filters + supports sorting queries (uses Claude API)."
            )
        else:
            st.error("Database not found! Run `build_search_db.py` first.")
            return

    # ── Main Area ────────────────────────────────────────────────────
    st.title("🔍 Company Search")
    st.caption("Search 150,000 companies across Myanmar, Norway, and Honduras")

    # Chat input
    user_query = st.chat_input("Search for companies... (e.g., 'renewable energy companies in Honduras')")

    if user_query:
        # Build manual filters
        manual_filters = {}
        if country_filter != "All":
            manual_filters["country_registration"] = country_filter
        if status_filter != "All":
            manual_filters["status"] = status_filter

        with st.spinner("Searching..."):
            if search_mode == "AI-Parsed":
                parsed = parse_query_with_llm(user_query)
                strategy = parsed.get("strategy", "semantic")
                ai_filters = parsed.get("filters", {})
                ai_filters = {k: v for k, v in ai_filters.items() if v}
                combined_filters = {**ai_filters, **manual_filters}

                if strategy == "sql":
                    order_by = parsed.get("order_by", "")
                    order_dir = parsed.get("order_dir", "DESC")
                    limit = parsed.get("limit", num_results)
                    where_extra = parsed.get("where_extra", "")

                    results = search_sql(
                        filters=combined_filters if combined_filters else None,
                        order_by=order_by,
                        order_dir=order_dir,
                        limit=limit,
                        where_extra=where_extra if where_extra else None,
                    )
                    st.session_state['last_results'] = results
                    st.session_state['last_query'] = user_query
                    st.session_state['last_show_score'] = False
                    st.session_state['last_info'] = f"🤖 SQL query · Sort: **{order_by} {order_dir}** · Filters: {combined_filters if combined_filters else 'none'}"
                    st.session_state['current_page'] = 1
                else:
                    search_text = parsed.get("search_text", user_query)
                    results = search_semantic(search_text, filters=combined_filters, top_k=num_results)
                    st.session_state['last_results'] = results
                    st.session_state['last_query'] = user_query
                    st.session_state['last_show_score'] = True
                    st.session_state['last_info'] = f"🤖 Semantic search: **{search_text}** · Filters: {combined_filters if combined_filters else 'none'}"
                    st.session_state['current_page'] = 1

            elif search_mode == "Semantic (AI)":
                results = search_semantic(user_query, filters=manual_filters if manual_filters else None, top_k=num_results)
                st.session_state['last_results'] = results
                st.session_state['last_query'] = user_query
                st.session_state['last_show_score'] = True
                st.session_state['last_info'] = None
                st.session_state['current_page'] = 1

            else:  # Keyword
                results = search_fts(user_query, filters=manual_filters if manual_filters else None, limit=num_results)
                st.session_state['last_results'] = results
                st.session_state['last_query'] = user_query
                st.session_state['last_show_score'] = False
                st.session_state['last_info'] = None
                st.session_state['current_page'] = 1

    # Display results (from current search or previous session)
    if 'last_results' in st.session_state and st.session_state['last_results'] is not None:
        results = st.session_state['last_results']
        show_score = st.session_state.get('last_show_score', True)
        info_msg = st.session_state.get('last_info')

        if info_msg:
            st.info(info_msg)

        if results:
            total = len(results)
            total_pages = (total + results_per_page - 1) // results_per_page

            # Initialize page
            if 'current_page' not in st.session_state:
                st.session_state['current_page'] = 1

            current_page = st.session_state['current_page']
            # Clamp page to valid range
            current_page = max(1, min(current_page, total_pages))

            start_idx = (current_page - 1) * results_per_page
            end_idx = min(start_idx + results_per_page, total)

            st.success(f"Found **{total}** companies for: *{st.session_state.get('last_query', '')}* — showing {start_idx + 1}–{end_idx}")

            # Top pagination
            if total_pages > 1:
                cols = st.columns([1, 1, 3, 1, 1])
                with cols[0]:
                    if st.button("◀ Prev", disabled=(current_page <= 1), use_container_width=True):
                        st.session_state['current_page'] = current_page - 1
                        st.rerun()
                with cols[1]:
                    st.markdown(f"**Page {current_page} / {total_pages}**")
                with cols[3]:
                    if st.button("Next ▶", disabled=(current_page >= total_pages), use_container_width=True):
                        st.session_state['current_page'] = current_page + 1
                        st.rerun()

            # Display current page of results
            page_results = results[start_idx:end_idx]
            for i, company in enumerate(page_results):
                clicked = render_company_card(company, start_idx + i, show_score=show_score)
                if clicked:
                    st.session_state['viewing_company'] = company['id']
                    st.rerun()

            # Bottom pagination
            if total_pages > 1:
                cols = st.columns([1, 1, 3, 1, 1])
                with cols[0]:
                    if st.button("◀ Prev", disabled=(current_page <= 1), key="prev_bottom", use_container_width=True):
                        st.session_state['current_page'] = current_page - 1
                        st.rerun()
                with cols[1]:
                    st.markdown(f"**Page {current_page} / {total_pages}**")
                with cols[3]:
                    if st.button("Next ▶", disabled=(current_page >= total_pages), key="next_bottom", use_container_width=True):
                        st.session_state['current_page'] = current_page + 1
                        st.rerun()
        else:
            st.warning("No companies found. Try broader terms or different filters.")

    elif not user_query:
        # Welcome message
        st.markdown("### Try these searches:")
        examples = [
            "🌿 `renewable energy companies in Honduras`",
            "🚢 `shipping logistics Myanmar`",
            "🏛️ `museums in Norway`",
            "📊 `top 20 companies with most share capital in Norway` *(AI-Parsed mode)*",
            "👥 `companies with most employees in Norway` *(AI-Parsed mode)*",
            "🏗️ `construction`",
        ]
        for ex in examples:
            st.markdown(f"- {ex}")


if __name__ == "__main__":
    main()
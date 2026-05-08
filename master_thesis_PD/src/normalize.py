"""
Normalizer: Standardize extracted values to canonical format.

After auto_mapper.py extracts raw values, this script normalizes them:
1. Dates → YYYY-MM-DD (pure Python)
2. Country names → ISO 3166-1 alpha-2 codes (pure Python)
3. Status values → canonical status codes (LLM creates mapping, Python applies)
4. Company type values → canonical type codes (LLM creates mapping, Python applies)
5. Share capital → clean numeric values

LLM: Claude Opus 4.5 (no fallback — single model for reproducibility).

Usage (from project root):
    python src/normalize.py
"""

import os
import sys
import json
import time
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import CANONICAL_SCHEMA

load_dotenv()

# ── LLM Setup (Claude Opus 4.5) ──────────────────────────────────────────────
#
# Normalization only makes 2 LLM calls per country (status + company_type),
# so 6 calls total for 3 countries. Opus 4.5 is used here for consistency
# with auto_mapper.py and because the categorical mapping task benefits
# from strong multilingual understanding (e.g. mapping "Disuelta" or
# "Stiftelse" to the right canonical codes).
#
CLAUDE_MODEL = "claude-opus-4-5-20251101"
CLAUDE_MAX_TOKENS = 4096

_claude_client = None

def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        _claude_client = anthropic.Anthropic(api_key=api_key)
    return _claude_client


# ── 1. Date Normalization (Pure Python) ──────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%d",       # 2018-05-17
    "%d/%m/%Y",       # 17/05/2018
    "%m/%d/%Y",       # 05/17/2018
    "%d-%m-%Y",       # 17-05-2018
    "%Y/%m/%d",       # 2018/05/17
    "%d.%m.%Y",       # 17.05.2018
    "%Y%m%d",         # 20180517
    "%B %d, %Y",      # May 17, 2018
    "%d %B %Y",       # 17 May 2018
    "%d %b %Y",       # 17 May 2018
]

def normalize_date(value):
    """Try multiple date formats, return YYYY-MM-DD or empty string."""
    if not value or pd.isna(value):
        return ''
    
    value = str(value).strip()
    if not value or value in ('', '-', 'None', 'null', 'N/A'):
        return ''
    
    # Handle datetime strings with time component
    if 'T' in value:
        value = value.split('T')[0]
    
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            # Sanity check: year between 1800 and 2030
            if 1800 <= dt.year <= 2030:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return ''  # Return empty


# ── 2. Country Code Normalization (Pure Python) ─────────────────────────────

COUNTRY_NAME_TO_CODE = {
    # Common names that might appear in the data
    "myanmar": "MM", "burma": "MM",
    "norway": "NO", "norge": "NO",
    "honduras": "HN",
    "singapore": "SG",
    "united states": "US", "usa": "US", "us": "US",
    "united kingdom": "GB", "uk": "GB", "england": "GB",
    "china": "CN", "peoples republic of china": "CN",
    "japan": "JP", "nippon": "JP",
    "india": "IN",
    "thailand": "TH",
    "malaysia": "MY",
    "vietnam": "VN", "viet nam": "VN",
    "indonesia": "ID",
    "australia": "AU",
    "canada": "CA",
    "germany": "DE", "deutschland": "DE",
    "france": "FR",
    "italy": "IT", "italia": "IT",
    "spain": "ES", "españa": "ES",
    "sweden": "SE", "sverige": "SE",
    "denmark": "DK", "danmark": "DK",
    "finland": "FI", "suomi": "FI",
    "netherlands": "NL", "holland": "NL",
    "belgium": "BE", "belgique": "BE",
    "switzerland": "CH", "schweiz": "CH", "suisse": "CH",
    "austria": "AT", "österreich": "AT",
    "portugal": "PT",
    "brazil": "BR", "brasil": "BR",
    "mexico": "MX", "méxico": "MX",
    "south korea": "KR", "korea": "KR",
    "taiwan": "TW",
    "hong kong": "HK",
    "philippines": "PH",
    "new zealand": "NZ",
    "ireland": "IE",
    "south africa": "ZA",
    "russia": "RU",
    "turkey": "TR", "türkiye": "TR",
    "poland": "PL", "polska": "PL",
    "czech republic": "CZ", "czechia": "CZ",
    "romania": "RO",
    "hungary": "HU", "magyarország": "HU",
    "greece": "GR",
    "israel": "IL",
    "united arab emirates": "AE", "uae": "AE",
    "saudi arabia": "SA",
    "qatar": "QA",
    "kuwait": "KW",
    "bahrain": "BH",
    "egypt": "EG",
    "nigeria": "NG",
    "kenya": "KE",
    "ghana": "GH",
    "colombia": "CO",
    "argentina": "AR",
    "chile": "CL",
    "peru": "PE",
    "venezuela": "VE",
    "costa rica": "CR",
    "panama": "PA", "panamá": "PA",
    "guatemala": "GT",
    "el salvador": "SV",
    "nicaragua": "NI",
    "belize": "BZ",
    "dominican republic": "DO",
    "cuba": "CU",
    "jamaica": "JM",
    "puerto rico": "PR",
    "british virgin islands": "VG", "bvi": "VG",
    "cayman islands": "KY",
    "bermuda": "BM",
    "luxembourg": "LU",
    "liechtenstein": "LI",
    "monaco": "MC",
    "iceland": "IS", "ísland": "IS",
    "malta": "MT",
    "cyprus": "CY",
    "croatia": "HR", "hrvatska": "HR",
    "serbia": "RS",
    "ukraine": "UA",
    "cambodia": "KH",
    "laos": "LA",
    "bangladesh": "BD",
    "sri lanka": "LK",
    "pakistan": "PK",
    "nepal": "NP",
}

def normalize_country(value):
    """Convert country name to ISO 3166-1 alpha-2 code."""
    if not value or pd.isna(value):
        return ''
    
    value = str(value).strip()
    if not value or value in ('', '-', 'None', 'null'):
        return ''
    
    # Already a 2-letter code?
    if len(value) == 2 and value.isalpha():
        return value.upper()
    
    # Lookup by name
    code = COUNTRY_NAME_TO_CODE.get(value.lower())
    if code:
        return code
    
    return value  # Return as-is if no match


# ── 3. LLM-Assisted Categorical Normalization ───────────────────────────────

def get_unique_values(df, column, max_values=100):
    """Get unique non-empty values from a column, sorted by frequency."""
    values = df[column].dropna()
    values = values[values != '']
    counts = values.value_counts()
    return counts.head(max_values)


def build_status_taxonomy_for_prompt():
    """Build a concise representation of the status taxonomy for the LLM."""
    status_info = CANONICAL_SCHEMA["status"]["allowed_values"]
    lines = []
    for code, info in status_info.items():
        lines.append(f'  "{code}": "{info["label"]}" — {info["description"]} (stage: {info["life_cycle_stage"]})')
    return '\n'.join(lines)


def build_company_type_taxonomy_for_prompt():
    """Build a concise representation of the company type taxonomy for the LLM."""
    type_info = CANONICAL_SCHEMA["company_type"]["allowed_values"]
    lines = []
    for cat_code, cat_info in type_info.items():
        lines.append(f'  Category "{cat_code}" ({cat_info["label"]}): {cat_info["description"]}')
        for sub_code, sub_label in cat_info["sub_types"].items():
            lines.append(f'    "{sub_code}": {sub_label}')
    return '\n'.join(lines)


def llm_map_values(unique_values, field_name, taxonomy_str, country_name):
    """
    Ask the LLM to map raw values to canonical codes.
    
    Args:
        unique_values: dict of {value: count} pairs
        field_name: "status" or "company_type"
        taxonomy_str: the taxonomy description for the prompt
        country_name: for context
    
    Returns:
        dict mapping raw_value -> canonical_code
    """
    values_list = []
    for val, count in unique_values.items():
        values_list.append(f'  "{val}" (appears {count:,} times)')
    values_str = '\n'.join(values_list)

    prompt = f"""You are a data normalization expert. Map the following raw "{field_name}" values from the {country_name} company registry to the canonical codes below.

**Raw values to map:**
{values_str}

**Canonical taxonomy:**
{taxonomy_str}

**Instructions:**
1. Map EACH raw value to exactly ONE canonical code from the taxonomy.
2. Consider the meaning in the original language (e.g., "ACTIVO" = Active in Spanish).
3. If a value doesn't fit any specific sub-code, use the closest match.
4. Every raw value MUST be mapped — do not skip any.
5. PAY ATTENTION TO VERB TENSE for status values:
   - Past tense (completed action) means the process is FINISHED → map to "CEA" (Ceased).
     Examples: "Liquidada" (was liquidated), "Disuelta" (was dissolved), "Cancelada" (was cancelled),
     "Fusionada" (was merged), "Struck Off" → all map to "CEA" because the company is dead.
   - Present tense or progressive means the process is ONGOING → map to the specific dying stage.
     Examples: "In Liquidation", "Under Avvikling", "En liquidación" → map to "LIQ", "DSL", etc.
   - "Suspended" or "Dormant" → "INA" (Inactive), NOT ceased.

**Output Format (JSON only, no markdown, no extra text):**
{{
  "mappings": {{
    "raw_value_1": "CODE",
    "raw_value_2": "CODE"
  }}
}}

Respond with ONLY valid JSON."""

    for attempt in range(2):  # first try + one JSON-parse retry
        try:
            response = _get_claude_client().messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('\n', 1)[1]
                result_text = result_text.rsplit('```', 1)[0]
                result_text = result_text.strip()

            parsed = json.loads(result_text)
            mappings = parsed.get("mappings", parsed)
            print(f"    (using model: {CLAUDE_MODEL})")
            return mappings

        except json.JSONDecodeError:
            if attempt == 0:
                print(f"    JSON parse error — retrying once")
                continue
            print(f"    JSON parse error on retry — giving up")

        except Exception as e:
            err = str(e)
            err_lower = err.lower()
            if "429" in err or "rate_limit" in err_lower:
                print(f"    Rate-limited. Waiting 60s and retrying...")
                time.sleep(60)
                if attempt == 0:
                    continue
            elif "401" in err or "authentication" in err_lower:
                print(f"    !! Authentication error: {err[:200]}")
                raise
            else:
                print(f"    Error: {err[:200]}")

    print(f"    ERROR: Failed to get mappings for {field_name}")
    return {}


# ── 4. Share Capital Normalization (Pure Python) ─────────────────────────────

def normalize_share_capital(value):
    """Clean share capital amount to a numeric value."""
    if not value or pd.isna(value):
        return ''
    
    value = str(value).strip()
    if not value or value in ('', '-', 'None', 'null', 'N/A'):
        return ''
    
    # Remove currency symbols and whitespace
    value = re.sub(r'[^\d.,\-]', '', value)
    
    # Handle European format (1.000.000,50 → 1000000.50)
    if ',' in value and '.' in value:
        if value.rindex(',') > value.rindex('.'):
            # European: dots are thousands, comma is decimal
            value = value.replace('.', '').replace(',', '.')
        # else: American format, just remove commas
        else:
            value = value.replace(',', '')
    elif ',' in value:
        # Could be decimal or thousands — if 3 digits after comma, it's thousands
        parts = value.split(',')
        if len(parts[-1]) == 3:
            value = value.replace(',', '')
        else:
            value = value.replace(',', '.')
    
    try:
        num = float(value)
        if num == 0:
            return ''  # Treat 0 as missing
        return num
    except ValueError:
        return ''


# ── Main Pipeline ────────────────────────────────────────────────────────────

def normalize_country_data(df, country_name):
    """
    Apply all normalizations to a country's DataFrame.
    Returns the normalized DataFrame and a log of changes.
    """
    log = {"country": country_name, "changes": {}}

    # ── Dates ──────────────────────────────────────────────────────────
    if 'registration_date' in df.columns:
        before = df['registration_date'].copy()
        df['registration_date'] = df['registration_date'].apply(normalize_date)
        changed = (before != df['registration_date']).sum()
        log["changes"]["registration_date"] = f"{changed:,} values reformatted"
        print(f"    registration_date: {changed:,} values reformatted to YYYY-MM-DD")

    # ── Country codes ──────────────────────────────────────────────────
    country_cols = [c for c in df.columns if c.startswith('country.')]
    for col in country_cols:
        before = df[col].copy()
        df[col] = df[col].apply(normalize_country)
        changed = (before != df[col]).sum()
        if changed > 0:
            log["changes"][col] = f"{changed:,} values converted to ISO codes"
            print(f"    {col}: {changed:,} values converted to ISO codes")

    # ── Share capital ──────────────────────────────────────────────────
    if 'share_capital.amount' in df.columns:
        before_empty = (df['share_capital.amount'].apply(lambda x: x == '' or pd.isna(x))).sum()
        df['share_capital.amount'] = df['share_capital.amount'].apply(normalize_share_capital)
        after_empty = (df['share_capital.amount'].apply(lambda x: x == '' or pd.isna(x))).sum()
        zeroes_removed = after_empty - before_empty
        if zeroes_removed > 0:
            log["changes"]["share_capital.amount"] = f"{zeroes_removed:,} zero values cleared"
            print(f"    share_capital.amount: {zeroes_removed:,} zero values cleared")

    # ── Status (LLM-assisted) ──────────────────────────────────────────
    if 'status' in df.columns:
        unique_statuses = get_unique_values(df, 'status')
        if len(unique_statuses) > 0:
            print(f"\n    Status: {len(unique_statuses)} unique values found")
            for val, count in unique_statuses.head(10).items():
                print(f"      {val:40s}: {count:>6,}")
            if len(unique_statuses) > 10:
                print(f"      ... and {len(unique_statuses) - 10} more")

            taxonomy_str = build_status_taxonomy_for_prompt()
            print(f"    Asking Claude to map status values...")
            status_map = llm_map_values(unique_statuses, "status", taxonomy_str, country_name)

            if status_map:
                print(f"    Status mappings:")
                for raw, code in status_map.items():
                    print(f"      {raw:40s} → {code}")

                df['status'] = df['status'].map(status_map).fillna(df['status'])
                log["changes"]["status"] = status_map

    # ── Company Type (LLM-assisted) ────────────────────────────────────
    if 'company_type' in df.columns:
        unique_types = get_unique_values(df, 'company_type')
        if len(unique_types) > 0:
            print(f"\n    Company type: {len(unique_types)} unique values found")
            for val, count in unique_types.head(10).items():
                print(f"      {val:40s}: {count:>6,}")
            if len(unique_types) > 10:
                print(f"      ... and {len(unique_types) - 10} more")

            taxonomy_str = build_company_type_taxonomy_for_prompt()
            print(f"    Asking Claude to map company type values...")
            type_map = llm_map_values(unique_types, "company_type", taxonomy_str, country_name)

            if type_map:
                print(f"    Company type mappings:")
                for raw, code in type_map.items():
                    print(f"      {raw:40s} → {code}")

                df['company_type'] = df['company_type'].map(type_map).fillna(df['company_type'])
                log["changes"]["company_type"] = type_map

    return df, log


def main():
    processed_dir = Path('data/processed')
    output_dir = Path('data/normalized')
    output_dir.mkdir(exist_ok=True, parents=True)

    countries = {
        "Myanmar": "myanmar_flat.csv",
        "Norway": "norway_flat.csv",
        "Honduras": "honduras_flat.csv",
    }

    all_logs = []

    for country_name, filename in countries.items():
        input_path = processed_dir / filename
        if not input_path.exists():
            print(f"\n  SKIP {country_name}: {input_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Normalizing: {country_name}")
        print(f"{'='*60}")

        df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
        print(f"  Loaded {len(df):,} records, {len(df.columns)} columns")

        df, log = normalize_country_data(df, country_name)

        # Save normalized CSV
        output_path = output_dir / filename.replace('_flat.csv', '_normalized.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n  Saved to: {output_path}")

        # Show data coverage after normalization
        print(f"\n  Data coverage (post-normalization):")
        for col in df.columns:
            non_empty = df[col].apply(lambda x: x != '' and x is not None and pd.notna(x)).sum()
            pct = non_empty / len(df) * 100
            print(f"    {col:40s}: {non_empty:>6,} / {len(df):,} ({pct:.1f}%)")

        all_logs.append(log)

        print(f"\n  Waiting 10 seconds...")
        time.sleep(10)

    # Save normalization log
    log_path = output_dir / 'normalization_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Normalization log saved to: {log_path}")

    print(f"\n{'='*60}")
    print("All countries normalized!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
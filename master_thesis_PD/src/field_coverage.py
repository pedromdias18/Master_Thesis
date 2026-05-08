"""
field_coverage.py — Count non-empty values per canonical field, per country.

Reads the three normalized CSVs and reports how many of the 50,000 records
in each country have a non-empty value for each of the 23 canonical sub-fields.
Output is printed to the terminal and written to data/db/field_coverage.json
for use in Chapter 4.

Usage (from project root):
    python src/field_coverage.py
"""

import json
from pathlib import Path
import pandas as pd

NORMALIZED_DIR = Path('data/normalized')
OUTPUT_PATH    = Path('data/db/field_coverage.json')

FILES = {
    "Myanmar":  "myanmar_normalized.csv",
    "Norway":   "norway_normalized.csv",
    "Honduras": "honduras_normalized.csv",
}

# All 23 canonical sub-fields. Missing columns in a given CSV are treated as
# 0 coverage (that field was not mapped for that country).
CANONICAL_FIELDS = [
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


def non_empty_count(series):
    """Count values that are not NaN, not empty, and not just whitespace."""
    s = series.fillna('').astype(str).str.strip()
    return int((s != '').sum())


def main():
    report = {}

    for country, filename in FILES.items():
        path = NORMALIZED_DIR / filename
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {country}")
            continue

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        total = len(df)
        print(f"\n=== {country} ({total:,} records) ===")

        country_report = {"total_records": total, "fields": {}}
        for field in CANONICAL_FIELDS:
            if field in df.columns:
                count = non_empty_count(df[field])
            else:
                count = 0
            pct = 100.0 * count / total if total else 0.0
            country_report["fields"][field] = {"count": count, "pct": round(pct, 2)}
            print(f"  {field:42s}  {count:>7,} / {total:,}  ({pct:5.1f}%)")

        report[country] = country_report

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

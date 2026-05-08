"""
Human-in-the-Loop Mapping Review GUI

Review and correct:
1. Extraction rules (which JSON field maps to which canonical field)
2. Normalization mappings (raw values → standardized codes)

After review, re-process the data with corrected rules.

Usage (from project root):
    streamlit run src/review_mappings.py
"""

import os
import sys
import json
import time
import sqlite3
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import CANONICAL_SCHEMA

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR = Path('data/raw')
SAMPLES_DIR = Path('data/samples')
PROCESSED_DIR = Path('data/processed')
NORMALIZED_DIR = Path('data/normalized')

COUNTRIES = {
    "Myanmar": {
        "raw": RAW_DIR / 'mm_entities.json',
        "samples": SAMPLES_DIR / 'myanmar_samples.json',
        "rules": PROCESSED_DIR / 'myanmar_flat_rules.json',
        "flat_csv": PROCESSED_DIR / 'myanmar_flat.csv',
        "norm_csv": NORMALIZED_DIR / 'myanmar_normalized.csv',
        "norm_log": NORMALIZED_DIR / 'normalization_log.json',
    },
    "Norway": {
        "raw": RAW_DIR / 'no_entities.json',
        "samples": SAMPLES_DIR / 'norway_samples.json',
        "rules": PROCESSED_DIR / 'norway_flat_rules.json',
        "flat_csv": PROCESSED_DIR / 'norway_flat.csv',
        "norm_csv": NORMALIZED_DIR / 'norway_normalized.csv',
        "norm_log": NORMALIZED_DIR / 'normalization_log.json',
    },
    "Honduras": {
        "raw": RAW_DIR / 'hn_entities.json',
        "samples": SAMPLES_DIR / 'honduras_samples.json',
        "rules": PROCESSED_DIR / 'honduras_flat_rules.json',
        "flat_csv": PROCESSED_DIR / 'honduras_flat.csv',
        "norm_csv": NORMALIZED_DIR / 'honduras_normalized.csv',
        "norm_log": NORMALIZED_DIR / 'normalization_log.json',
    },
}


# ── Schema Helpers ───────────────────────────────────────────────────────────

def get_all_canonical_fields():
    """Get flat list of all canonical field paths including sub-fields."""
    fields = []
    for field_name, field_info in CANONICAL_SCHEMA.items():
        if field_info["data_type"] == "object" and "schema" in field_info:
            for sub_name, sub_info in field_info["schema"].items():
                fields.append({
                    "path": f"{field_name}.{sub_name}",
                    "description": sub_info.get("description", ""),
                    "data_type": sub_info.get("data_type", "string"),
                    "parent": field_name,
                })
        else:
            fields.append({
                "path": field_name,
                "description": field_info.get("description", ""),
                "data_type": field_info.get("data_type", "string"),
                "parent": None,
            })
    return fields


def get_all_json_paths(sample_records, prefix=""):
    """Recursively extract all possible JSON paths from sample records."""
    paths = set()

    for record in sample_records:
        _extract_paths(record, prefix, paths)

    return sorted(paths)


def _extract_paths(obj, prefix, paths):
    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{prefix}.{key}" if prefix else key
            paths.add(current)
            _extract_paths(value, current, paths)
    elif isinstance(obj, list) and len(obj) > 0:
        paths.add(f"{prefix}[*]")
        paths.add(f"{prefix}[0]")
        if isinstance(obj[0], dict):
            for key, value in obj[0].items():
                paths.add(f"{prefix}[0].{key}")
                paths.add(f"{prefix}[*].{key}")
                _extract_paths(value, f"{prefix}[0].{key}", paths)


EXTRACTION_TYPES = [
    "direct",
    "nested_field",
    "first_element_field",
    "join_list",
    "boolean_logic",
    "hardcode",
    "fallback_chain",
]

# ── Status & Company Type Lookups ────────────────────────────────────────────

STATUS_LABELS = {
    "FOR": "Formation — Pre-registered; business not yet started",
    "RER": "Re-registered — Back in business after struck-off/dormancy",
    "ACT": "Active — Registered and active; in business",
    "INA": "Inactive — Dormant or suspended; no business activities",
    "ADM": "Administration — Expired, forfeited, or under sequester",
    "LIP": "Liquidation, provisional",
    "LIQ": "Liquidation — In winding-up or striking-off",
    "LII": "Liquidation compulsory – insolvency",
    "LIS": "Liquidation compulsory – solvency",
    "LIM": "Liquidation voluntary – members",
    "LIC": "Liquidation voluntary – creditors",
    "REC": "Receivership",
    "BAN": "Bankruptcy",
    "DSL": "Dissolution",
    "MER": "Merger",
    "CON": "Conversion",
    "CEA": "Ceased — No longer registered; out of business",
}

COMPANY_TYPE_LABELS = {
    "ASS": "Association", "COO": "Cooperative Society", "NPO": "Not-for-Profit",
    "INO": "International Organization",
    "BRA": "Domestic Branch",
    "LTD": "Limited Company", "PLC": "Public Limited Company",
    "PVT": "Private Limited Company", "LLC": "Limited Liability Company",
    "LTG": "Company Limited by Guarantee",
    "PUC": "Public Unlimited Company", "PVU": "Private Unlimited Company",
    "OPC": "One Person Limited Company", "LLO": "One Person LLC",
    "SLC": "Simplified Limited Company", "SLO": "One Person Simplified LC",
    "LPS": "Limited Partnership with Share Capital",
    "RPC": "Restricted Purpose Company", "EST": "Establishment (Anstalt)",
    "FCO": "Foreign Company", "FBR": "Foreign Branch",
    "GOA": "Public Administration", "GOS": "Public Service",
    "GOE": "Public Education", "GOD": "Domestic Government Entity",
    "GOF": "Foreign Governmental Organization",
    "PPS": "Sole Proprietorship", "BNM": "Business Name",
    "SPS": "Simple Partnership", "LLP": "Limited Liability Partnership",
    "ULT": "General Partnership",
    "TRU": "Trust Fund", "FOU": "Foundation",
    "ICO": "Collective Investments – other",
    "ICV": "Collective Investments (variable capital)",
    "ICF": "Collective Investments (fixed capital)",
    "LCI": "LP for Collective Investments",
    "SPC": "Segregated Portfolio Company", "PCC": "Protected Cell Company",
    "TRC": "Trust Company",
}


# ── Re-processing ────────────────────────────────────────────────────────────

def reprocess_country(country_name, paths, rules):
    """Re-run extraction with updated rules for a country."""
    from auto_mapper import apply_rules_to_all

    # Load raw data
    with open(paths["raw"], 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # Apply rules
    flat_records = apply_rules_to_all(all_data, rules)

    # Save
    df = pd.DataFrame(flat_records)
    df.to_csv(paths["flat_csv"], index=False, encoding='utf-8')

    # Save updated rules
    with open(paths["rules"], 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    return df


# ── Streamlit App ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Mapping Review",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 Mapping Review — Human-in-the-Loop")
    st.caption("Review and correct LLM-generated extraction rules and normalization mappings")

    tab_extract, tab_normalize = st.tabs(["📋 Extraction Rules", "🔄 Normalization Mappings"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: EXTRACTION RULES
    # ══════════════════════════════════════════════════════════════════
    with tab_extract:
        country_name = st.selectbox("Select country", list(COUNTRIES.keys()), key="extract_country")
        paths = COUNTRIES[country_name]

        # Load rules
        if not paths["rules"].exists():
            st.error(f"Rules file not found: {paths['rules']}. Run auto_mapper.py first.")
            return

        with open(paths["rules"], 'r', encoding='utf-8') as f:
            current_rules = json.load(f)

        # Load samples to get available JSON paths
        if paths["samples"].exists():
            with open(paths["samples"], 'r', encoding='utf-8') as f:
                samples = json.load(f)
            available_paths = get_all_json_paths(samples)
        else:
            available_paths = []
            st.warning("Sample file not found — cannot show available JSON paths")

        # Build rules lookup
        rules_by_field = {r["canonical_field"]: r for r in current_rules}

        # Get all canonical fields
        canonical_fields = get_all_canonical_fields()

        # Initialize session state for edits
        state_key = f"extract_edits_{country_name}"
        if state_key not in st.session_state:
            st.session_state[state_key] = {}

        st.markdown(f"### {country_name} — Extraction Rules")
        st.markdown(f"**{len(current_rules)}** rules from LLM · **{len(canonical_fields)}** canonical fields · **{len(available_paths)}** available JSON paths")

        # Show sample data expander
        if samples:
            with st.expander("👀 View sample data (first record)"):
                st.json(samples[0])

        st.divider()

        # Track changes
        has_changes = False
        edited_rules = list(current_rules)  # Copy

        # Group fields by parent
        current_parent = None

        for cf_info in canonical_fields:
            cf_path = cf_info["path"]
            cf_desc = cf_info["description"]
            parent = cf_info["parent"]

            # Section header for parent fields
            if parent and parent != current_parent:
                current_parent = parent
                parent_info = CANONICAL_SCHEMA.get(parent, {})
                st.markdown(f"#### {parent}")
                if parent_info.get("description"):
                    st.caption(parent_info["description"])

            elif not parent and current_parent is not None:
                current_parent = None

            # Current mapping
            existing_rule = rules_by_field.get(cf_path)

            if existing_rule:
                current_json_path = str(existing_rule.get("json_path", ""))
                current_type = existing_rule.get("extraction_type", "direct")
                current_reasoning = existing_rule.get("reasoning", "")
                mapped = True
            else:
                current_json_path = ""
                current_type = "direct"
                current_reasoning = ""
                mapped = False

            # Build display
            col_field, col_path, col_type, col_status = st.columns([2, 3, 1.5, 0.5])

            with col_field:
                label = f"**{cf_path}**" if mapped else f"~~{cf_path}~~"
                st.markdown(label)
                if cf_desc:
                    st.caption(cf_desc[:80])

            with col_path:
                # Dropdown with all available paths + current + empty option
                path_options = ["— unmapped —"] + available_paths
                if current_json_path and current_json_path not in path_options:
                    path_options.insert(1, current_json_path)

                # Find current index
                if mapped and current_json_path in path_options:
                    default_idx = path_options.index(current_json_path)
                else:
                    default_idx = 0

                new_path = st.selectbox(
                    f"JSON path for {cf_path}",
                    path_options,
                    index=default_idx,
                    key=f"path_{country_name}_{cf_path}",
                    label_visibility="collapsed",
                )

            with col_type:
                if new_path != "— unmapped —":
                    type_idx = EXTRACTION_TYPES.index(current_type) if current_type in EXTRACTION_TYPES else 0
                    new_type = st.selectbox(
                        f"Type for {cf_path}",
                        EXTRACTION_TYPES,
                        index=type_idx,
                        key=f"type_{country_name}_{cf_path}",
                        label_visibility="collapsed",
                    )
                else:
                    new_type = None
                    st.caption("—")

            with col_status:
                if new_path == "— unmapped —" and not mapped:
                    st.caption("⬜")
                elif new_path == "— unmapped —" and mapped:
                    st.caption("🗑️")
                    has_changes = True
                elif (new_path != current_json_path) or (new_type and new_type != current_type):
                    st.caption("✏️")
                    has_changes = True
                else:
                    st.caption("✅")

            # Show reasoning if exists
            if current_reasoning and mapped:
                st.caption(f"💡 *{current_reasoning}*")

            # Track edits
            if new_path != "— unmapped —":
                st.session_state[state_key][cf_path] = {
                    "json_path": new_path,
                    "extraction_type": new_type or "direct",
                }
            else:
                st.session_state[state_key].pop(cf_path, None)

            st.markdown("---")

        # Save & reprocess buttons
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            if st.button("💾 Save Rules", type="primary", use_container_width=True):
                # Build updated rules list from session state edits + unchanged rules
                new_rules = []
                all_edits = st.session_state[state_key]

                for cf_info in canonical_fields:
                    cf_path = cf_info["path"]

                    if cf_path in all_edits:
                        edit = all_edits[cf_path]
                        # Check if this was an existing rule or new
                        existing = rules_by_field.get(cf_path, {})
                        new_rules.append({
                            "canonical_field": cf_path,
                            "json_path": edit["json_path"],
                            "extraction_type": edit["extraction_type"],
                            "reasoning": existing.get("reasoning", "User override"),
                        })
                    elif cf_path in rules_by_field:
                        # Keep original rule unchanged
                        new_rules.append(rules_by_field[cf_path])

                # Save
                with open(paths["rules"], 'w', encoding='utf-8') as f:
                    json.dump(new_rules, f, indent=2, ensure_ascii=False)

                st.success(f"Saved {len(new_rules)} rules to {paths['rules']}")

        with col2:
            if st.button("🔄 Re-process Data", use_container_width=True):
                # Load current saved rules
                with open(paths["rules"], 'r', encoding='utf-8') as f:
                    saved_rules = json.load(f)

                with st.spinner(f"Re-processing {country_name} with {len(saved_rules)} rules..."):
                    try:
                        df = reprocess_country(country_name, paths, saved_rules)
                        st.success(f"Re-processed {len(df):,} records → {paths['flat_csv']}")
                    except Exception as e:
                        st.error(f"Error during re-processing: {e}")

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: NORMALIZATION MAPPINGS
    # ══════════════════════════════════════════════════════════════════
    with tab_normalize:
        norm_country = st.selectbox("Select country", list(COUNTRIES.keys()), key="norm_country")
        norm_paths = COUNTRIES[norm_country]

        # Load normalization log
        norm_log_path = NORMALIZED_DIR / 'normalization_log.json'
        if not norm_log_path.exists():
            st.error("Normalization log not found. Run normalize.py first.")
            return

        with open(norm_log_path, 'r', encoding='utf-8') as f:
            all_logs = json.load(f)

        # Find log for selected country
        country_log = None
        for log in all_logs:
            if log.get("country") == norm_country:
                country_log = log
                break

        if not country_log:
            st.warning(f"No normalization log for {norm_country}")
            return

        changes = country_log.get("changes", {})

        # ── Status Mapping ───────────────────────────────────────────
        st.markdown(f"### {norm_country} — Status Mapping")

        status_map = changes.get("status", {})
        if isinstance(status_map, str):
            st.info(f"Status: {status_map}")
        elif isinstance(status_map, dict) and status_map:
            # All possible status codes for dropdown
            status_options = list(STATUS_LABELS.keys())
            status_option_labels = {k: f"{k} — {v}" for k, v in STATUS_LABELS.items()}

            norm_state_key = f"status_edits_{norm_country}"
            if norm_state_key not in st.session_state:
                st.session_state[norm_state_key] = dict(status_map)

            col_raw, col_mapped, col_override = st.columns([3, 2, 3])
            with col_raw:
                st.markdown("**Raw Value**")
            with col_mapped:
                st.markdown("**LLM Mapped To**")
            with col_override:
                st.markdown("**Your Override**")

            for raw_val, mapped_code in sorted(status_map.items(), key=lambda x: x[0].lower()):
                col_raw, col_mapped, col_override = st.columns([3, 2, 3])

                with col_raw:
                    st.markdown(f"`{raw_val}`")

                with col_mapped:
                    label = STATUS_LABELS.get(mapped_code, mapped_code)
                    st.caption(f"{mapped_code} — {label}")

                with col_override:
                    current_override = st.session_state[norm_state_key].get(raw_val, mapped_code)
                    idx = status_options.index(current_override) if current_override in status_options else 0

                    new_code = st.selectbox(
                        f"Override for {raw_val}",
                        status_options,
                        index=idx,
                        format_func=lambda x: status_option_labels.get(x, x),
                        key=f"status_{norm_country}_{raw_val}",
                        label_visibility="collapsed",
                    )
                    st.session_state[norm_state_key][raw_val] = new_code
        else:
            st.info("No status mappings found in log.")

        st.divider()

        # ── Company Type Mapping ─────────────────────────────────────
        st.markdown(f"### {norm_country} — Company Type Mapping")

        type_map = changes.get("company_type", {})
        if isinstance(type_map, str):
            st.info(f"Company type: {type_map}")
        elif isinstance(type_map, dict) and type_map:
            type_options = list(COMPANY_TYPE_LABELS.keys())
            type_option_labels = {k: f"{k} — {v}" for k, v in COMPANY_TYPE_LABELS.items()}

            type_state_key = f"type_edits_{norm_country}"
            if type_state_key not in st.session_state:
                st.session_state[type_state_key] = dict(type_map)

            col_raw, col_mapped, col_override = st.columns([3, 2, 3])
            with col_raw:
                st.markdown("**Raw Value**")
            with col_mapped:
                st.markdown("**LLM Mapped To**")
            with col_override:
                st.markdown("**Your Override**")

            for raw_val, mapped_code in sorted(type_map.items(), key=lambda x: x[0].lower()):
                col_raw, col_mapped, col_override = st.columns([3, 2, 3])

                with col_raw:
                    st.markdown(f"`{raw_val}`")

                with col_mapped:
                    label = COMPANY_TYPE_LABELS.get(mapped_code, mapped_code)
                    st.caption(f"{mapped_code} — {label}")

                with col_override:
                    current_override = st.session_state[type_state_key].get(raw_val, mapped_code)
                    idx = type_options.index(current_override) if current_override in type_options else 0

                    new_code = st.selectbox(
                        f"Override for {raw_val}",
                        type_options,
                        index=idx,
                        format_func=lambda x: type_option_labels.get(x, x),
                        key=f"type_{norm_country}_{raw_val}",
                        label_visibility="collapsed",
                    )
                    st.session_state[type_state_key][raw_val] = new_code
        else:
            st.info("No company type mappings found in log.")

        st.divider()

        # Save & reprocess normalization
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 Save & Re-normalize", type="primary", use_container_width=True, key="save_norm"):
                # Load the flat CSV (pre-normalization)
                if not norm_paths["flat_csv"].exists():
                    st.error("Flat CSV not found. Run auto_mapper.py first.")
                else:
                    df = pd.read_csv(norm_paths["flat_csv"], dtype=str, keep_default_na=False)

                    # Apply status overrides
                    norm_status_key = f"status_edits_{norm_country}"
                    if norm_status_key in st.session_state:
                        updated_status_map = st.session_state[norm_status_key]
                        if 'status' in df.columns:
                            df['status'] = df['status'].map(updated_status_map).fillna(df['status'])

                    # Apply company type overrides
                    norm_type_key = f"type_edits_{norm_country}"
                    if norm_type_key in st.session_state:
                        updated_type_map = st.session_state[norm_type_key]
                        if 'company_type' in df.columns:
                            df['company_type'] = df['company_type'].map(updated_type_map).fillna(df['company_type'])

                    # Apply date normalization
                    if 'registration_date' in df.columns:
                        from normalize import normalize_date, normalize_country, normalize_share_capital
                        df['registration_date'] = df['registration_date'].apply(normalize_date)

                    # Apply country normalization
                    country_cols = [c for c in df.columns if c.startswith('country.')]
                    for col in country_cols:
                        df[col] = df[col].apply(normalize_country)

                    # Apply share capital normalization
                    if 'share_capital.amount' in df.columns:
                        df['share_capital.amount'] = df['share_capital.amount'].apply(normalize_share_capital)

                    # Save
                    NORMALIZED_DIR.mkdir(exist_ok=True, parents=True)
                    output_path = norm_paths["norm_csv"]
                    df.to_csv(output_path, index=False, encoding='utf-8')

                    # Update log
                    for log in all_logs:
                        if log.get("country") == norm_country:
                            if norm_status_key in st.session_state:
                                log["changes"]["status"] = st.session_state[norm_status_key]
                            if norm_type_key in st.session_state:
                                log["changes"]["company_type"] = st.session_state[norm_type_key]
                            log["changes"]["human_reviewed"] = True

                    with open(norm_log_path, 'w', encoding='utf-8') as f:
                        json.dump(all_logs, f, indent=2, ensure_ascii=False, default=str)

                    st.success(f"Saved {len(df):,} records to {output_path}")
                    st.info("Run `python src/build_search_db.py` to rebuild the search database with updated data.")


if __name__ == "__main__":
    main()
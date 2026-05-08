"""
Automatic Schema Mapper: Raw JSON → Canonical Format in one AI step.

This is the core of the thesis. Given a bulk JSON file of company records:
1. Load sample records (picked by sample_picker.py or manually)
2. Send them to Claude Opus 4.5 along with the canonical schema
3. Claude returns extraction rules (JSON path → canonical field)
4. Apply those rules to ALL records with plain Python
5. Save the result as a flat CSV

No hardcoded flattening. No per-country custom code.

Usage (from project root):
    python src/auto_mapper.py
"""

import os
import sys
import json
import time
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Make sure Python can find schema.py in src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import CANONICAL_SCHEMA

# Load API key
load_dotenv()

# ── LLM Setup (Claude Opus 4.5) ──────────────────────────────────────────────
#
# Why Opus 4.5:
#   - Anthropic's most capable model as of late 2025.
#   - $5/$25 per MTok input/output.
#   - Schema mapping is a structured-reasoning task that benefits from
#     Opus-class intelligence — fewer hallucinated paths, better coverage
#     on the first pass, and more reliable JSON output.
#   - No fallback chain: if Opus fails, we want to know about it rather
#     than silently degrade to a weaker model.
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


def get_simplified_schema_for_prompt():
    """
    Create a simplified version of the schema for the LLM prompt.
    Only includes what the LLM needs to understand each field.
    """
    simplified = {}
    for field_name, field_info in CANONICAL_SCHEMA.items():
        entry = {
            "description": field_info["description"],
            "data_type": field_info["data_type"],
        }
        if "synonyms" in field_info:
            entry["synonyms"] = field_info["synonyms"]
        if field_info["data_type"] == "object" and "schema" in field_info:
            entry["sub_fields"] = {
                k: v["description"]
                for k, v in field_info["schema"].items()
            }
        return_examples = field_info.get("examples", [])
        if return_examples:
            entry["examples"] = return_examples[:3]
        simplified[field_name] = entry
    return simplified


def extract_rules_with_llm(sample_records, country_name="unknown"):
    """
    Send sample raw JSON records to Claude and ask it to produce extraction rules.
    """
    schema_for_prompt = get_simplified_schema_for_prompt()

    # Truncate very long text fields in samples to save tokens
    def truncate_record(record, max_str_len=200, max_list_items=3):
        if isinstance(record, dict):
            return {k: truncate_record(v, max_str_len, max_list_items) for k, v in record.items()}
        elif isinstance(record, list):
            truncated = [truncate_record(item, max_str_len, max_list_items) for item in record[:max_list_items]]
            if len(record) > max_list_items:
                truncated.append(f"... ({len(record) - max_list_items} more items)")
            return truncated
        elif isinstance(record, str) and len(record) > max_str_len:
            return record[:max_str_len] + "..."
        return record

    truncated_samples = [truncate_record(r) for r in sample_records]

    prompt = f"""You are a data integration expert. You are given sample JSON records from a company registry dataset ("{country_name}") and a target canonical schema.

Your task: produce EXTRACTION RULES that map paths in the raw JSON to canonical schema fields.

**Sample Records (showing {len(truncated_samples)} representative records):**
{json.dumps(truncated_samples, indent=2, ensure_ascii=False)}

**Canonical Schema (target fields):**
{json.dumps(schema_for_prompt, indent=2, ensure_ascii=False)}

**Instructions:**
1. Analyze the JSON structure and figure out where each piece of company data lives.
2. For EACH canonical field (including sub-fields of object types), produce an extraction rule.
3. Consider all levels of nesting — data might be inside nested dicts, arrays of dicts, etc.
4. If a canonical field has no matching data in the JSON, skip it (don't produce a rule).
5. For the "country" field, determine the ISO 3166-1 alpha-2 code from context and hardcode it.
6. For "company_type", ALWAYS prefer human-readable descriptions/labels over internal codes.
   For example, if the data has both a code ("STI") and a description ("Stiftelse"), use the DESCRIPTION.
   The description gives much better context for downstream normalization.

**CRITICAL PATH FORMAT RULES:**
- Use ONLY dot notation for nested objects: organisasjonsform.beskrivelse
- For array access, use [0] for first element: Información General[0].Razón Social
- For collecting from all array elements, use [*]: Activities[*].CorpActivityType
- Field names with spaces need NO quotes: Información General[0].Razón Social (NOT "Razón Social")
- For fallback chains, provide json_path as a JSON array: ["path1", "path2"]

**Extraction Types:**
- "direct": Simple path like Corp.CompanyName — just navigate the keys
- "nested_field": Path through nested dicts, like organisasjonsform.beskrivelse
- "first_element_field": First element of array then a field, like Información General[0].Razón Social
- "join_list": Collect a field from all array elements and join with " | ", like Activities[*].CorpActivityType. Also works for joining a list of strings, like aktivitet[*]
- "boolean_logic": Derive value from boolean flags. Format: field=value->result|field=value->result|else->default
- "hardcode": A fixed value, e.g. country code
- "fallback_chain": Try multiple paths in order, use first non-empty. json_path MUST be a JSON array of path strings.

**Output Format (JSON only, no markdown, no extra text):**
{{
  "rules": [
    {{
      "canonical_field": "company_name.legal_name",
      "json_path": "Corp.CompanyName",
      "extraction_type": "direct",
      "reasoning": "Brief explanation"
    }},
    {{
      "canonical_field": "industry",
      "json_path": "Activities[*].CorpActivityType",
      "extraction_type": "join_list",
      "reasoning": "Multiple activity codes, join them"
    }},
    {{
      "canonical_field": "country",
      "json_path": "MM",
      "extraction_type": "hardcode",
      "reasoning": "Myanmar data"
    }},
    {{
      "canonical_field": "status",
      "json_path": "konkurs=true->Bankrupt|underAvvikling=true->Under liquidation|else->Active",
      "extraction_type": "boolean_logic",
      "reasoning": "Status derived from boolean flags"
    }},
    {{
      "canonical_field": "description",
      "json_path": ["vedtektsfestetFormaal[*]", "aktivitet[*]"],
      "extraction_type": "fallback_chain",
      "reasoning": "Try purpose text first, fall back to activity text"
    }}
  ]
}}

Respond with ONLY valid JSON. No markdown code fences. No extra text."""

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
            rules = parsed.get("rules", parsed)
            if isinstance(rules, dict) and "rules" in rules:
                rules = rules["rules"]

            print(f"  LLM returned {len(rules)} extraction rules")
            return rules

        except json.JSONDecodeError:
            if attempt == 0:
                print(f"  JSON parse error — retrying once")
                continue
            print(f"  JSON parse error on retry — giving up on this call")
            return []

        except Exception as e:
            err = str(e)
            err_lower = err.lower()
            if "429" in err or "rate_limit" in err_lower:
                print(f"  Rate-limited. Waiting 60s and retrying...")
                time.sleep(60)
                if attempt == 0:
                    continue
                return []
            if "401" in err or "authentication" in err_lower or "invalid_api_key" in err_lower:
                print(f"  !! Authentication error: {err[:200]}")
                raise
            print(f"  Error: {err[:200]}")
            return []

    return []


def get_all_canonical_subfields():
    """
    Get a flat list of all canonical field paths including sub-fields.
    E.g., "company_name.legal_name", "share_capital.amount", "status", etc.
    """
    fields = []
    for field_name, field_info in CANONICAL_SCHEMA.items():
        if field_info["data_type"] == "object" and "schema" in field_info:
            for sub_name in field_info["schema"]:
                fields.append(f"{field_name}.{sub_name}")
        else:
            fields.append(field_name)
    return fields


def find_missing_fields(rules):
    """
    Compare the rules produced by Pass 1 against all canonical sub-fields.
    Return a list of canonical fields that have no rule.
    """
    mapped = {rule["canonical_field"] for rule in rules}
    all_fields = get_all_canonical_subfields()
    return [f for f in all_fields if f not in mapped]


def second_pass_fill_gaps(sample_records, existing_rules, missing_fields, country_name="unknown"):
    """
    PASS 2: Ask the LLM to specifically look for data matching the missed canonical fields.
    
    This combats non-determinism by explicitly drawing attention to fields the first
    pass missed, and showing the LLM which fields were already mapped (so it doesn't
    duplicate work).
    """
    if not missing_fields:
        return []

    schema_for_prompt = get_simplified_schema_for_prompt()

    # Build a focused schema showing only the missing fields
    missing_schema = {}
    for mf in missing_fields:
        parts = mf.split('.', 1)
        parent = parts[0]
        if parent in schema_for_prompt:
            if parent not in missing_schema:
                missing_schema[parent] = schema_for_prompt[parent]

    # Show what's already mapped so the LLM doesn't duplicate
    already_mapped = []
    for rule in existing_rules:
        already_mapped.append(f"  {rule['canonical_field']} <- {rule['extraction_type']}: {rule['json_path']}")
    already_mapped_str = '\n'.join(already_mapped)

    # Truncate samples
    def truncate_record(record, max_str_len=200, max_list_items=3):
        if isinstance(record, dict):
            return {k: truncate_record(v, max_str_len, max_list_items) for k, v in record.items()}
        elif isinstance(record, list):
            truncated = [truncate_record(item, max_str_len, max_list_items) for item in record[:max_list_items]]
            if len(record) > max_list_items:
                truncated.append(f"... ({len(record) - max_list_items} more items)")
            return truncated
        elif isinstance(record, str) and len(record) > max_str_len:
            return record[:max_str_len] + "..."
        return record

    truncated_samples = [truncate_record(r) for r in sample_records]

    prompt = f"""You are a data integration expert doing a SECOND PASS review.

In a first pass, we mapped most fields from a "{country_name}" company registry dataset to our canonical schema. But some canonical fields were NOT mapped. Your job is to look AGAIN at the raw data and determine if any of the missing fields CAN be mapped.

**Already mapped (DO NOT duplicate these):**
{already_mapped_str}

**Missing canonical fields that need mapping (if data exists):**
{json.dumps(missing_fields, indent=2)}

**Schema definitions for the missing fields:**
{json.dumps(missing_schema, indent=2, ensure_ascii=False)}

**Sample Records:**
{json.dumps(truncated_samples, indent=2, ensure_ascii=False)}

**Instructions:**
1. For EACH missing field, carefully search the sample data for ANY field that could contain this information.
2. Look at field names, values, nested objects, and arrays — the data might be in an unexpected place.
3. Even if the value is 0 or appears empty in some records, if the field EXISTS in the JSON structure, produce a rule for it.
4. If you genuinely cannot find a matching source for a field, skip it.
5. Use the same path format rules and extraction types as before.

**CRITICAL PATH FORMAT RULES:**
- Use ONLY dot notation for nested objects
- For array access, use [0] for first element, [*] for all elements
- Field names with spaces need NO quotes
- For fallback chains, provide json_path as a JSON array

**Output Format (JSON only, no markdown, no extra text):**
{{
  "rules": [
    {{
      "canonical_field": "share_capital.amount",
      "json_path": "Corp.ShareCapitalValue",
      "extraction_type": "direct",
      "reasoning": "ShareCapitalValue contains the capital amount, even though it shows as 0 in some samples"
    }}
  ]
}}

If NO additional rules can be produced, return: {{"rules": []}}

Respond with ONLY valid JSON. No markdown code fences. No extra text."""

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
            rules = parsed.get("rules", parsed)
            if isinstance(rules, dict) and "rules" in rules:
                rules = rules["rules"]

            # Filter out any rules that duplicate already-mapped fields
            existing_fields = {r["canonical_field"] for r in existing_rules}
            new_rules = [r for r in rules if r["canonical_field"] not in existing_fields]

            print(f"  Pass 2 returned {len(new_rules)} new rules")
            return new_rules

        except json.JSONDecodeError:
            if attempt == 0:
                print(f"  Pass 2: JSON parse error — retrying once")
                continue
            print(f"  Pass 2: JSON parse error on retry — giving up")
            return []

        except Exception as e:
            err = str(e)
            err_lower = err.lower()
            if "429" in err or "rate_limit" in err_lower:
                print(f"  Rate-limited. Waiting 60s and retrying...")
                time.sleep(60)
                if attempt == 0:
                    continue
                return []
            if "401" in err or "authentication" in err_lower:
                print(f"  !! Authentication error: {err[:200]}")
                raise
            print(f"  Pass 2 error: {err[:200]}")
            return []

    return []


# ── Robust Extraction Engine ─────────────────────────────────────────────────


def parse_path(path_str):
    """
    Parse a JSON path string into a list of navigation steps.
    
    Handles:
        "Corp.CompanyName"                      -> ["Corp", "CompanyName"]
        "Información General[0].Razón Social"   -> ["Información General", 0, "Razón Social"]
        "Activities[*].CorpActivityType"         -> ["Activities", "*", "CorpActivityType"]
        "organisasjonsform.beskrivelse"          -> ["organisasjonsform", "beskrivelse"]
        "forretningsadresse.adresse"             -> ["forretningsadresse", "adresse"]
        'Información General[0]."Razón Social"' -> ["Información General", 0, "Razón Social"]
        "aktivitet[*]"                           -> ["aktivitet", "*"]
    
    Strategy: split on [ ] and . intelligently, handling field names with spaces.
    """
    # Remove surrounding quotes if present
    path_str = path_str.strip().strip("'\"")
    
    steps = []
    # Tokenize: split on [N], [*], and . but keep field names with spaces intact
    # We process character by character to handle dots in field names correctly
    
    # First, handle array accessors [0], [*], etc. by replacing them with a separator
    # Replace [N] and [*] with a unique delimiter
    DELIM = '\x00'
    processed = re.sub(r'\[(\d+)\]', lambda m: f'{DELIM}{m.group(1)}{DELIM}', path_str)
    processed = re.sub(r'\[\*\]', f'{DELIM}*{DELIM}', processed)
    
    # Now split on the delimiter and dots, but we need to be smart about dots
    # The challenge: "Información General" contains a space, and "forretningsadresse.adresse"
    # uses a dot as separator. We need to figure out when a dot is a separator vs part of a name.
    
    # Strategy: split on DELIM first, then for each segment, try to navigate
    # by splitting on dots and seeing if keys exist. But we don't have the record here.
    # So we split on dots and will try each segment as a key during navigation.
    
    parts = processed.split(DELIM)
    for part in parts:
        part = part.strip('.').strip()
        if not part:
            continue
        if part == '*':
            steps.append('*')
        elif part.isdigit():
            steps.append(int(part))
        else:
            # Remove quotes around field names
            part = part.strip('"').strip("'")
            # Split on dots for nested access
            for sub in part.split('.'):
                sub = sub.strip().strip('"').strip("'")
                if sub:
                    steps.append(sub)
    
    return steps


def navigate(record, steps):
    """
    Navigate a record using parsed path steps.
    Returns the value found, or None if path doesn't exist.
    
    When encountering a key that doesn't exist, tries combining it with
    the next step(s) to handle field names containing dots.
    """
    current = record
    i = 0
    
    while i < len(steps):
        step = steps[i]
        
        if current is None:
            return None
        
        if step == '*':
            # Wildcard: current must be a list, return it for the caller to iterate
            if isinstance(current, list):
                # Collect from remaining path for each element
                remaining = steps[i+1:]
                if remaining:
                    results = []
                    for item in current:
                        val = navigate(item, remaining)
                        if val is not None:
                            results.append(val)
                    return results
                else:
                    return current
            return None
        
        elif isinstance(step, int):
            # Array index
            if isinstance(current, list) and step < len(current):
                current = current[step]
            else:
                return None
        
        elif isinstance(current, dict):
            if step in current:
                current = current[step]
            else:
                # Try combining this step with next steps to handle
                # field names that might contain dots
                # e.g., if the key is "Razón Social" but was split from a dot path
                found = False
                for lookahead in range(1, min(4, len(steps) - i)):
                    combined = '.'.join(str(s) for s in steps[i:i+lookahead+1])
                    if combined in current:
                        current = current[combined]
                        i += lookahead  # Skip the extra steps we consumed
                        found = True
                        break
                    # Also try with spaces instead of dots
                    combined_space = ' '.join(str(s) for s in steps[i:i+lookahead+1])
                    if combined_space in current:
                        current = current[combined_space]
                        i += lookahead
                        found = True
                        break
                
                if not found:
                    return None
        else:
            return None
        
        i += 1
    
    return current


def extract_value(record, path_str):
    """
    Extract a value from a record using a path string.
    Handles all path formats including arrays, wildcards, and nested access.
    
    Returns the extracted value (may be a list for wildcard paths).
    """
    steps = parse_path(path_str)
    return navigate(record, steps)


def apply_rule(record, rule):
    """
    Apply a single extraction rule to a record and return the extracted value.
    """
    extraction_type = rule.get("extraction_type", "direct")
    json_path = rule.get("json_path", "")

    try:
        if extraction_type in ("direct", "nested_field"):
            value = extract_value(record, json_path)
            return _clean_value(value)

        elif extraction_type == "first_element_field":
            value = extract_value(record, json_path)
            return _clean_value(value)

        elif extraction_type == "join_list":
            value = extract_value(record, json_path)
            if isinstance(value, list):
                # Flatten and join
                str_values = []
                for v in value:
                    cleaned = _clean_value(v)
                    if cleaned and cleaned != '':
                        str_values.append(str(cleaned))
                return ' | '.join(str_values)
            return _clean_value(value)

        elif extraction_type == "boolean_logic":
            parts = json_path.split('|')
            for part in parts:
                part = part.strip()
                if part.startswith('else->'):
                    return part.split('->', 1)[1].strip()
                if '->' in part:
                    condition, result = part.split('->', 1)
                    condition = condition.strip()
                    result = result.strip()
                    if '==' in condition:
                        # Handle "field == value" (with double equals)
                        field, expected = condition.split('==', 1)
                        field = field.strip()
                        expected = expected.strip().lower().strip('"').strip("'")
                        actual = extract_value(record, field)
                        if str(actual).lower().strip() == expected:
                            return result
                    elif '=' in condition:
                        field, expected = condition.split('=', 1)
                        field = field.strip()
                        expected = expected.strip().lower()
                        actual = extract_value(record, field)
                        if str(actual).lower().strip() == expected:
                            return result
            return ''

        elif extraction_type == "hardcode":
            return json_path

        elif extraction_type == "fallback_chain":
            # json_path should be a list of paths
            if isinstance(json_path, list):
                paths = json_path
            elif isinstance(json_path, str):
                # Try to parse as JSON array
                try:
                    paths = json.loads(json_path)
                except:
                    paths = [p.strip() for p in json_path.split('||')]
            else:
                paths = [str(json_path)]

            for path in paths:
                path = path.strip().strip('"').strip("'")
                value = extract_value(record, path)
                
                # If it's a list (from [*] wildcard), join it
                if isinstance(value, list):
                    str_values = []
                    for v in value:
                        cleaned = _clean_value(v)
                        if cleaned and cleaned != '':
                            str_values.append(str(cleaned))
                    if str_values:
                        return ' '.join(str_values)
                    continue  # Try next fallback
                
                cleaned = _clean_value(value)
                if cleaned and cleaned != '':
                    return cleaned
            
            return ''

        else:
            # Unknown type, try generic extraction
            value = extract_value(record, json_path)
            return _clean_value(value)

    except Exception as e:
        return ''


def _clean_value(value):
    """Clean an extracted value."""
    if value is None:
        return ''
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s in ('', '-', 'None', 'null', 'N/A'):
            return ''
        return s
    if isinstance(value, list):
        str_vals = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return ' '.join(str_vals) if str_vals else ''
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def apply_rules_to_all(data, rules):
    """
    Apply extraction rules to ALL records in the dataset.
    Returns a list of flat dictionaries (one per record).
    """
    records = []

    for i, raw_record in enumerate(data):
        flat = {}
        for rule in rules:
            canonical_field = rule["canonical_field"]
            value = apply_rule(raw_record, rule)
            flat[canonical_field] = value
        records.append(flat)

        if (i + 1) % 10000 == 0:
            print(f"    Processed {i+1:,} / {len(data):,} records...")

    return records


def process_country(country_name, raw_json_path, samples_json_path, output_csv_path):
    """
    Full pipeline for one country:
    1. Load samples
    2. Get extraction rules from LLM
    3. Apply rules to all records
    4. Save as CSV
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {country_name}")
    print(f"{'='*60}")

    # Load sample records
    print(f"  Loading samples from: {samples_json_path}")
    with open(samples_json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} sample records")

    # Get extraction rules from LLM
    print(f"\n  Asking Claude for extraction rules...")
    rules = extract_rules_with_llm(samples, country_name=country_name)

    if not rules:
        print(f"  ERROR: No rules returned for {country_name}")
        return None

    # Display Pass 1 rules
    print(f"\n  Pass 1 Extraction Rules:")
    print(f"  {'-'*55}")
    for rule in rules:
        cf = rule.get('canonical_field', '?')
        jp = rule.get('json_path', '?')
        et = rule.get('extraction_type', '?')
        reasoning = rule.get('reasoning', '')
        jp_str = str(jp)[:60]
        print(f"  {cf:40s} <- {et}: {jp_str}")
        print(f"    {reasoning}")

    # ── PASS 2: Fill gaps ────────────────────────────────────────────────
    missing = find_missing_fields(rules)
    if missing:
        print(f"\n  Pass 2: {len(missing)} canonical fields not yet mapped:")
        for mf in missing:
            print(f"    - {mf}")
        print(f"\n  Asking Claude to look again for these fields...")
        time.sleep(5)  # Rate limit buffer

        new_rules = second_pass_fill_gaps(samples, rules, missing, country_name=country_name)

        if new_rules:
            print(f"\n  Pass 2 found {len(new_rules)} additional rules:")
            for rule in new_rules:
                cf = rule.get('canonical_field', '?')
                jp = rule.get('json_path', '?')
                et = rule.get('extraction_type', '?')
                reasoning = rule.get('reasoning', '')
                jp_str = str(jp)[:60]
                print(f"  {cf:40s} <- {et}: {jp_str}")
                print(f"    {reasoning}")
            rules.extend(new_rules)
        else:
            print(f"  Pass 2: No additional rules found.")

        # Check what's still missing after both passes
        still_missing = find_missing_fields(rules)
        if still_missing:
            print(f"\n  Still unmapped after 2 passes ({len(still_missing)}):")
            for mf in still_missing:
                print(f"    - {mf}")
        else:
            print(f"\n  All canonical fields now have rules!")
    else:
        print(f"\n  All canonical fields mapped in Pass 1 — no Pass 2 needed.")

    # Save rules for reference
    rules_path = output_csv_path.replace('.csv', '_rules.json')
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    print(f"\n  Rules saved to: {rules_path}")

    # --- Quick validation: apply rules to first sample and check ---
    print(f"\n  Validating rules against first sample...")
    test_result = {}
    for rule in rules:
        cf = rule["canonical_field"]
        val = apply_rule(samples[0], rule)
        test_result[cf] = val
        val_str = str(val)[:60] if val else "(empty)"
        status = "OK" if val else "EMPTY"
        print(f"    [{status:5s}] {cf:40s}: {val_str}")

    empty_count = sum(1 for v in test_result.values() if not v or v == '')
    total_count = len(test_result)
    print(f"\n  Validation: {total_count - empty_count}/{total_count} fields have data in sample 1")

    # Load ALL records
    print(f"\n  Loading full dataset: {raw_json_path}")
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    print(f"  Loaded {len(all_data):,} records")

    # Apply rules to all records
    print(f"  Applying extraction rules to all records...")
    flat_records = apply_rules_to_all(all_data, rules)

    # Convert to DataFrame and save
    df = pd.DataFrame(flat_records)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n  Saved {len(df):,} records to: {output_csv_path}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")

    # Show non-empty stats
    print(f"\n  Data coverage:")
    for col in df.columns:
        non_empty = df[col].apply(lambda x: x != '' and x is not None and pd.notna(x)).sum()
        pct = non_empty / len(df) * 100
        print(f"    {col:40s}: {non_empty:>6,} / {len(df):,} ({pct:.1f}%)")

    return df


def main():
    """
    Run the full auto-mapping pipeline for all 3 countries.
    Prerequisites: Run sample_picker.py first to generate data/samples/*.json
    """
    raw_dir = Path('data/raw')
    samples_dir = Path('data/samples')
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)

    countries = {
        "Myanmar": {
            "raw": raw_dir / 'mm_entities.json',
            "samples": samples_dir / 'myanmar_samples.json',
            "output": output_dir / 'myanmar_flat.csv',
        },
        "Norway": {
            "raw": raw_dir / 'no_entities.json',
            "samples": samples_dir / 'norway_samples.json',
            "output": output_dir / 'norway_flat.csv',
        },
        "Honduras": {
            "raw": raw_dir / 'hn_entities.json',
            "samples": samples_dir / 'honduras_samples.json',
            "output": output_dir / 'honduras_flat.csv',
        },
    }

    for country_name, paths in countries.items():
        if not paths["raw"].exists():
            print(f"\n  SKIP {country_name}: raw file not found at {paths['raw']}")
            continue
        if not paths["samples"].exists():
            print(f"\n  SKIP {country_name}: samples not found at {paths['samples']}")
            print(f"  Run 'python src/sample_picker.py' first!")
            continue

        df = process_country(
            country_name,
            str(paths["raw"]),
            str(paths["samples"]),
            str(paths["output"])
        )

        # Wait between API calls
        print(f"\n  Waiting 10 seconds before next country...")
        time.sleep(10)

    print(f"\n{'='*60}")
    print("All countries processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
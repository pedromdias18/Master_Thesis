"""
Ablation Study: Experiment Runner

Runs 4 experiments across 3 countries to determine which architectural
components are necessary for reliable LLM-based schema mapping.

Experiments:
  E0 (baseline): two_pass=ON,  sample_picker=ON,  rich_schema=ON
  E1 (no 2pass): two_pass=OFF, sample_picker=ON,  rich_schema=ON
  E2 (no picker): two_pass=ON, sample_picker=OFF, rich_schema=ON
  E3 (min schema): two_pass=ON, sample_picker=ON, rich_schema=OFF

RQ2: Which architectural components are necessary for reliable and
     cost-efficient LLM-based schema mapping across heterogeneous registries?

Usage (from project root):
    python src/run_experiments.py              # run all experiments (this session)
    python src/run_experiments.py E1           # run only experiment E1
    python src/run_experiments.py E0 E3        # run E0 and E3
    python src/run_experiments.py aggregate    # read results/*.json from all sessions
                                               # and produce the consolidated summary
                                               # + failure analysis for the thesis
"""

import os
import sys
import json
import time
import random
import copy
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import CANONICAL_SCHEMA

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR = Path('data/raw')
SAMPLES_DIR = Path('data/samples')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

COUNTRIES = {
    "Myanmar": {
        "raw": RAW_DIR / 'mm_entities.json',
        "samples": SAMPLES_DIR / 'myanmar_samples.json',
    },
    "Norway": {
        "raw": RAW_DIR / 'no_entities.json',
        "samples": SAMPLES_DIR / 'norway_samples.json',
    },
    "Honduras": {
        "raw": RAW_DIR / 'hn_entities.json',
        "samples": SAMPLES_DIR / 'honduras_samples.json',
    },
}

EXPERIMENTS = {
    "E0": {
        "name": "E0 — Baseline (all ON)",
        "two_pass": True,
        "use_sample_picker": True,
        "rich_schema": True,
    },
    "E1": {
        "name": "E1 — No two-pass",
        "two_pass": False,
        "use_sample_picker": True,
        "rich_schema": True,
    },
    "E2": {
        "name": "E2 — Random samples (no picker)",
        "two_pass": True,
        "use_sample_picker": False,
        "rich_schema": True,
    },
    "E3": {
        "name": "E3 — Minimal schema (no descriptions/synonyms/examples)",
        "two_pass": True,
        "use_sample_picker": True,
        "rich_schema": False,
    },
}


# ── Schema Variants ──────────────────────────────────────────────────────────

def get_rich_schema():
    """Full schema with descriptions, examples, synonyms — the default."""
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
        examples = field_info.get("examples", [])
        if examples:
            entry["examples"] = examples[:3]
        simplified[field_name] = entry
    return simplified


def get_minimal_schema():
    """Stripped schema: field names + data types only. No descriptions, synonyms, examples."""
    simplified = {}
    for field_name, field_info in CANONICAL_SCHEMA.items():
        entry = {
            "data_type": field_info["data_type"],
        }
        if field_info["data_type"] == "object" and "schema" in field_info:
            entry["sub_fields"] = list(field_info["schema"].keys())
        simplified[field_name] = entry
    return simplified


# ── Sample Selection ─────────────────────────────────────────────────────────

def get_random_samples(raw_json_path, n_samples=3, seed=42):
    """Pick N random records (reproducible with seed)."""
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    random.seed(seed)
    indices = random.sample(range(len(data)), n_samples)
    samples = [data[i] for i in indices]
    print(f"    Random samples picked: indices {indices}")
    return samples


def get_picked_samples(samples_json_path):
    """Load the pre-selected coverage-optimized samples."""
    with open(samples_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── LLM Calls (Claude Opus 4.5) ────────────────────────────────────────────
#
# What changed vs. the Gemini version:
#   - Dropped the 13s global throttle: unnecessary at 50 RPM.
#   - Dropped the daily budget guard: there's no RPD cap on Anthropic's API.

CLAUDE_MODEL = "claude-sonnet-4-5"   # Pinned snapshot for reproducibility.
                                               # Rolling alias would be "claude-opus-4-5".
CLAUDE_MAX_TOKENS = 4096             # comfortably above the rule-list outputs we see.

# Per-million-token pricing for cost tracking (does not affect billing).
COST_INPUT_PER_MTOK = 5.00
COST_OUTPUT_PER_MTOK = 25.00

# Lazy client init so `aggregate` mode works without the anthropic package.
_claude_client = None
_calls_made = 0
_total_input_tokens = 0
_total_output_tokens = 0


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        _claude_client = anthropic.Anthropic(api_key=api_key)
    return _claude_client


def _estimated_cost_so_far():
    in_cost = (_total_input_tokens / 1_000_000) * COST_INPUT_PER_MTOK
    out_cost = (_total_output_tokens / 1_000_000) * COST_OUTPUT_PER_MTOK
    return in_cost + out_cost


def call_gemini(prompt):
    """Call Claude with one JSON-parse retry. Returns parsed rules list.

    (Function name kept for compatibility with the rest of the script — it's
    now calling Claude, not Gemini.)

    Error handling is simpler than the Gemini version because Anthropic's API
    is much more reliable on this tier: no 20-RPD ceiling, far fewer 503s,
    and the official SDK already handles transient retries internally via
    tenacity. We just need to cope with occasional malformed JSON output.
    """
    global _calls_made, _total_input_tokens, _total_output_tokens

    for attempt in range(5):  # first try + up to 4 JSON-parse retries
        _calls_made += 1
        print(f"    [claude call #{_calls_made}]  (running cost: ${_estimated_cost_so_far():.4f})")

        try:
            response = _get_claude_client().messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            # Track usage for cost sanity check.
            _total_input_tokens += response.usage.input_tokens
            _total_output_tokens += response.usage.output_tokens

            # Anthropic returns a list of content blocks; for plain text the
            # first block is what we want.
            result_text = response.content[0].text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('\n', 1)[1]
                result_text = result_text.rsplit('```', 1)[0]
                result_text = result_text.strip()

            parsed = json.loads(result_text)
            rules = parsed.get("rules", parsed)
            if isinstance(rules, dict) and "rules" in rules:
                rules = rules["rules"]
            return rules

        except json.JSONDecodeError:
            if attempt < 4:
                print(f"    JSON parse error — retrying (attempt {attempt + 2}/5)")
                continue
            print(f"    JSON parse error after 5 attempts — giving up on this call")
            return []

        except Exception as e:
            err = str(e)
            err_lower = err.lower()

            # Rate-limit (should basically never happen on tier 1 with this
            # script, but guard anyway).
            if "429" in err or "rate_limit" in err_lower:
                print(f"    Rate-limited. Waiting 60s and retrying (attempt {attempt + 2}/5)...")
                time.sleep(60)
                if attempt < 4:
                    continue
                return []

            # Authentication — stop immediately, no point burning more calls.
            if "401" in err or "authentication" in err_lower or "invalid_api_key" in err_lower:
                print(f"    !! Authentication error: {err[:200]}")
                print(f"    Check ANTHROPIC_API_KEY in your .env file.")
                raise

            # Anything else: log and return empty for this call.
            print(f"    Error: {err[:200]}")
            return []

    return []


def truncate_record(record, max_str_len=200, max_list_items=3):
    """Truncate long values in a record to save tokens."""
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


# ── Core Extraction ──────────────────────────────────────────────────────────

def build_extraction_prompt(sample_records, schema_for_prompt, country_name):
    """Build the extraction prompt, used by both passes."""
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

**CRITICAL PATH FORMAT RULES:**
- Use ONLY dot notation for nested objects: organisasjonsform.beskrivelse
- For array access, use [0] for first element: Información General[0].Razón Social
- For collecting from all array elements, use [*]: Activities[*].CorpActivityType
- Field names with spaces need NO quotes
- For fallback chains, provide json_path as a JSON array: ["path1", "path2"]

**Extraction Types:**
- "direct": Simple path like Corp.CompanyName
- "nested_field": Path through nested dicts, like organisasjonsform.beskrivelse
- "first_element_field": First element of array then a field, like Información General[0].Razón Social
- "join_list": Collect a field from all array elements and join with " | "
- "boolean_logic": Derive value from boolean flags. Format: field=value->result|field=value->result|else->default
- "hardcode": A fixed value, e.g. country code
- "fallback_chain": Try multiple paths in order, use first non-empty. json_path MUST be a JSON array.

**Output Format (JSON only, no markdown, no extra text):**
{{
  "rules": [
    {{
      "canonical_field": "company_name.legal_name",
      "json_path": "Corp.CompanyName",
      "extraction_type": "direct",
      "reasoning": "Brief explanation"
    }}
  ]
}}

Respond with ONLY valid JSON. No markdown code fences. No extra text."""

    return prompt


def get_all_canonical_subfields():
    """Get flat list of all canonical field paths."""
    fields = []
    for field_name, field_info in CANONICAL_SCHEMA.items():
        if field_info["data_type"] == "object" and "schema" in field_info:
            for sub_name in field_info["schema"]:
                fields.append(f"{field_name}.{sub_name}")
        else:
            fields.append(field_name)
    return fields


def find_missing_fields(rules):
    """Find canonical fields with no rule."""
    mapped = {rule["canonical_field"] for rule in rules}
    return [f for f in get_all_canonical_subfields() if f not in mapped]


def second_pass(sample_records, existing_rules, missing_fields, schema_for_prompt, country_name):
    """Pass 2: Ask LLM about specific missed fields."""
    if not missing_fields:
        return []

    already_mapped = [
        {"canonical_field": r["canonical_field"], "json_path": r["json_path"]}
        for r in existing_rules
    ]

    truncated_samples = [truncate_record(r) for r in sample_records]

    prompt = f"""You are a data integration expert doing a SECOND PASS review.

In a first pass, you mapped most fields from a "{country_name}" company registry dataset.
But these canonical fields were NOT mapped:
{json.dumps(missing_fields, indent=2)}

**Already mapped (DO NOT duplicate these):**
{json.dumps(already_mapped, indent=2)}

**Sample Records:**
{json.dumps(truncated_samples, indent=2, ensure_ascii=False)}

**Schema for missed fields:**
{json.dumps(schema_for_prompt, indent=2, ensure_ascii=False)}

Look CAREFULLY at the sample data. For each missed field, check if ANY data exists
that could map to it — even if values are 0, empty arrays, or uncommon fields.
If the data genuinely doesn't exist, don't force a mapping.

Return ONLY valid JSON with any new rules found:
{{
  "rules": [
    {{
      "canonical_field": "...",
      "json_path": "...",
      "extraction_type": "...",
      "reasoning": "..."
    }}
  ]
}}"""

    return call_gemini(prompt)


# ── Evaluation (imported from evaluate_mapping.py) ───────────────────────────

from evaluate_mapping import GROUND_TRUTH, evaluate_country  # type: ignore


# ── Failure Analysis ─────────────────────────────────────────────────────────

def failure_analysis(baseline_results, ablated_results, experiment_id):
    """
    Compare ablated results against baseline to find what broke.
    Returns a list of field-level differences.
    """
    failures = []

    for b_detail in baseline_results["details"]:
        cf = b_detail["canonical_field"]

        # Find matching detail in ablated
        a_detail = None
        for d in ablated_results["details"]:
            if d["canonical_field"] == cf:
                a_detail = d
                break

        if a_detail is None:
            # Field completely missing from ablated results
            if b_detail.get("status") not in ("OPTIONAL_SKIP",):
                failures.append({
                    "field": cf,
                    "baseline_status": b_detail.get("status"),
                    "ablated_status": "MISSING_FROM_EVAL",
                    "error_type": "Missing",
                    "explanation": "Field not in ablated evaluation",
                })
            continue

        b_status = b_detail.get("status", "")
        a_status = a_detail.get("status", "")

        # Skip if both are optional skips
        if b_status == "OPTIONAL_SKIP" and a_status == "OPTIONAL_SKIP":
            continue

        # Detect regressions
        if b_status == "CORRECT" and a_status != "CORRECT":
            error_type = "Missing" if a_status == "MISSING" else "Wrong path" if a_status == "TYPE_OK" else "Wrong type" if a_status == "PATH_OK" else "Wrong"

            explanation = ""
            if a_status == "MISSING":
                explanation = "LLM did not produce a rule for this field"
            elif a_status in ("PATH_OK", "TYPE_OK", "WRONG"):
                b_path = b_detail.get("llm_path", "?")
                a_path = a_detail.get("llm_path", "?")
                b_type = b_detail.get("llm_type", "?")
                a_type = a_detail.get("llm_type", "?")
                if str(b_path) != str(a_path):
                    explanation = f"Path changed: {b_path} → {a_path}"
                if b_type != a_type:
                    explanation += f" Type changed: {b_type} → {a_type}"
                explanation = explanation.strip()

            failures.append({
                "field": cf,
                "baseline_status": b_status,
                "ablated_status": a_status,
                "error_type": error_type,
                "explanation": explanation,
            })

    return failures


# ── Run One Experiment ───────────────────────────────────────────────────────

def run_experiment(experiment_id, config, country_name, country_paths):
    """Run a single experiment for a single country. Returns result dict."""
    print(f"\n  {'─'*50}")
    print(f"  {config['name']} — {country_name}")
    print(f"  {'─'*50}")
    print(f"    two_pass={config['two_pass']}, sample_picker={config['use_sample_picker']}, rich_schema={config['rich_schema']}")

    # 1. Select samples
    if config["use_sample_picker"]:
        print(f"    Loading coverage-optimized samples...")
        samples = get_picked_samples(country_paths["samples"])
    else:
        print(f"    Picking random samples...")
        samples = get_random_samples(country_paths["raw"], n_samples=3, seed=42)

    # 2. Select schema
    if config["rich_schema"]:
        schema_for_prompt = get_rich_schema()
        schema_label = "rich"
    else:
        schema_for_prompt = get_minimal_schema()
        schema_label = "minimal"

    print(f"    Schema: {schema_label} ({len(json.dumps(schema_for_prompt))} chars)")

    # 3. Pass 1: Extract rules
    print(f"    Pass 1: Extracting rules...")
    prompt = build_extraction_prompt(samples, schema_for_prompt, country_name)
    rules = call_gemini(prompt)
    llm_calls = 1

    if not rules:
        print(f"    ERROR: No rules returned")
        return None

    print(f"    Pass 1: {len(rules)} rules")

    # 4. Pass 2 (if enabled)
    if config["two_pass"]:
        missing = find_missing_fields(rules)
        if missing:
            print(f"    Pass 2: {len(missing)} fields missing, asking LLM...")
            new_rules = second_pass(samples, rules, missing, schema_for_prompt, country_name)
            llm_calls += 1
            if new_rules:
                print(f"    Pass 2: Found {len(new_rules)} additional rules")
                rules.extend(new_rules)
            else:
                print(f"    Pass 2: No additional rules found")
        else:
            print(f"    Pass 2: All fields mapped in Pass 1")
    else:
        print(f"    Pass 2: DISABLED")

    # 5. Evaluate
    gt = GROUND_TRUTH.get(country_name)
    if not gt:
        print(f"    WARNING: No ground truth for {country_name}")
        return None

    eval_results = evaluate_country(country_name, gt, rules)

    # 6. Build result object
    result = {
        "experiment": experiment_id,
        "experiment_name": config["name"],
        "country": country_name,
        "config": {
            "two_pass": config["two_pass"],
            "use_sample_picker": config["use_sample_picker"],
            "rich_schema": config["rich_schema"],
        },
        "timestamp": datetime.now().isoformat(),
        "llm_calls": llm_calls,
        "num_rules": len(rules),
        "rules": rules,
        "metrics": {
            "total_required": eval_results["total_required"],
            "total_expected": eval_results["total_expected"],
            "field_coverage": round(eval_results["req_field_pct"], 1),
            "path_accuracy": round(eval_results["req_path_pct"], 1),
            "full_accuracy": round(eval_results["req_full_pct"], 1),
            "optional_skips": eval_results["optional_missing"],
            "missing_fields": eval_results["missing_fields"],
            "extra_fields": eval_results["extra_fields"],
        },
        "details": eval_results["details"],
    }

    # Print summary
    m = result["metrics"]
    print(f"\n    RESULTS: coverage={m['field_coverage']}%, path={m['path_accuracy']}%, full={m['full_accuracy']}%")
    print(f"    Rules: {result['num_rules']}, LLM calls: {result['llm_calls']}")
    if m["missing_fields"]:
        print(f"    MISSING: {m['missing_fields']}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def run_mode(requested):
    """Execute experiments and save per-country JSON results. No summary.

    Called in each API-key session. Each session saves its individual
    E{exp}_{country}.json files to RESULTS_DIR and stops. Aggregation across
    sessions happens later via `aggregate_mode`.

    Fail-fast: if any country within an experiment fails (run_experiment
    returns None), the script stops immediately. A partial experiment can't
    be compared against a full baseline, so continuing would just waste calls.
    """
    print("=" * 60)
    print("  ABLATION STUDY — Experiment Runner")
    print("=" * 60)
    print(f"  Experiments: {requested}")
    print(f"  Countries:   {list(COUNTRIES.keys())}")
    print(f"  Results dir: {RESULTS_DIR}")
    print()

    aborted = False

    for exp_id in requested:
        if exp_id not in EXPERIMENTS:
            print(f"  Unknown experiment: {exp_id}")
            continue

        config = EXPERIMENTS[exp_id]
        print(f"\n{'='*60}")
        print(f"  {config['name']}")
        print(f"{'='*60}")

        for country_name, country_paths in COUNTRIES.items():
            if not country_paths["raw"].exists():
                print(f"  SKIP {country_name}: raw file not found")
                continue

            result = run_experiment(exp_id, config, country_name, country_paths)

            if result is None:
                # Something inside run_experiment returned None: Pass 1 got no
                # rules back, or ground truth was missing. Either way this
                # experiment is broken — stop before wasting calls on the rest.
                print(f"\n{'!'*60}")
                print(f"  ABORTING: {exp_id} failed for {country_name}.")
                print(f"  Not running later countries or experiments — a partial")
                print(f"  experiment is useless for the ablation comparison.")
                print(f"  Fix the underlying issue and re-run this experiment:")
                print(f"    python src/run_experiments.py {exp_id}")
                print(f"{'!'*60}")
                aborted = True
                break

            filename = f"{exp_id}_{country_name.lower()}.json"
            filepath = RESULTS_DIR / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"    Saved: {filepath}")

            # (No throttle needed — Anthropic tier 1 limits are far above
            # anything this script produces.)

        if aborted:
            break

    if aborted:
        print(f"\n  Session aborted.")
        print(f"  Calls made: {_calls_made}   Estimated cost so far: ${_estimated_cost_so_far():.4f}")
        return

    print(f"\n{'='*60}")
    print(f"  Session complete.")
    print(f"  Calls made: {_calls_made}")
    print(f"  Input tokens:  {_total_input_tokens:,}")
    print(f"  Output tokens: {_total_output_tokens:,}")
    print(f"  Estimated cost: ${_estimated_cost_so_far():.4f}")
    print(f"  Run `python src/run_experiments.py aggregate` once all sessions")
    print(f"  are finished to produce the consolidated summary + failure analysis.")
    print(f"{'='*60}")


def aggregate_mode():
    """Read every E*_<country>.json in RESULTS_DIR and produce the full summary.

    This is the thesis-facing output: summary table, failure analysis, and the
    two JSON reports (`experiment_summary.json`, `failure_analysis.json`).
    Does not make any API calls. Safe to run any time.
    """
    print("=" * 60)
    print("  ABLATION STUDY — AGGREGATE (across all sessions)")
    print("=" * 60)
    print(f"  Reading from: {RESULTS_DIR}")
    print()

    # ── Load every per-country result file from disk ────────────────────
    all_results = {}
    found_files = []
    for exp_id in EXPERIMENTS:
        for country_name in COUNTRIES:
            filepath = RESULTS_DIR / f"{exp_id}_{country_name.lower()}.json"
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                all_results.setdefault(exp_id, {})[country_name] = result
                found_files.append(filepath.name)

    if not found_files:
        print("  No result files found. Run experiments first.")
        return

    print(f"  Loaded {len(found_files)} result files:")
    for fname in sorted(found_files):
        print(f"    • {fname}")
    print()

    # ── Summary Table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ABLATION STUDY — SUMMARY")
    print(f"{'='*80}")

    header = f"  {'Experiment':<35s} {'Country':<12s} {'Coverage':>9s} {'Path':>9s} {'Full':>9s} {'Rules':>6s} {'Calls':>6s}"
    print(header)
    print(f"  {'─'*35} {'─'*12} {'─'*9} {'─'*9} {'─'*9} {'─'*6} {'─'*6}")

    # Iterate in canonical experiment order (E0, E1, E2, E3) for stable output
    for exp_id in EXPERIMENTS:
        if exp_id not in all_results:
            continue
        for country_name in COUNTRIES:
            if country_name not in all_results[exp_id]:
                continue
            result = all_results[exp_id][country_name]
            m = result["metrics"]
            print(f"  {result['experiment_name']:<35s} {country_name:<12s} "
                  f"{m['field_coverage']:>8.1f}% {m['path_accuracy']:>8.1f}% {m['full_accuracy']:>8.1f}% "
                  f"{result['num_rules']:>6d} {result['llm_calls']:>6d}")

    # ── Failure Analysis (baseline vs each ablation) ────────────────────
    if "E0" in all_results:
        print(f"\n\n{'='*80}")
        print(f"  FAILURE ANALYSIS — What broke in each ablation?")
        print(f"{'='*80}")

        for exp_id in EXPERIMENTS:
            if exp_id == "E0" or exp_id not in all_results:
                continue

            print(f"\n  {EXPERIMENTS[exp_id]['name']}:")
            print(f"  {'─'*60}")

            any_failures = False
            for country_name in COUNTRIES:
                if country_name not in all_results["E0"] or country_name not in all_results[exp_id]:
                    continue

                baseline = all_results["E0"][country_name]
                ablated = all_results[exp_id][country_name]
                failures = failure_analysis(baseline, ablated, exp_id)

                if failures:
                    any_failures = True
                    print(f"\n    {country_name}:")
                    print(f"    {'Field':<40s} {'Baseline':<12s} {'Ablated':<12s} {'Error':<15s} {'Explanation'}")
                    print(f"    {'─'*40} {'─'*12} {'─'*12} {'─'*15} {'─'*30}")
                    for f in failures:
                        print(f"    {f['field']:<40s} {f['baseline_status']:<12s} {f['ablated_status']:<12s} "
                              f"{f['error_type']:<15s} {f['explanation']}")

            if not any_failures:
                print(f"    No regressions detected.")

        # Save failure analysis JSON
        failure_report = {}
        for exp_id in EXPERIMENTS:
            if exp_id == "E0" or exp_id not in all_results:
                continue
            failure_report[exp_id] = {}
            for country_name in COUNTRIES:
                if country_name not in all_results.get("E0", {}) or country_name not in all_results.get(exp_id, {}):
                    continue
                failures = failure_analysis(all_results["E0"][country_name], all_results[exp_id][country_name], exp_id)
                failure_report[exp_id][country_name] = failures

        report_path = RESULTS_DIR / 'failure_analysis.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(failure_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Failure analysis saved to: {report_path}")
    else:
        print(f"\n  (Skipping failure analysis — E0 baseline not yet available.)")

    # ── Save full summary JSON ──────────────────────────────────────────
    summary = {}
    for exp_id in EXPERIMENTS:
        if exp_id not in all_results:
            continue
        summary[exp_id] = {}
        for country_name in COUNTRIES:
            if country_name not in all_results[exp_id]:
                continue
            result = all_results[exp_id][country_name]
            summary[exp_id][country_name] = {
                "metrics": result["metrics"],
                "num_rules": result["num_rules"],
                "llm_calls": result["llm_calls"],
                "config": result["config"],
            }

    summary_path = RESULTS_DIR / 'experiment_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Summary saved to: {summary_path}")

    # ── Coverage check ──────────────────────────────────────────────────
    expected = len(EXPERIMENTS) * len(COUNTRIES)
    actual = sum(len(v) for v in all_results.values())
    print(f"\n  Coverage: {actual}/{expected} experiment-country combinations present.")
    if actual < expected:
        missing = []
        for exp_id in EXPERIMENTS:
            for country_name in COUNTRIES:
                if country_name not in all_results.get(exp_id, {}):
                    missing.append(f"{exp_id}/{country_name}")
        print(f"  Missing: {', '.join(missing)}")

    print(f"\n{'='*60}")
    print("  Aggregation complete.")
    print(f"{'='*60}")


def main():
    args = sys.argv[1:]

    # Dispatch: `aggregate` is a separate mode; anything else is a run request.
    if args and args[0].lower() == "aggregate":
        aggregate_mode()
        return

    requested = args if args else list(EXPERIMENTS.keys())
    requested = [e.upper() for e in requested]
    run_mode(requested)


if __name__ == "__main__":
    main()
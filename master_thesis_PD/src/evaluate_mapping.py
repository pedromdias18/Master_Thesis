"""
Evaluation: Compare LLM-generated extraction rules against manual ground truth.

Updated for:
- Two-pass extraction approach
- Restructured schema (country object, business/postal address, registration_date)
- Fields marked optional when the source data genuinely doesn't have them

Usage (from project root):
    python src/evaluate_mapping.py
"""

import json
from pathlib import Path


GROUND_TRUTH = {
    "Myanmar": {
        "source_file": "myanmar_flat_rules.json",
        "expected_rules": {
            # ── Names ──────────────────────────────────────────────────
            "company_name.legal_name": {
                "json_path": "Corp.CompanyName",
                "extraction_type": "direct",
            },
            "company_name.local_language_name": {
                "json_path": "Corp.AltName",
                "extraction_type": "direct",
            },
            "company_name.trade_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Myanmar data does not have a trade name field",
            },
            "company_name.short_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Myanmar data does not have a short name field",
            },
            # ── Identifiers ────────────────────────────────────────────
            "unique_identifier.registration_number": {
                "json_path": "Corp.RegistrationNumber",
                "extraction_type": "direct",
            },
            "unique_identifier.prior_registration_number": {
                "json_path": "Corp.PriorRegistrationNumber",
                "extraction_type": "direct",
            },
            # ── Parent company ─────────────────────────────────────────
            "parent_company.registration_number": {
                "json_path": "Corp.HoldingCompanyRegNumber",
                "extraction_type": "direct",
            },
            "parent_company.company_name": {
                "json_path": "Corp.HoldingCompanyName",
                "extraction_type": "direct",
            },
            # ── Classification ─────────────────────────────────────────
            "company_type": {
                "json_path": "Corp.CompanyType",
                "extraction_type": "direct",
            },
            "status": {
                "json_path": "Corp.Status",
                "extraction_type": "direct",
            },
            # ── Country ────────────────────────────────────────────────
            "country.registration": {
                "json_path": "MM",
                "extraction_type": "hardcode",
            },
            "country.business_address": {
                "json_path": "Corp.HoldingCompanyJurisdiction",
                "extraction_type": ["direct", "hardcode"],
                "accept_any": True,
                "optional": True,
                "note": "Only populated for overseas entities (~2.8%)",
            },
            "country.postal_address": {
                "json_path": "Corp.PostalAddress",
                "extraction_type": "direct",
                "flexible_match": True,
                "optional": True,
                "note": "Data is mostly empty",
            },
            # ── Addresses ──────────────────────────────────────────────
            "business_address.street": {
                "json_path": "Corp.RegisteredOfficeAddress",
                "extraction_type": ["direct", "fallback_chain"],
                "accept_any": True,
                "flexible_match": True,
                "optional": True,
                "note": "Data is mostly empty but path exists",
            },
            "business_address.city": {
                "json_path": "Corp.RegisteredOfficeAddress",
                "extraction_type": "direct",
                "flexible_match": True,
                "optional": True,
                "note": "Data is mostly empty but path exists",
            },
            "postal_address.street": {
                "json_path": "Corp.PostalAddress",
                "extraction_type": ["direct", "fallback_chain"],
                "accept_any": True,
                "flexible_match": True,
                "optional": True,
                "note": "Data is mostly empty",
            },
            "postal_address.city": {
                "json_path": "Corp.PostalAddress",
                "extraction_type": "direct",
                "flexible_match": True,
                "optional": True,
                "note": "Data is mostly empty",
            },
            # ── Financials ─────────────────────────────────────────────
            "share_capital.amount": {
                "json_path": "Corp.ShareCapitalValue",
                "extraction_type": "direct",
            },
            "share_capital.currency": {
                "json_path": "Corp.ShareCurrency",
                "extraction_type": "direct",
            },
            "share_capital.fully_paid": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Myanmar data does not have a fully_paid field",
            },
            # ── Other ──────────────────────────────────────────────────
            "num_employees": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Myanmar data does not have employee count",
            },
            "registration_date": {
                "json_path": "Corp.RegistrationDateFormatted",
                "extraction_type": "direct",
            },
            "description": {
                "json_path": "Activities[*].CorpActivityType",
                "extraction_type": "join_list",
            },
        },
    },
    "Norway": {
        "source_file": "norway_flat_rules.json",
        "expected_rules": {
            # ── Names ──────────────────────────────────────────────────
            "company_name.legal_name": {
                "json_path": "navn",
                "extraction_type": "direct",
            },
            "company_name.trade_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Norway data does not have a separate trade name",
            },
            "company_name.short_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Norway data does not have a short name",
            },
            "company_name.local_language_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Norway data does not have a separate local language name",
            },
            # ── Identifiers ────────────────────────────────────────────
            "unique_identifier.registration_number": {
                "json_path": "organisasjonsnummer",
                "extraction_type": "direct",
            },
            "unique_identifier.prior_registration_number": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Norway data does not have a prior registration number",
            },
            # ── Parent company ─────────────────────────────────────────
            "parent_company.registration_number": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Norway data does not have parent company in this dataset",
            },
            "parent_company.company_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Norway data does not have parent company in this dataset",
            },
            # ── Classification ─────────────────────────────────────────
            "company_type": {
                "json_path": ["organisasjonsform.kode", "organisasjonsform.beskrivelse"],
                "extraction_type": ["nested_field", "direct"],
                "accept_any": True,
            },
            "status": {
                "json_path": "konkurs|underAvvikling|underTvangsavviklingEllerTvangsopplosning",
                "extraction_type": "boolean_logic",
                "check_references": ["konkurs", "underAvvikling", "underTvangsavviklingEllerTvangsopplosning"],
            },
            # ── Country ────────────────────────────────────────────────
            "country.registration": {
                "json_path": "NO",
                "extraction_type": "hardcode",
            },
            "country.business_address": {
                "json_path": "forretningsadresse.landkode",
                "extraction_type": ["nested_field", "direct"],
                "accept_any": True,
            },
            "country.postal_address": {
                "json_path": "postadresse.landkode",
                "extraction_type": ["nested_field", "direct"],
                "accept_any": True,
            },
            # ── Addresses ──────────────────────────────────────────────
            "business_address.street": {
                "json_path": "forretningsadresse.adresse",
                "extraction_type": ["join_list", "direct", "nested_field"],
                "accept_any": True,
            },
            "business_address.city": {
                "json_path": "forretningsadresse.poststed",
                "extraction_type": ["direct", "nested_field"],
                "accept_any": True,
            },
            "postal_address.street": {
                "json_path": "postadresse.adresse",
                "extraction_type": ["join_list", "direct", "nested_field"],
                "accept_any": True,
            },
            "postal_address.city": {
                "json_path": "postadresse.poststed",
                "extraction_type": ["direct", "nested_field"],
                "accept_any": True,
            },
            # ── Financials ─────────────────────────────────────────────
            "num_employees": {
                "json_path": "antallAnsatte",
                "extraction_type": "direct",
            },
            "share_capital.amount": {
                "json_path": "kapital.belop",
                "extraction_type": ["nested_field", "direct"],
                "accept_any": True,
            },
            "share_capital.currency": {
                "json_path": "kapital.valuta",
                "extraction_type": ["nested_field", "direct"],
                "accept_any": True,
            },
            "share_capital.fully_paid": {
                "json_path": "kapital.fulltInnbetalt",
                "extraction_type": ["nested_field", "direct"],
                "accept_any": True,
            },
            # ── Other ──────────────────────────────────────────────────
            "registration_date": {
                "json_path": ["registreringsdatoEnhetsregisteret", "registreringsdatoForetaksregisteret", "stiftelsesdato"],
                "extraction_type": ["direct", "fallback_chain"],
                "accept_any": True,
            },
            "description": {
                "json_path": ["vedtektsfestetFormaal", "aktivitet"],
                "extraction_type": ["fallback_chain", "join_list"],
                "accept_any": True,
                "check_references": ["vedtektsfestetFormaal", "aktivitet"],
            },
        },
    },
    "Honduras": {
        "source_file": "honduras_flat_rules.json",
        "expected_rules": {
            # ── Names ──────────────────────────────────────────────────
            "company_name.legal_name": {
                "json_path": ["Información General[0].Razón Social", "Información General[0].Denominación Social"],
                "extraction_type": ["fallback_chain", "first_element_field"],
                "accept_any": True,
            },
            "company_name.trade_name": {
                "json_path": "Información General[0].Nombre Comercial",
                "extraction_type": "first_element_field",
            },
            "company_name.short_name": {
                "json_path": "Información General[0].Siglas",
                "extraction_type": "first_element_field",
            },
            "company_name.local_language_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Honduras data is all in Spanish, no separate local language name",
            },
            # ── Identifiers ────────────────────────────────────────────
            "unique_identifier.registration_number": {
                "json_path": "FileNumber",
                "extraction_type": "direct",
            },
            "unique_identifier.prior_registration_number": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Honduras data does not have a prior registration number",
            },
            # ── Parent company ─────────────────────────────────────────
            "parent_company.registration_number": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Honduras data does not have parent company info",
            },
            "parent_company.company_name": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Honduras data does not have parent company info",
            },
            # ── Classification ─────────────────────────────────────────
            "company_type": {
                "json_path": "Información General[0].Tipo de Comerciante",
                "extraction_type": "first_element_field",
            },
            "status": {
                "json_path": "Información General[0].Estado",
                "extraction_type": "first_element_field",
            },
            # ── Country ────────────────────────────────────────────────
            "country.registration": {
                "json_path": "HN",
                "extraction_type": "hardcode",
            },
            "country.business_address": {
                "json_path": "HN",
                "extraction_type": "hardcode",
            },
            "country.postal_address": {
                "json_path": "HN",
                "extraction_type": "hardcode",
                "optional": True,
            },
            # ── Addresses ──────────────────────────────────────────────
            "business_address.street": {
                "json_path": "Información General[0].Dirección de la Empresa",
                "extraction_type": "first_element_field",
            },
            "business_address.city": {
                "json_path": "Información General[0].Ubicación de la Empresa",
                "extraction_type": "first_element_field",
            },
            "postal_address.street": {
                "json_path": "Información General[0].Domicilio",
                "extraction_type": "first_element_field",
                "optional": True,
                "note": "Domicilio is a valid but imprecise source for postal address",
            },
            "postal_address.city": {
                "json_path": "Información General[0].Domicilio",
                "extraction_type": "first_element_field",
                "optional": True,
                "note": "Domicilio contains city but needs parsing",
            },
            # ── Financials ─────────────────────────────────────────────
            "share_capital.amount": {
                "json_path": ["Capital[0].Capital Pagado", "Capital[0].Capital Suscrito",
                              "Capital[0].Capital Maximus", "Capital[0].Capital Mínimo"],
                "extraction_type": ["fallback_chain", "first_element_field"],
                "accept_any": True,
            },
            "share_capital.currency": {
                "json_path": "HNL",
                "extraction_type": "hardcode",
            },
            "share_capital.fully_paid": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Honduras data does not have explicit fully_paid flag",
            },
            "num_employees": {
                "json_path": None,
                "extraction_type": None,
                "optional": True,
                "note": "Honduras data does not have employee count",
            },
            # ── Other ──────────────────────────────────────────────────
            "registration_date": {
                "json_path": ["Información General[0].Vigencia", "Resúmenes[0].Fecha Inscripción"],
                "extraction_type": "first_element_field",
                "accept_any": True,
            },
            "description": {
                "json_path": ["Información General[0].Finalidad",
                              "Información General[0].Objeto Social de la Empresa"],
                "extraction_type": ["first_element_field", "fallback_chain"],
                "accept_any": True,
            },
        },
    },
}


# ── Evaluation Engine (unchanged) ────────────────────────────────────────────

def normalize_path(path):
    if isinstance(path, list):
        return [normalize_path(p) for p in path]
    s = str(path).strip().strip('"').strip("'").lower()
    s = s.replace('[*]', '').replace('[0]', '')
    return s


def path_matches(llm_path, expected_path, flexible=False):
    if isinstance(llm_path, list):
        llm_paths = [normalize_path(p) for p in llm_path]
    else:
        llm_paths = [normalize_path(llm_path)]

    if isinstance(expected_path, list):
        expected_paths = [normalize_path(p) for p in expected_path]
    else:
        expected_paths = [normalize_path(expected_path)]

    if flexible:
        for lp in llm_paths:
            for ep in expected_paths:
                if lp.startswith(ep) or ep.startswith(lp):
                    return True
        return False

    for lp in llm_paths:
        for ep in expected_paths:
            if lp == ep or ep in lp or lp in ep:
                return True
    return False


def check_references(llm_path_str, required_refs):
    path_lower = str(llm_path_str).lower()
    found = sum(1 for ref in required_refs if ref.lower() in path_lower)
    return found, len(required_refs)


# Aliases that LLMs sometimes produce for the same extraction type.
# Keys are the non-canonical variants, values are the canonical form.
_TYPE_ALIASES = {
    "hardcoded":        "hardcode",
    "hard_code":        "hardcode",
    "nested":           "nested_field",
    "first_element":    "first_element_field",
    "join":             "join_list",
    "fallback":         "fallback_chain",
    "boolean":          "boolean_logic",
}

def normalize_extraction_type(t):
    """Collapse trivial spelling variants so evaluation is not tripped up
    by LLM outputs like 'hardcoded' vs the canonical 'hardcode'."""
    if not t:
        return t
    return _TYPE_ALIASES.get(t, t)


def evaluate_country(country_name, ground_truth, llm_rules):
    expected = ground_truth["expected_rules"]
    llm_by_field = {rule["canonical_field"]: rule for rule in llm_rules}

    results = {
        "country": country_name,
        "total_expected": 0,
        "total_required": 0,
        "total_llm": len(llm_rules),
        "field_matches": 0,
        "path_matches": 0,
        "type_matches": 0,
        "full_matches": 0,
        "optional_missing": 0,
        "missing_fields": [],
        "extra_fields": [],
        "details": [],
    }

    for cf, expected_rule in expected.items():
        is_optional = expected_rule.get("optional", False)
        results["total_expected"] += 1
        if not is_optional:
            results["total_required"] += 1

        detail = {
            "canonical_field": cf,
            "expected_path": expected_rule["json_path"],
            "expected_type": expected_rule["extraction_type"],
            "optional": is_optional,
        }

        if cf not in llm_by_field:
            if is_optional:
                detail["status"] = "OPTIONAL_SKIP"
                detail["path_correct"] = None
                detail["type_correct"] = None
                results["optional_missing"] += 1
            else:
                detail["status"] = "MISSING"
                detail["path_correct"] = False
                detail["type_correct"] = False
                results["missing_fields"].append(cf)
        else:
            llm_rule = llm_by_field[cf]
            detail["llm_path"] = llm_rule["json_path"]
            detail["llm_type"] = llm_rule["extraction_type"]
            results["field_matches"] += 1

            flexible = expected_rule.get("flexible_match", False)

            if "check_references" in expected_rule:
                found, total = check_references(str(llm_rule["json_path"]), expected_rule["check_references"])
                detail["path_correct"] = (found == total)
                detail["refs_found"] = f"{found}/{total}"
            elif expected_rule["json_path"] is None:
                # Optional field with no expected path — if LLM produced something, that's a bonus
                detail["path_correct"] = True
            else:
                detail["path_correct"] = path_matches(llm_rule["json_path"], expected_rule["json_path"], flexible=flexible)

            if detail["path_correct"]:
                results["path_matches"] += 1

            if expected_rule["extraction_type"] is None:
                detail["type_correct"] = True
            else:
                expected_types = expected_rule["extraction_type"]
                if isinstance(expected_types, str):
                    expected_types = [expected_types]
                llm_type = normalize_extraction_type(llm_rule["extraction_type"])
                expected_types = [normalize_extraction_type(t) for t in expected_types]
                detail["type_correct"] = llm_type in expected_types

            if detail["type_correct"]:
                results["type_matches"] += 1

            if detail["path_correct"] and detail["type_correct"]:
                results["full_matches"] += 1
                detail["status"] = "CORRECT"
            elif detail["path_correct"]:
                detail["status"] = "PATH_OK"
            elif detail["type_correct"]:
                detail["status"] = "TYPE_OK"
            else:
                detail["status"] = "WRONG"

        results["details"].append(detail)

    for cf in llm_by_field:
        if cf not in expected:
            results["extra_fields"].append(cf)

    # Metrics on required fields only
    n_req = results["total_required"]
    req_found = sum(1 for d in results["details"]
                    if not d.get("optional") and d.get("status") not in ("MISSING",))
    req_path = sum(1 for d in results["details"]
                   if not d.get("optional") and d.get("path_correct") is True)
    req_full = sum(1 for d in results["details"]
                   if not d.get("optional") and d.get("path_correct") is True and d.get("type_correct") is True)

    results["req_field_pct"] = (req_found / n_req * 100) if n_req > 0 else 0
    results["req_path_pct"] = (req_path / n_req * 100) if n_req > 0 else 0
    results["req_full_pct"] = (req_full / n_req * 100) if n_req > 0 else 0

    # Overall metrics
    n_all = results["total_expected"]
    results["all_field_pct"] = (results["field_matches"] / n_all * 100) if n_all > 0 else 0
    results["all_path_pct"] = (results["path_matches"] / n_all * 100) if n_all > 0 else 0
    results["all_full_pct"] = (results["full_matches"] / n_all * 100) if n_all > 0 else 0

    return results


def print_results(results):
    print(f"\n{'='*70}")
    print(f"  {results['country']} — Evaluation Results")
    print(f"{'='*70}")
    print(f"  Expected (total/required): {results['total_expected']} / {results['total_required']}")
    print(f"  LLM rules produced:        {results['total_llm']}")
    print(f"  Optional fields skipped:   {results['optional_missing']}")
    print()
    print(f"  Required fields:")
    print(f"    Coverage:      {results['req_field_pct']:.1f}%")
    print(f"    Path accuracy: {results['req_path_pct']:.1f}%")
    print(f"    Full accuracy: {results['req_full_pct']:.1f}%")
    print()
    print(f"  All fields (incl. optional):")
    print(f"    Coverage:      {results['field_matches']}/{results['total_expected']} ({results['all_field_pct']:.1f}%)")
    print(f"    Path accuracy: {results['path_matches']}/{results['total_expected']} ({results['all_path_pct']:.1f}%)")
    print(f"    Full accuracy: {results['full_matches']}/{results['total_expected']} ({results['all_full_pct']:.1f}%)")

    if results["missing_fields"]:
        print(f"\n  MISSING required: {results['missing_fields']}")
    if results["extra_fields"]:
        print(f"  EXTRA (not in ground truth): {results['extra_fields']}")

    print(f"\n  {'Field':<42s} {'Status':<15s} {'Path':<6s} {'Type':<6s} {'Opt?'}")
    print(f"  {'-'*42} {'-'*15} {'-'*6} {'-'*6} {'-'*4}")
    for d in results["details"]:
        cf = d["canonical_field"]
        status = d["status"]
        opt = "opt" if d.get("optional") else ""
        if status in ("MISSING", "OPTIONAL_SKIP"):
            p = "—"
            t = "—"
        else:
            p = "✓" if d.get("path_correct") else "✗"
            t = "✓" if d.get("type_correct") else "✗"
        refs = d.get("refs_found", "")
        extra = f" [{refs}]" if refs else ""
        print(f"  {cf:<42s} {status:<15s} {p:<6s} {t:<6s} {opt}{extra}")


def main():
    rules_dir = Path('data/processed')
    all_results = []

    for country_name, gt in GROUND_TRUTH.items():
        rules_path = rules_dir / gt["source_file"]
        if not rules_path.exists():
            print(f"\n  SKIP {country_name}: {rules_path} not found")
            continue

        with open(rules_path, 'r', encoding='utf-8') as f:
            llm_rules = json.load(f)

        results = evaluate_country(country_name, gt, llm_rules)
        print_results(results)
        all_results.append(results)

    if all_results:
        print(f"\n{'='*70}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*70}")

        t_req = sum(r["total_required"] for r in all_results)
        t_all = sum(r["total_expected"] for r in all_results)
        t_found = sum(r["field_matches"] for r in all_results)
        t_path = sum(r["path_matches"] for r in all_results)
        t_full = sum(r["full_matches"] for r in all_results)

        print(f"  Required fields:   {t_req}")
        print(f"  Total fields:      {t_all}")
        print(f"  LLM found:         {t_found}/{t_all} ({t_found/t_all*100:.1f}%)")
        print(f"  Path correct:      {t_path}/{t_all} ({t_path/t_all*100:.1f}%)")
        print(f"  Fully correct:     {t_full}/{t_all} ({t_full/t_all*100:.1f}%)")

        # Required-only summary
        r_found = sum(
            sum(1 for d in r["details"] if not d.get("optional") and d.get("status") not in ("MISSING",))
            for r in all_results
        )
        r_path = sum(
            sum(1 for d in r["details"] if not d.get("optional") and d.get("path_correct") is True)
            for r in all_results
        )
        r_full = sum(
            sum(1 for d in r["details"] if not d.get("optional") and d.get("path_correct") and d.get("type_correct"))
            for r in all_results
        )
        print(f"\n  Required only:")
        print(f"    Found:           {r_found}/{t_req} ({r_found/t_req*100:.1f}%)")
        print(f"    Path correct:    {r_path}/{t_req} ({r_path/t_req*100:.1f}%)")
        print(f"    Fully correct:   {r_full}/{t_req} ({r_full/t_req*100:.1f}%)")

        output_path = rules_dir / 'evaluation_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
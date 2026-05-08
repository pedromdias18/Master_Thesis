"""
Canonical schema for the unified company database.
Every dataset will be mapped to these fields.

Only fields useful for searching and filtering companies are included.
Contact details (phone, email, website) and detailed share breakdowns
were removed as they don't serve the search use case.
"""

# This dictionary defines each field in your unified schema.
# The AI will use the "description" and "examples" to understand
# what each field means when mapping unknown columns.

CANONICAL_SCHEMA = {
    # ── Company Name (object with sub-fields) ────────────────────────────────
    "company_name": {
        "description": "The names of the company. Contains multiple name variants: "
                       "legal_name is the official registered name (always present), "
                       "trade_name is the commercial/business name used in trade, "
                       "short_name is an abbreviation or acronym (siglas), and "
                       "local_language_name is the name in the local/native language "
                       "(e.g. Burmese, Arabic, Chinese characters).",
        "data_type": "object",
        "required": True,
        "schema": {
            "legal_name": {
                "description": "The official registered name of the company as it appears in the national registry. "
                               "This is the primary name and is always present.",
                "data_type": "string",
                "required": True,
            },
            "trade_name": {
                "description": "The commercial or trading name of the company, also known as business name, "
                               "DBA (doing business as), or 'nombre comercial'. May differ from the legal name. "
                               "Only used as an additional name when legal_name is already populated. "
                               "If the dataset has only one name column, that column is always legal_name, not this field.",
                "data_type": "string",
                "required": False,
            },
            "short_name": {
                "description": "An abbreviation, acronym, or 'siglas' of the company name "
                               "(e.g. 'BMW' for Bayerische Motoren Werke, 'HSBC' for Hongkong and Shanghai Banking Corporation). "
                               "Only used as an additional name when legal_name is already populated. "
                               "If the dataset has only one name column, that column is always legal_name, not this field.",
                "data_type": "string",
                "required": False,
            },
            "local_language_name": {
                "description": "The company name written in the local or native language/script "
                               "(e.g. Burmese, Arabic, Chinese, Devanagari). Only used as an "
                               "additional name when legal_name is already populated. If the dataset "
                               "has only one name column, that column is always legal_name, not this field.",
                "data_type": "string",
                "required": False,
            },
        },
        "examples": [
            {"legal_name": "Siemens Aktiengesellschaft", "trade_name": "Siemens", "short_name": "SAG"},
            {"legal_name": "Tesco PLC", "trade_name": "Tesco"},
            {"legal_name": "Myanmar Golden Star Co., Ltd.", "local_language_name": "မြန်မာ ရွှေကြယ် ကုမ္ပဏီ"},
        ],
        "synonyms": ["name", "business name", "entity name", "firm name", "organization name",
                      "trade name", "legal name", "razão social", "denominação", "nome da empresa",
                      "nombre comercial", "siglas", "DBA", "doing business as", "alt name",
                      "alternative name", "local name", "native name"]
    },

    # ── Unique Identifier (object with sub-fields) ───────────────────────────
    "unique_identifier": {
        "description": "The official registration or identification numbers assigned by the national registry. "
                       "Contains the current registration_number and optionally a prior_registration_number "
                       "if the company was re-registered or migrated from an older system.",
        "data_type": "object",
        "required": False,
        "schema": {
            "registration_number": {
                "description": "The current official registration number, company number, or identification "
                               "number assigned by the national registry (e.g. Companies House number, "
                               "Handelsregisternummer, organisasjonsnummer, número de registro).",
                "data_type": "string",
                "required": True,
            },
            "prior_registration_number": {
                "description": "A previous registration number, used when the company was re-registered "
                               "under a new system or number (e.g. old registry number before migration, "
                               "former company number).",
                "data_type": "string",
                "required": False,
            },
        },
        "examples": [
            {"registration_number": "HRB 12345"},
            {"registration_number": "12345678", "prior_registration_number": "OLD-9876"},
            {"registration_number": "916543210"},
        ],
        "synonyms": ["company number", "registration number", "tax ID", "NIF", "NIPC",
                      "SIREN", "SIRET", "Handelsregisternummer", "CRN", "número de registro",
                      "organisasjonsnummer", "company ID", "entity ID", "prior registration",
                      "old registration number", "former number"]
    },

    # ── Parent Company (object with sub-fields) ──────────────────────────────
    "parent_company": {
        "description": "Information about the parent or holding company, if this company is a subsidiary. "
                       "Contains the parent's registration number and name.",
        "data_type": "object",
        "required": False,
        "schema": {
            "registration_number": {
                "description": "The registration number of the parent/holding company",
                "data_type": "string",
                "required": False,
            },
            "company_name": {
                "description": "The name of the parent/holding company",
                "data_type": "string",
                "required": False,
            },
        },
        "examples": [
            {"registration_number": "HRB 99999", "company_name": "Siemens AG"},
            {"registration_number": "C-12345", "company_name": "Myanmar Holdings Ltd."},
        ],
        "synonyms": ["holding company", "parent entity", "parent organization",
                      "holding company name", "holding company registration",
                      "parent company name", "parent company number",
                      "HoldingCompanyRegNumber", "HoldingCompanyName",
                      "empresa matriz", "société mère", "Muttergesellschaft"]
    },

    "company_type": {
        "description": "The legal form or structure of the company. Uses a two-level classification: "
                       "a broad category (single letter) and a specific sub-type (3-letter code). "
                       "The value stored should be the 3-letter sub-type code when known, "
                       "or the single-letter category code as fallback.",
        "data_type": "categorical",
        "required": False,
        "allowed_values": {
            # ── A – Association ──────────────────────────────────────────────
            # National/International Associations, Body Corporate, Economic Interest Groups
            "A": {
                "label": "Association",
                "description": "National and International Associations, Body Corporate, Economic Interest Groups",
                "sub_types": {
                    "ASS": "Association",
                    "COO": "Cooperative society",
                    "NPO": "Not-for-Profit Organization",
                    "INO": "International Organization",
                }
            },
            # ── B – Branch ───────────────────────────────────────────────────
            "B": {
                "label": "Branch",
                "description": "Branch of a local company",
                "sub_types": {
                    "BRA": "Domestic Branch",
                }
            },
            # ── C – Local Company ────────────────────────────────────────────
            # Local incorporated companies with capital.
            # Public: shares can be listed; Private: shares are not listed;
            # One Person: only 1 shareholder
            "C": {
                "label": "Local Company",
                "description": "Local incorporated companies with capital (public, private, or one-person)",
                "sub_types": {
                    # Limited liability
                    "LTD": "Limited Company",
                    "PLC": "Public Limited Company",
                    "PVT": "Private Limited Company",
                    "LLC": "Limited Liability Company",
                    "LTG": "Company Limited by Guarantee",
                    # Unlimited liability
                    "PUC": "Public Unlimited Company",
                    "PVU": "Private Unlimited Company",
                    # Special forms with limited liability/restrictions
                    "OPC": "One Person Limited Company",
                    "LLO": "One Person Limited Liability Company",
                    "SLC": "Simplified Limited Company",
                    "SLO": "One Person Simplified Limited Company",
                    "LPS": "Limited Partnership with Share Capital",
                    "RPC": "Restricted Purpose Company",
                    "EST": "Establishment (Anstalt)",
                }
            },
            # ── F – Foreign Entity ───────────────────────────────────────────
            # Foreign incorporated companies (both FCO and FBR are branches of a foreign company)
            "F": {
                "label": "Foreign Entity",
                "description": "Foreign incorporated companies",
                "sub_types": {
                    "FCO": "Foreign Company (registered in the country)",
                    "FBR": "Foreign Branch (not registered in the country)",
                }
            },
            # ── G – Governmental Organization ────────────────────────────────
            # Public sector companies; no share capital
            "G": {
                "label": "Governmental Organization",
                "description": "Public sector companies (central/local governments, education, foreign governmental). No share capital.",
                "sub_types": {
                    "GOA": "Public Administration",
                    "GOS": "Public Service",
                    "GOE": "Public Education",
                    "GOD": "Domestic (production/services entities)",
                    "GOF": "Foreign Governmental Organization",
                }
            },
            # ── P – Private Company ──────────────────────────────────────────
            # Local incorporated/unincorporated private entities; no share capital
            "P": {
                "label": "Private Company",
                "description": "Local incorporated and/or unincorporated private entities (business names, sole traders, partnerships). No share capital.",
                "sub_types": {
                    "PPS": "Sole Proprietorship / Trader (only 1 owner)",
                    "BNM": "Business Name (1 or multiple owners)",
                    "SPS": "Simple Partnership / Close Corporation (1 or multiple partners)",
                    "LLP": "Limited Liability Partnership / Limited Partnership (multiple partners)",
                    "ULT": "Unlimited Liability Partnership / General Partnership (multiple partners)",
                }
            },
            # ── T – Trust / Collective Investments ───────────────────────────
            # Trusts, investment funds, managed by another entity
            "T": {
                "label": "Trust / Collective Investments",
                "description": "Trusts (open or closed investment trusts), collective investment funds, venture capital trusts, managed by another entity",
                "sub_types": {
                    "TRU": "Trust Fund",
                    "FOU": "Foundation",
                    "ICO": "Collective Investments – other (Trust/Funds/Companies/Contracts etc.)",
                    "ICV": "Collective Investments with variable capital",
                    "ICF": "Collective Investments with fixed capital",
                    "LCI": "Limited Partnership for Collective Investments",
                    "SPC": "Segregated Portfolio Company",
                    "PCC": "Protected Cell Company",
                    "TRC": "Trust Company",
                }
            },
        },
        "examples": ["PLC", "PVT", "LLC", "ASS", "PPS", "LLP", "GOA", "TRU"],
        "synonyms": ["legal form", "entity type", "business type", "forma jurídica",
                      "type de société", "Rechtsform"]
    },
    "status": {
        "description": "The current life-cycle status of the company",
        "data_type": "categorical",
        "required": False,
        "allowed_values": {
            # ── Pre-life ─────────────────────────────────────────────────────
            "FOR": {
                "label": "Formation",
                "life_cycle_stage": "pre-life",
                "description": "Pre-registered or in process for registration; business not yet started",
            },
            # ── Life ─────────────────────────────────────────────────────────
            "RER": {
                "label": "Re-registered",
                "life_cycle_stage": "life",
                "description": "Re-registered after struck-off, dormancy, or resolution; active registration, back in business",
            },
            "ACT": {
                "label": "Active",
                "life_cycle_stage": "life",
                "description": "Registered and active; in business, business started",
            },
            "INA": {
                "label": "Inactive",
                "life_cycle_stage": "life",
                "description": "Dormant or suspended; no business activities, pending",
            },
            "ADM": {
                "label": "Administration",
                "life_cycle_stage": "life",
                "description": "Expired, forfeited, or under sequester (open issues like unpaid fees); in administration",
            },
            # ── Dying ────────────────────────────────────────────────────────
            "LIP": {
                "label": "Liquidation, provisional",
                "life_cycle_stage": "dying",
                "description": "In provisional liquidation",
            },
            "LIQ": {
                "label": "Liquidation",
                "life_cycle_stage": "dying",
                "description": "In liquidation, resolution, winding-up, or striking-off (in process for de-registration); default when details unknown",
            },
            "LII": {
                "label": "Liquidation compulsory – insolvency",
                "life_cycle_stage": "dying",
                "description": "In liquidation, compulsory/judicial winding-up (insolvency or unknown)",
            },
            "LIS": {
                "label": "Liquidation compulsory – solvency",
                "life_cycle_stage": "dying",
                "description": "In liquidation, compulsory/judicial winding-up (solvency)",
            },
            "LIM": {
                "label": "Liquidation voluntary – members",
                "life_cycle_stage": "dying",
                "description": "In liquidation, voluntary winding-up (members/partners or unknown)",
            },
            "LIC": {
                "label": "Liquidation voluntary – creditors",
                "life_cycle_stage": "dying",
                "description": "In liquidation, voluntary winding-up (creditors)",
            },
            "REC": {
                "label": "Receivership",
                "life_cycle_stage": "dying",
                "description": "In receivership",
            },
            "BAN": {
                "label": "Bankruptcy",
                "life_cycle_stage": "dying",
                "description": "In bankruptcy",
            },
            "DSL": {
                "label": "Dissolution",
                "life_cycle_stage": "dying",
                "description": "In dissolution (in process for de-registration)",
            },
            "MER": {
                "label": "Merger",
                "life_cycle_stage": "dying",
                "description": "In merger (in process for de-registration)",
            },
            "CON": {
                "label": "Conversion",
                "life_cycle_stage": "dying",
                "description": "In conversion (in process for de-registration)",
            },
            # ── Died ─────────────────────────────────────────────────────────
            "CEA": {
                "label": "Ceased",
                "life_cycle_stage": "died",
                "description": "No longer registered (struck-off, cancelled, ceased, closed, converted, dissolved, merged, rescinded, revoked, withdrawn); out of business",
            },
        },
        "examples": ["ACT", "INA", "LIQ", "CEA", "BAN", "MER"],
        "synonyms": ["company status", "entity status", "state", "situação", "statut",
                      "operating status", "life-cycle stage"]
    },
    # ── Country (object with sub-fields) ────────────────────────────────────
    "country": {
        "description": "Country information for the company. Contains the country of registration "
                       "(always known, since data comes from that country's registry), the country "
                       "from the business/operating address (may differ if the company operates abroad), "
                       "and the country from the postal address if available.",
        "data_type": "object",
        "required": True,
        "schema": {
            "registration": {
                "description": "The country where the company is officially registered, using ISO 3166-1 "
                               "alpha-2 codes. This is always known because the data comes from that "
                               "country's national registry. Always hardcoded based on the data source.",
                "data_type": "string",
                "required": True,
            },
            "business_address": {
                "description": "The country from the company's business or operating address, using "
                               "ISO 3166-1 alpha-2 codes. May differ from the registration country "
                               "(e.g., a company registered in Norway may operate in Sweden).",
                "data_type": "string",
                "required": False,
            },
            "postal_address": {
                "description": "The country from the company's postal or mailing address, using "
                               "ISO 3166-1 alpha-2 codes. May differ from both registration and "
                               "business address countries.",
                "data_type": "string",
                "required": False,
            },
        },
        "examples": [
            {"registration": "NO", "business_address": "NO", "postal_address": "NO"},
            {"registration": "NO", "business_address": "SE"},
            {"registration": "MM"},
        ],
        "synonyms": ["nation", "registered country", "country of incorporation", "país", "pays",
                      "country code", "land", "landkode"]
    },

    # ── Business Address (object with sub-fields) ────────────────────────────
    "business_address": {
        "description": "The business or operating address of the company. This is the address where "
                       "the company conducts its main business activities. Also known as registered "
                       "office address, headquarters, or 'forretningsadresse'.",
        "data_type": "object",
        "required": False,
        "schema": {
            "street": {
                "description": "The street address lines (e.g., 'Kjøpmannsgata 20', '123 High Street')",
                "data_type": "string",
                "required": False,
            },
            "city": {
                "description": "The city or municipality (e.g., 'OSLO', 'San Pedro Sula')",
                "data_type": "string",
                "required": False,
            },
        },
        "examples": [
            {"street": "Kjøpmannsgata 20", "city": "STJØRDAL"},
            {"street": "SAN PEDRO SULA, DEPARTAMENTO DE CORTES", "city": "SAN PEDRO SULA"},
        ],
        "synonyms": ["registered address", "headquarters", "office address", "business address",
                      "forretningsadresse", "registered office", "dirección de la empresa",
                      "ubicación", "Anschrift", "adresse", "morada", "street address"]
    },

    # ── Postal Address (object with sub-fields) ─────────────────────────────
    "postal_address": {
        "description": "The postal or mailing address of the company. May differ from the business "
                       "address. Used for correspondence. Also known as 'postadresse'.",
        "data_type": "object",
        "required": False,
        "schema": {
            "street": {
                "description": "The postal address street lines",
                "data_type": "string",
                "required": False,
            },
            "city": {
                "description": "The postal address city or municipality",
                "data_type": "string",
                "required": False,
            },
        },
        "examples": [
            {"street": "c/o Finn Jarle Sørli, Dyvasåsen 3", "city": "HELL"},
            {"street": "P.O. Box 123", "city": "OSLO"},
        ],
        "synonyms": ["mailing address", "postal address", "postadresse", "correspondence address",
                      "domicilio", "post address"]
    },

    "num_employees": {
        "description": "The number of employees or workforce size of the company",
        "data_type": "integer",
        "required": False,
        "examples": [5, 50, 1200, 85000],
        "synonyms": ["employees", "headcount", "workforce", "staff count", "number of workers",
                      "colaboradores", "Mitarbeiteranzahl", "effectif"]
    },
    "share_capital": {
        "description": "The total share capital (also known as registered capital, authorized capital, "
                       "or stated capital) of the company. This is the aggregate nominal/par value of all "
                       "issued shares. Stored as a numeric amount alongside a currency code.",
        "data_type": "object",
        "required": False,
        "schema": {
            "amount": {
                "description": "Total share capital amount",
                "data_type": "float",
            },
            "currency": {
                "description": "ISO 4217 currency code",
                "data_type": "string",
            },
            "fully_paid": {
                "description": "Whether the share capital has been fully paid up",
                "data_type": "boolean",
            },
        },
        "examples": [
            {"amount": 50000.00, "currency": "EUR", "fully_paid": True},
            {"amount": 100000.00, "currency": "GBP", "fully_paid": False},
        ],
        "synonyms": ["registered capital", "authorized capital", "stated capital", "capital social",
                      "Stammkapital", "Grundkapital", "capital souscrit", "capital autorizado",
                      "nominal capital", "issued capital"]
    },
    "registration_date": {
        "description": "The date the company was registered or incorporated in the national registry",
        "data_type": "date",
        "required": False,
        "examples": ["2005-03-15", "1998-01-01", "2020-11-22"],
        "synonyms": ["incorporation date", "registration date", "date of registration", "established",
                      "data de constituição", "date de création", "Gründungsdatum",
                      "registreringsdato", "fecha de inscripción", "fecha de registro"]
    },
    "description": {
        "description": "A free-text description of what the company does, its activities, or its purpose",
        "data_type": "text",
        "required": False,
        "examples": ["Manufacture of motor vehicles", "Retail sale of clothing", "Software consulting"],
        "synonyms": ["business description", "activity description", "purpose", "objeto social",
                      "finalidad", "activité", "Gegenstand", "business purpose"]
    },
}


# ── Company type lookups ─────────────────────────────────────────────────────

COMPANY_TYPE_CATEGORIES = {
    code: info["label"]
    for code, info in CANONICAL_SCHEMA["company_type"]["allowed_values"].items()
}

COMPANY_SUB_TYPES = {}
for _cat_code, _cat_info in CANONICAL_SCHEMA["company_type"]["allowed_values"].items():
    for _sub_code, _sub_label in _cat_info["sub_types"].items():
        COMPANY_SUB_TYPES[_sub_code] = {
            "label": _sub_label,
            "category": _cat_code,
        }

# ── Status lookups ───────────────────────────────────────────────────────────

STATUS_CODES = {
    code: info["label"]
    for code, info in CANONICAL_SCHEMA["status"]["allowed_values"].items()
}

LIFE_CYCLE_STAGES = {}
for _code, _info in CANONICAL_SCHEMA["status"]["allowed_values"].items():
    stage = _info["life_cycle_stage"]
    LIFE_CYCLE_STAGES.setdefault(stage, []).append(_code)


# ── Helper functions ─────────────────────────────────────────────────────────

def get_schema_summary():
    """
    Returns a simplified version of the schema for the AI to understand.
    Used in prompts to the LLM.
    """
    summary = {}
    for field_name, field_info in CANONICAL_SCHEMA.items():
        entry = {
            "description": field_info["description"],
            "examples": field_info["examples"],
            "synonyms": field_info["synonyms"],
        }
        # Include allowed values for categorical fields so the AI knows valid options
        if field_info["data_type"] == "categorical":
            entry["allowed_values"] = field_info["allowed_values"]
        if field_info["data_type"] == "object" and "schema" in field_info:
            entry["schema"] = field_info["schema"]
        summary[field_name] = entry
    return summary


def get_required_fields():
    """Returns list of required field names"""
    return [name for name, info in CANONICAL_SCHEMA.items() if info.get("required", False)]


def get_categorical_fields():
    """Returns dict of categorical fields and their allowed values"""
    return {
        name: info["allowed_values"]
        for name, info in CANONICAL_SCHEMA.items()
        if info["data_type"] == "categorical"
    }


def get_company_category_for_subtype(sub_type_code: str) -> str | None:
    """Given a 3-letter sub-type code, return its parent category letter (A-T) or None."""
    entry = COMPANY_SUB_TYPES.get(sub_type_code)
    return entry["category"] if entry else None


def get_statuses_by_lifecycle(stage: str) -> list[str]:
    """Return all status codes for a given life-cycle stage (pre-life, life, dying, died)."""
    return LIFE_CYCLE_STAGES.get(stage, [])
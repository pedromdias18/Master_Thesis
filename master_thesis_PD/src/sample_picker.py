"""
Sample Picker: Automatically find the best sample records from a bulk JSON file.

Scans all records and picks N samples that together cover the maximum number
of non-empty fields. This ensures the LLM sees every possible field at least once.

Usage (from project root):
    python src/sample_picker.py
"""

import json
import sys
from pathlib import Path


def count_non_empty(obj, prefix=""):
    """
    Recursively count all non-empty leaf fields in a nested object.
    Returns a set of field paths that have actual data.
    
    Example: {"Corp": {"CompanyName": "Acme", "TaxId": ""}}
    Returns: {"Corp.CompanyName"}  (TaxId is empty so excluded)
    """
    filled = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            child_filled = count_non_empty(value, path)
            if child_filled:
                filled.update(child_filled)
            # If dict value is a simple non-empty value
            elif not isinstance(value, (dict, list)) and _has_value(value):
                filled.add(path)

    elif isinstance(obj, list):
        if len(obj) > 0:
            # For lists, check if any element has data
            # Use the first element to get field paths
            for i, element in enumerate(obj[:3]):  # Check first 3 elements max
                child_filled = count_non_empty(element, prefix)
                filled.update(child_filled)
            # Also mark the list itself as "has data"
            if filled or any(_has_value(x) for x in obj if not isinstance(x, (dict, list))):
                filled.add(prefix)

    elif _has_value(obj):
        filled.add(prefix)

    return filled


def _has_value(value):
    """Check if a value is meaningfully non-empty."""
    if value is None:
        return False
    if isinstance(value, bool):
        return True  # Booleans are always meaningful
    if isinstance(value, (int, float)):
        return True  # Numbers are always meaningful (even 0)
    if isinstance(value, str):
        s = value.strip()
        return s != '' and s != '-' and s.lower() not in ('none', 'null', 'n/a', 'empty')
    return False


def pick_best_samples(json_path, n_samples=3, scan_limit=None):
    """
    Pick the N records from a JSON array file that together cover the most fields.
    
    Uses a greedy algorithm:
    1. Score every record by how many non-empty fields it has
    2. Pick the record with the most filled fields
    3. Pick the next record that fills the most NEW fields (not already covered)
    4. Repeat until we have N samples
    
    Args:
        json_path: Path to the bulk JSON file (array of objects)
        n_samples: How many samples to pick (default 3)
        scan_limit: Only scan first N records (None = scan all, useful for huge files)
    
    Returns:
        list of (index, record, filled_fields_set) tuples
    """
    print(f"  Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    to_scan = total if scan_limit is None else min(scan_limit, total)
    print(f"  Total records: {total:,}, scanning: {to_scan:,}")

    # Score each record
    print(f"  Analyzing field coverage...")
    scored = []
    all_possible_fields = set()

    for i in range(to_scan):
        filled = count_non_empty(data[i])
        scored.append((i, filled))
        all_possible_fields.update(filled)

    print(f"  Total unique field paths found across all records: {len(all_possible_fields)}")

    # Greedy selection: pick records that maximize NEW field coverage
    selected = []
    covered = set()

    for pick_round in range(n_samples):
        best_idx = None
        best_new_count = -1
        best_filled = None

        for i, filled in scored:
            # How many NEW fields does this record add?
            new_fields = filled - covered
            if len(new_fields) > best_new_count:
                best_new_count = len(new_fields)
                best_idx = i
                best_filled = filled

        if best_idx is not None:
            covered.update(best_filled)
            selected.append((best_idx, data[best_idx], best_filled))
            # Remove from candidates
            scored = [(i, f) for i, f in scored if i != best_idx]
            print(f"  Sample {pick_round+1}: record #{best_idx} "
                  f"({len(best_filled)} fields filled, {best_new_count} new)")

    # Summary
    uncovered = all_possible_fields - covered
    print(f"\n  Coverage: {len(covered)}/{len(all_possible_fields)} field paths covered")
    if uncovered:
        print(f"  Uncovered fields ({len(uncovered)}):")
        for f in sorted(uncovered)[:20]:
            print(f"    - {f}")
        if len(uncovered) > 20:
            print(f"    ... and {len(uncovered) - 20} more")

    return selected, data


def main():
    """Pick best samples for all 3 countries and save them."""
    
    raw_dir = Path('data/raw')
    output_dir = Path('data/samples')
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = {
        "Myanmar": raw_dir / 'mm_entities.json',
        "Norway": raw_dir / 'no_entities.json',
        "Honduras": raw_dir / 'hn_entities.json',
    }

    for country, filepath in datasets.items():
        print(f"\n{'='*60}")
        print(f"  {country}")
        print(f"{'='*60}")

        if not filepath.exists():
            print(f"  ERROR: File not found: {filepath}")
            continue

        selected, _ = pick_best_samples(filepath, n_samples=3)

        # Save the sample records
        samples = [record for _, record, _ in selected]
        out_path = output_dir / f"{country.lower()}_samples.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(samples)} samples to: {out_path}")

    print(f"\n{'='*60}")
    print("Done! Samples saved to data/samples/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
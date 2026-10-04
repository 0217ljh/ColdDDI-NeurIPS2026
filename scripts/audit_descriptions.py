"""Audit clinical descriptions and optionally mask entity names."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MASK_TOKEN = "[MASKED_ENTITY]"
# Apply masking patterns in experiment order.
ENTITY_PATTERNS = (
    r"CYP\d[A-Z]?\d*", r"UGT\d[A-Z]?\d*", r"MAO(?:-?[AB])?",
    r"monoamine oxidase", r"P-?glycoprotein", r"\bP-?gp\b",
    r"OATP\d?[A-Z]?\d?", r"BCRP", r"ABCG2", r"OCT[123]", r"OAT[13]",
    r"MATE[12]", r"MRP\d?", r"NTCP", r"BSEP", r"HMG-CoA reductase",
    r"VKORC1", r"Angiotensin-converting enzyme", r"\bACE\b",
    r"Cholinesterase", r"Acetylcholinesterase", r"5-HT\d[A-Z]?",
    r"dopamine D\d receptor", r"D\d receptor", r"beta-?[12] receptor",
    r"alpha-?[12] receptor", r"muscarinic M\d", r"GABA-?A receptor",
    r"NMDA receptor", r"AMPA receptor", r"EGFR", r"VEGFR", r"mTOR",
    r"BCR-ABL", r"COX-?[12]", r"cyclooxygenase", r"alpha-?1-acid glycoprotein",
)
DDI_PATTERNS = (
    r"drug[- ]drug interaction", r"drug interactions?",
    r"interactions? with (?:other|another|multiple|many|various|several)? ?(?:medications?|drugs?|agents?)",
    r"interact(?:s|ing)? with (?:other|another|many|several|multiple|various)?\s*(?:medications?|drugs?|agents?)",
    r"has (?:many|numerous|several) interactions?", r"prone to interactions?",
    r"be aware of interactions?", r"potential (?:drug )?interactions?",
    r"significant interactions?", r"known (?:drug )?interactions?",
    r"pharmacokinetic interactions?", r"pharmacodynamic interactions?", r"co-?administrat",
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_text_map(path: Path) -> dict[str, str]:
    """Read a nonempty drug-ID-to-text object without coercing malformed values."""
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a nonempty JSON object.")
    for key, value in data.items():
        if not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: invalid text for {key!r}.")
    return data


def mask_description(text: str, own_entities: list[str]) -> tuple[str, list[dict]]:
    """Mask per-drug KG names and short-form entities, recording substitutions."""
    replacements = []
    rules = [("own_kg", re.escape(term)) for term in
             sorted(set(own_entities), key=lambda term: (-len(term), term))]
    rules += [("short_form", pattern) for pattern in ENTITY_PATTERNS]
    for source, pattern in rules:
        regex = re.compile(r"\b" + pattern + r"\b", re.IGNORECASE)
        matches = list(regex.finditer(text))
        if matches:
            replacements.append({"source": source, "pattern": pattern,
                                 "matches": [m.group(0) for m in matches],
                                 "count": len(matches)})
            text = regex.sub(MASK_TOKEN, text)
    text = re.sub(r"(\[MASKED_ENTITY\])(\s+\[MASKED_ENTITY\])+", MASK_TOKEN, text)
    text = re.sub(r"(\[MASKED_ENTITY\])(?:\s*[,;]\s*\[MASKED_ENTITY\])+", MASK_TOKEN, text)
    return text, replacements


def audit_descriptions(
    descriptions: dict[str, str],
    own_entities: dict[str, list[str]] | None = None,
    drug_names: dict[str, str] | None = None,
) -> dict:
    """Report mask counts and candidate residual mentions without altering the input."""
    records = {}
    for drug_id, text in sorted(descriptions.items()):
        hits = []
        rules = [("short_form", r"\b" + pattern + r"\b") for pattern in ENTITY_PATTERNS]
        rules += [("ddi_phrase", pattern) for pattern in DDI_PATTERNS]
        rules += [("own_kg", r"\b" + re.escape(term) + r"\b")
                  for term in sorted(set((own_entities or {}).get(drug_id, [])))]
        self_name = (drug_names or {}).get(drug_id, "").lower()
        rules += [("other_drug", r"\b" + re.escape(name.strip()) + r"\b")
                  for other_id, name in sorted((drug_names or {}).items())
                  if other_id != drug_id and len(name.strip()) >= 4
                  and name.strip().lower() != self_name]
        for category, pattern in rules:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                hits.append({"category": category, "text": match.group(0),
                             "start": match.start(), "end": match.end()})
        records[drug_id] = {
            "description_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "mask_count": text.count(MASK_TOKEN),
            "candidate_mentions": hits,
        }
    return {
        "audit_type": "masked_description_verification",
        "checks": {"short_form_entities": True, "ddi_phrases": True,
                   "per_drug_kg_entities": own_entities is not None,
                   "other_drug_names": drug_names is not None},
        "summary": {
            "description_count": len(records),
            "descriptions_with_masks": sum(r["mask_count"] > 0 for r in records.values()),
            "mask_count": sum(r["mask_count"] for r in records.values()),
            "descriptions_with_candidate_mentions": sum(bool(r["candidate_mentions"]) for r in records.values()),
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON object: {drug_id: description}.")
    parser.add_argument("--output", required=True, type=Path, help="Audit report JSON.")
    parser.add_argument("--kg-entities", type=Path,
                        help="Optional JSON: {drug_id: [entity names]}.")
    parser.add_argument("--drug-names", type=Path,
                        help="Optional JSON: {drug_id: drug name}.")
    parser.add_argument("--masked-output", type=Path,
                        help="Optionally apply historical masks and save a separate JSON.")
    args = parser.parse_args()
    inputs = [p.resolve() for p in (args.input, args.kg_entities, args.drug_names) if p]
    outputs = [p.resolve() for p in (args.output, args.masked_output) if p]
    if len(set(outputs)) != len(outputs) or any(p in inputs for p in outputs):
        parser.error("Output paths must be distinct and must not overwrite inputs.")
    descriptions = load_text_map(args.input)
    names = load_text_map(args.drug_names) if args.drug_names else None
    entities = None
    if args.kg_entities:
        entities = json.loads(args.kg_entities.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(entities, dict) or any(
            not isinstance(v, list) or any(not isinstance(t, str) or not t.strip() for t in v)
            for v in entities.values()
        ):
            parser.error("--kg-entities must map drug IDs to lists of nonempty entity names.")
        missing = sorted(descriptions.keys() - entities.keys())
        if missing:
            parser.error(f"--kg-entities has no entry for {len(missing)} drugs (use [] for no entities).")
    if names is not None and not descriptions.keys() <= names.keys():
        parser.error("--drug-names must include every input drug.")
    report = audit_descriptions(descriptions, entities, names)
    report["source_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.masked_output:
        masked, operations = {}, {}
        for drug_id, text in sorted(descriptions.items()):
            masked[drug_id], operations[drug_id] = mask_description(text, (entities or {}).get(drug_id, []))
        report["masking_operations"] = operations
        args.masked_output.write_text(json.dumps(masked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

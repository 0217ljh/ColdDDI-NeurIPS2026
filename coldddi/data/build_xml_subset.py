"""Extract a subset of DrugBank XML.

Given the path to a `full database.xml` and a list of DrugBank IDs to keep,
write a smaller XML that preserves the original schema and namespace but
contains only the chosen drug records. `<drug-interactions>` lists are
also pruned: only interactions whose partner is in the keep-set survive,
so the resulting XML is a self-consistent closed graph.

Two intended use cases:

1. **Sample-mode reconstruction.**  Pair the 160-drug list shipped at
   `data-private/subset/subset_drug_ids.txt` with this script to produce
   `data-private/subset/drugbank_subset_160drugs.xml`. The result is
   user-local (`data-private/` is git-ignored) and license-isolated.
2. **Toy fixture.**  A much smaller (~25-drug) subset is shipped at
   `tests/fixtures/drugbank_toy.xml` so reviewers can run the entire
   pipeline end-to-end without obtaining a DrugBank academic licence
   first. Per the paper's reproducibility appendix, the toy is a minimal
   real subset rather than a synthetic stand-in (CC-BY-NC 4.0 allows
   small-scale non-commercial redistribution with attribution).

Public surface
--------------
- :func:`extract_subset`
- :func:`main` — CLI:
  ``python -m coldddi.data.build_xml_subset --xml PATH --drugs-list PATH --out PATH``
"""

from __future__ import annotations

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DRUGBANK_NS: str = "http://www.drugbank.ca"
NS_PREFIX: str = f"{{{DRUGBANK_NS}}}"

ET.register_namespace("", DRUGBANK_NS)


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    if tag.startswith(NS_PREFIX):
        return tag[len(NS_PREFIX):]
    return tag


def _primary_id(drug: ET.Element) -> str | None:
    for did in drug.findall(f"{NS_PREFIX}drugbank-id"):
        if did.attrib.get("primary") == "true":
            return (did.text or "").strip() or None
    return None


def _read_drugs_list(path: Path) -> set[str]:
    """Read a one-DrugBank-ID-per-line text file into a set."""
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def extract_subset(
    xml_path: Path,
    keep_drugs: set[str],
    out_path: Path,
    *,
    verbose: bool = True,
) -> dict[str, int]:
    """Stream-extract a subset of `xml_path` covering only `keep_drugs`.

    Parameters
    ----------
    xml_path
        Path to the full ``drugbank_5.1.13.xml`` (or any DrugBank release).
    keep_drugs
        Set of primary DrugBank IDs (e.g. ``{"DB00001", "DB00006", ...}``)
        to retain in the output.
    out_path
        Destination XML file. Parent directories are created on demand.
    verbose
        If True, print progress every 50 retained drugs.

    Returns
    -------
    dict
        ``{"n_total_drugs_in_source", "n_drugs_kept", "n_drug_interactions_kept"}``.
    """
    n_total = 0
    n_kept = 0
    n_di_kept = 0

    # Use ("start", "end") so the first event reaches the *root* element
    # (events=("end",) alone returns the deepest first-finished leaf).
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, src_root = next(context)
    out_root = ET.Element(src_root.tag, attrib=dict(src_root.attrib))

    for event, elem in context:
        if event != "end" or _local_tag(elem) != "drug":
            continue
        n_total += 1
        pid = _primary_id(elem)
        if pid is None or pid not in keep_drugs:
            elem.clear()
            continue

        # Prune drug-interactions in-place: keep only those whose partner
        # is also in `keep_drugs`. Drugs with all partners outside the set
        # end up with an empty <drug-interactions/> element, which is
        # legal under the schema.
        dis_root = elem.find(f"{NS_PREFIX}drug-interactions")
        if dis_root is not None:
            survivors: list[ET.Element] = []
            for di in list(dis_root):
                partner_id = di.findtext(f"{NS_PREFIX}drugbank-id", default="").strip()
                if partner_id in keep_drugs:
                    survivors.append(di)
            for di in list(dis_root):
                dis_root.remove(di)
            for di in survivors:
                dis_root.append(di)
            n_di_kept += len(survivors)

        out_root.append(copy.deepcopy(elem))
        n_kept += 1

        # Clear the source element and root so iterparse can release memory
        # of the ~17,000 unrelated drugs. We've already deep-copied what we
        # need, so clearing is safe.
        elem.clear()
        src_root.clear()

        if verbose and n_kept % 50 == 0:
            print(f"  ... kept {n_kept} drugs so far", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out_root).write(
        str(out_path),
        encoding="utf-8",
        xml_declaration=True,
    )
    if verbose:
        print(
            f"[build_xml_subset] {n_kept}/{n_total} drugs kept; "
            f"{n_di_kept} DDIs kept; written to {out_path}",
            flush=True,
        )
    return {
        "n_total_drugs_in_source": n_total,
        "n_drugs_kept": n_kept,
        "n_drug_interactions_kept": n_di_kept,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a subset of a DrugBank full database XML.",
    )
    parser.add_argument("--xml", required=True, type=Path, help="Source DrugBank full database XML")
    parser.add_argument(
        "--drugs-list",
        required=True,
        type=Path,
        help="Plain text file with one DrugBank ID per line",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output XML path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    args = parser.parse_args(argv)

    keep = _read_drugs_list(args.drugs_list)
    if not keep:
        print(f"No DrugBank IDs found in {args.drugs_list}", file=sys.stderr)
        return 1
    print(f"[build_xml_subset] keeping {len(keep)} drugs from {args.xml}", flush=True)
    extract_subset(args.xml, keep, args.out, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

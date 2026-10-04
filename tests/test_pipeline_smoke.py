"""Test every public pipeline stage on data/public/drugbank_toy.xml.

Expected counts use the fixed 100-drug fixture and Appendix A.1/A.3 criteria.
Module-scoped fixtures share extraction, filtering, and annotation outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_XML = REPO_ROOT / "data" / "public" / "drugbank_toy.xml"

# Ensure `coldddi` is importable when running pytest from any CWD.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    not TOY_XML.exists(),
    reason=f"Toy XML not found at {TOY_XML}",
)


# Fixtures (module-scoped — each pipeline stage runs once for all tests)


@pytest.fixture(scope="module")
def toy_raw_dir(tmp_path_factory: pytest.TempPathFactory):
    """Stage 1a: parse the toy XML once and write the seven raw CSVs."""
    from coldddi.data.extract import parse_drugbank_xml, write_raw_tables

    raw = parse_drugbank_xml(TOY_XML)
    out_dir = tmp_path_factory.mktemp("toy_raw")
    write_raw_tables(raw, out_dir)
    return raw, out_dir


@pytest.fixture(scope="module")
def toy_filter_report(toy_raw_dir):
    """Stage 1b: apply the seven-step filter to the toy."""
    pytest.importorskip("rdkit")
    from coldddi.data.filter import run_filter_pipeline

    raw, _ = toy_raw_dir
    return run_filter_pipeline(raw, verbose=False)


@pytest.fixture(scope="module")
def toy_filtered_dir(toy_filter_report, tmp_path_factory: pytest.TempPathFactory):
    """Stage 1b on disk."""
    from coldddi.data.filter import write_filter_report

    out_dir = tmp_path_factory.mktemp("toy_filtered")
    write_filter_report(toy_filter_report, out_dir)
    return out_dir


@pytest.fixture(scope="module")
def toy_pkpd_labels(toy_filtered_dir):
    """Stage 2a: PK/PD keyword labelling on the 24 retained types."""
    from coldddi.annotations.pkpd_keywords import label_ddi_types

    edges = pd.read_csv(toy_filtered_dir / "ddi_edges.csv")
    return label_ddi_types(edges["ddi_type"])


@pytest.fixture(scope="module")
def toy_ab_result(toy_filtered_dir, toy_pkpd_labels):
    """Stage 2b: A/B subdivision."""
    from coldddi.annotations.ab_subdivision import run_ab_subdivision

    edges = pd.read_csv(toy_filtered_dir / "ddi_edges.csv")
    drugs = pd.read_csv(toy_filtered_dir / "drugs.csv")
    id2name = dict(zip(drugs["drugbank_id"], drugs["name"]))
    per_pair, per_type = run_ab_subdivision(
        ddi_edges=edges,
        pk_pd_labels=toy_pkpd_labels,
        enzymes_csv=toy_filtered_dir / "drug_enzymes.csv",
        targets_csv=toy_filtered_dir / "drug_targets.csv",
        transporters_csv=toy_filtered_dir / "drug_transporters.csv",
        carriers_csv=toy_filtered_dir / "drug_carriers.csv",
        drug_id_to_name=id2name,
        verbose=False,
    )
    return per_pair, per_type


@pytest.fixture(scope="module")
def toy_type_a_tables(toy_ab_result):
    """Stage 2c: derive mediating_entities + action_pairs."""
    from coldddi.annotations.derive_type_a_tables import derive_type_a_tables

    per_pair, _ = toy_ab_result
    return derive_type_a_tables(per_pair)


# Stage 1a — extract


class TestExtract:
    EXPECTED_CSVS = (
        "drugs.csv",
        "ddi_edges.csv",
        "drug_enzymes.csv",
        "drug_targets.csv",
        "drug_transporters.csv",
        "drug_carriers.csv",
        "drug_pathways.csv",
    )

    def test_drug_count_is_100(self, toy_raw_dir):
        raw, _ = toy_raw_dir
        assert len(raw.drugs) == 100

    def test_edge_count_after_dedupe_is_1540(self, toy_raw_dir):
        raw, _ = toy_raw_dir
        assert len(raw.edges) == 1540

    def test_seven_csvs_written(self, toy_raw_dir):
        _, out_dir = toy_raw_dir
        for fname in self.EXPECTED_CSVS:
            assert (out_dir / fname).is_file(), f"missing {fname}"

    def test_drug_schema(self, toy_raw_dir):
        raw, _ = toy_raw_dir
        assert set(raw.drugs.columns) == {
            "drugbank_id", "name", "type", "smiles", "groups",
        }

    def test_edge_schema_present(self, toy_raw_dir):
        raw, _ = toy_raw_dir
        assert {"drug_a_id", "drug_b_id", "drug_a_name", "description"}.issubset(
            raw.edges.columns
        )

    def test_entity_csvs_have_drugbank_id_first(self, toy_raw_dir):
        _, out_dir = toy_raw_dir
        for fname in (
            "drug_enzymes.csv",
            "drug_targets.csv",
            "drug_transporters.csv",
            "drug_carriers.csv",
            "drug_pathways.csv",
        ):
            df = pd.read_csv(out_dir / fname)
            assert df.columns[0] == "drugbank_id", f"{fname} first column is {df.columns[0]}"


def test_parse_empty_xml_has_fixed_schema(tmp_path):
    """Empty XML produces tables with the required columns."""
    from coldddi.data.extract import parse_drugbank_xml

    empty_xml = tmp_path / "empty.xml"
    empty_xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<drugbank xmlns="http://www.drugbank.ca" version="5.1"></drugbank>\n'
    )
    raw = parse_drugbank_xml(empty_xml)
    assert list(raw.drugs.columns) == ["drugbank_id", "name", "type", "smiles", "groups"]
    assert list(raw.edges.columns) == ["drug_a_id", "drug_b_id", "drug_a_name", "description"]
    assert len(raw.drugs) == 0
    assert len(raw.edges) == 0


# Stage 1b — filter


class TestFilter:
    """Final Step-7 numbers must stay locked at 86 / 1383 / 24."""

    def test_step7_paper_alignment(self, toy_filter_report):
        final = toy_filter_report.step_stats[-1]
        assert final.step == 7
        assert final.n_drugs == 86
        assert final.n_edges == 1383
        assert final.n_types == 24

    def test_aux_tables_pruned_to_final_drugs(self, toy_filter_report):
        final_drugs = set(toy_filter_report.drugs["drugbank_id"])
        for tbl_name in (
            "enzymes", "targets", "transporters", "carriers", "pathways",
        ):
            tbl = getattr(toy_filter_report, tbl_name)
            assert set(tbl["drugbank_id"]).issubset(final_drugs), (
                f"{tbl_name} contains rows for drugs that did not survive Step 7"
            )

    def test_filtered_dir_writes_type_to_text_json(self, toy_filtered_dir):
        # Persist type_to_text with the filter report.
        assert (toy_filtered_dir / "type_to_text.json").is_file()

    def test_filtered_dir_writes_seven_csvs_and_stats(self, toy_filtered_dir):
        for fname in TestExtract.EXPECTED_CSVS:
            assert (toy_filtered_dir / fname).is_file()
        assert (toy_filtered_dir / "stats.json").is_file()


# Stage 2a — pkpd_keywords


class TestPKPD:
    def test_24_types_with_pd_17_pk_7(self, toy_pkpd_labels):
        assert len(toy_pkpd_labels) == 24
        counts = toy_pkpd_labels["pk_pd_label"].value_counts().to_dict()
        assert counts.get("PD") == 17
        assert counts.get("PK") == 7

    def test_empty_input_returns_fixed_schema(self):
        from coldddi.annotations.pkpd_keywords import label_ddi_types

        df = label_ddi_types([])
        assert list(df.columns) == [
            "ddi_type", "pk_pd_label", "matched_pk_keywords", "matched_pd_keywords",
        ]
        assert len(df) == 0


# Stage 2b — ab_subdivision


class TestABSubdivision:
    def test_total_pairs_1383(self, toy_ab_result):
        per_pair, _ = toy_ab_result
        assert len(per_pair) == 1383

    def test_type_a_count_is_523(self, toy_ab_result):
        per_pair, _ = toy_ab_result
        assert int(per_pair["has_key_entity"].sum()) == 523

    def test_per_pair_schema(self, toy_ab_result):
        per_pair, _ = toy_ab_result
        for col in (
            "drug_a_id", "drug_b_id", "ddi_type", "pk_pd_label",
            "key_entity_id", "key_entity_name", "key_entity_type",
            "action_drug_a", "action_drug_b", "match_pattern",
            "mechanism_chain", "chain_type", "confidence",
            "has_key_entity", "key_entity_candidates",
        ):
            assert col in per_pair.columns

    def test_summary_table_24_types(self, toy_ab_result):
        _, per_type = toy_ab_result
        assert len(per_type) == 24

    def test_required_csv_missing_raises_fast(
        self, tmp_path, toy_filtered_dir, toy_pkpd_labels
    ):
        """Missing required entity CSVs fail early."""
        from coldddi.annotations.ab_subdivision import run_ab_subdivision

        edges = pd.read_csv(toy_filtered_dir / "ddi_edges.csv")
        drugs = pd.read_csv(toy_filtered_dir / "drugs.csv")
        id2name = dict(zip(drugs["drugbank_id"], drugs["name"]))
        with pytest.raises(FileNotFoundError):
            run_ab_subdivision(
                ddi_edges=edges,
                pk_pd_labels=toy_pkpd_labels,
                enzymes_csv=tmp_path / "missing_enzymes.csv",
                targets_csv=toy_filtered_dir / "drug_targets.csv",
                transporters_csv=toy_filtered_dir / "drug_transporters.csv",
                carriers_csv=toy_filtered_dir / "drug_carriers.csv",
                drug_id_to_name=id2name,
                verbose=False,
            )


# Stage 2c — derive_type_a_tables


class TestDeriveTypeA:
    def test_mediating_entities_523_rows(self, toy_type_a_tables):
        mediating, _ = toy_type_a_tables
        assert len(mediating) == 523

    def test_action_pairs_523_rows(self, toy_type_a_tables):
        _, action = toy_type_a_tables
        assert len(action) == 523

    def test_mediating_schema(self, toy_type_a_tables):
        mediating, _ = toy_type_a_tables
        assert list(mediating.columns) == [
            "drug_a_id", "drug_b_id", "pk_pd_label",
            "entity_id", "entity_name", "entity_type",
        ]

    def test_action_pairs_schema(self, toy_type_a_tables):
        _, action = toy_type_a_tables
        assert list(action.columns) == [
            "drug_a_id", "drug_b_id", "pk_pd_label",
            "action_drug_a", "action_drug_b", "match_pattern",
            "mechanism_chain", "chain_type", "confidence",
        ]

    def test_string_bool_round_trip(self):
        """CSV-roundtripped 'False' must not be truthy."""
        from coldddi.annotations.derive_type_a_tables import derive_type_a_tables

        df = pd.DataFrame(
            {
                "drug_a_id": ["A1", "B1", "C1"],
                "drug_b_id": ["A2", "B2", "C2"],
                "pk_pd_label": ["PK", "PD", "PK"],
                "has_key_entity": ["True", "False", "true"],  # CSV round-trip
                "key_entity_id": ["E1", "E2", "E3"],
                "key_entity_name": ["n1", "n2", "n3"],
                "key_entity_type": ["enzyme", "target", "enzyme"],
                "action_drug_a": ["inhibitor", "agonist", "inhibitor"],
                "action_drug_b": ["substrate", "agonist", "substrate"],
                "match_pattern": ["p", "p", "p"],
                "mechanism_chain": ["c", "c", "c"],
                "chain_type": ["t", "t", "t"],
                "confidence": ["high", "high", "high"],
            }
        )
        mediating, action = derive_type_a_tables(df)
        # astype(bool) would wrongly retain the non-empty string "False".
        assert len(mediating) == 2
        assert len(action) == 2
        assert set(mediating["drug_a_id"]) == {"A1", "C1"}

    def test_string_bool_unknown_value_raises(self):
        from coldddi.annotations.derive_type_a_tables import derive_type_a_tables

        df = pd.DataFrame(
            {
                "drug_a_id": ["A"],
                "drug_b_id": ["B"],
                "pk_pd_label": ["PK"],
                "has_key_entity": ["maybe"],
                "key_entity_id": ["E"],
                "key_entity_name": ["n"],
                "key_entity_type": ["enzyme"],
                "action_drug_a": ["inhibitor"],
                "action_drug_b": ["substrate"],
                "match_pattern": ["p"],
                "mechanism_chain": ["c"],
                "chain_type": ["t"],
                "confidence": ["high"],
            }
        )
        with pytest.raises(ValueError, match="cannot be parsed as bool"):
            derive_type_a_tables(df)


# build_xml_subset — root-tag regression


class TestBuildXMLSubset:
    """Subset XML retains the <drugbank> root."""

    def test_toy_root_tag_is_drugbank(self):
        import xml.etree.ElementTree as ET

        root = ET.parse(TOY_XML).getroot()
        # Strip namespace before comparing.
        tag = root.tag.split("}", 1)[-1] if "}" in root.tag else root.tag
        assert tag == "drugbank"

    def test_toy_root_namespace_present(self):
        import xml.etree.ElementTree as ET

        root = ET.parse(TOY_XML).getroot()
        assert root.tag.startswith("{http://www.drugbank.ca}")


# Stage 3 — release_parquet


@pytest.fixture(scope="module")
def toy_enriched_dir(
    toy_filtered_dir,
    toy_pkpd_labels,
    toy_ab_result,
    toy_type_a_tables,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Materialize a toy enriched directory on disk for Stage 3 to consume."""
    out = tmp_path_factory.mktemp("toy_enriched")
    toy_pkpd_labels.to_csv(out / "ddi_pk_pd_labels.csv", index=False)
    per_pair, per_type = toy_ab_result
    per_pair.to_csv(out / "ddi_key_entities.csv", index=False)
    per_type.to_csv(out / "ddi_key_entities_type_summary.csv", index=False)
    mediating, action = toy_type_a_tables
    mediating.to_parquet(out / "mediating_entities.parquet", index=False)
    action.to_parquet(out / "action_pairs.parquet", index=False)
    return out


class TestReleaseParquet:
    def test_sample_mode_writes_four_parquets(self, toy_enriched_dir, tmp_path):
        from coldddi.data.release_parquet import dump_release_parquets

        out = tmp_path / "release"
        written = dump_release_parquets(
            enriched_dir=toy_enriched_dir,
            out_dir=out,
            mode="sample",
        )
        assert written["pkpd"].name == "pkpd.parquet"
        assert written["ab"].name == "ab_sample.parquet"
        assert written["mediating_entities"].name == "mediating_entities_sample.parquet"
        assert written["action_pairs"].name == "action_pairs_sample.parquet"
        for path in written.values():
            assert path.is_file()

    def test_full_mode_omits_sample_suffix(self, toy_enriched_dir, tmp_path):
        from coldddi.data.release_parquet import dump_release_parquets

        out = tmp_path / "release_full"
        written = dump_release_parquets(
            enriched_dir=toy_enriched_dir,
            out_dir=out,
            mode="full",
        )
        assert written["ab"].name == "ab.parquet"
        assert written["mediating_entities"].name == "mediating_entities.parquet"
        assert written["action_pairs"].name == "action_pairs.parquet"

    def test_pkpd_parquet_row_count(self, toy_enriched_dir, tmp_path):
        from coldddi.data.release_parquet import dump_release_parquets

        out = tmp_path / "rel"
        written = dump_release_parquets(
            enriched_dir=toy_enriched_dir, out_dir=out, mode="sample"
        )
        df = pd.read_parquet(written["pkpd"])
        # Stage 3 takes the toy's 24 retained PK/PD types from enriched_dir.
        assert len(df) == 24
        assert {"ddi_type", "pk_pd_label"}.issubset(df.columns)

    def test_ab_sample_parquet_row_count(self, toy_enriched_dir, tmp_path):
        from coldddi.data.release_parquet import dump_release_parquets

        out = tmp_path / "rel"
        written = dump_release_parquets(
            enriched_dir=toy_enriched_dir, out_dir=out, mode="sample"
        )
        df = pd.read_parquet(written["ab"])
        assert len(df) == 1383

    def test_ab_has_key_entity_is_real_bool(self, toy_enriched_dir, tmp_path):
        """Parquet preserves bool dtype without string coercion."""
        from coldddi.data.release_parquet import dump_release_parquets

        out = tmp_path / "rel"
        written = dump_release_parquets(
            enriched_dir=toy_enriched_dir, out_dir=out, mode="sample"
        )
        df = pd.read_parquet(written["ab"])
        assert df["has_key_entity"].dtype == bool

    def test_mediating_and_action_parquets_523_rows(self, toy_enriched_dir, tmp_path):
        from coldddi.data.release_parquet import dump_release_parquets

        out = tmp_path / "rel"
        written = dump_release_parquets(
            enriched_dir=toy_enriched_dir, out_dir=out, mode="sample"
        )
        assert pd.read_parquet(written["mediating_entities"]).shape[0] == 523
        assert pd.read_parquet(written["action_pairs"]).shape[0] == 523

    def test_full_pkpd_override(self, toy_enriched_dir, tmp_path):
        """Sample mode can pull pkpd from a separate full source."""
        from coldddi.data.release_parquet import dump_release_parquets

        # Synthesize a "full" pkpd csv with a sentinel row count
        full_pkpd = tmp_path / "full_pkpd.csv"
        pd.DataFrame(
            {
                "ddi_type": [f"t{i}" for i in range(215)],
                "pk_pd_label": ["PD"] * 215,
                "matched_pk_keywords": [""] * 215,
                "matched_pd_keywords": [""] * 215,
            }
        ).to_csv(full_pkpd, index=False)

        out = tmp_path / "rel"
        written = dump_release_parquets(
            enriched_dir=toy_enriched_dir,
            out_dir=out,
            mode="sample",
            full_pkpd_csv=full_pkpd,
        )
        assert pd.read_parquet(written["pkpd"]).shape[0] == 215


class TestShippedReleaseParquets:
    """Sanity-check the parquets that actually ship in `annotations/`."""

    def setup_method(self):
        self.dir = REPO_ROOT / "annotations"

    @pytest.mark.skipif(
        not (REPO_ROOT / "annotations" / "pkpd.parquet").exists(),
        reason="pkpd.parquet not generated yet",
    )
    def test_pkpd_full_215_rows(self):
        df = pd.read_parquet(self.dir / "pkpd.parquet")
        assert len(df) == 215  # full PK/PD type table — license-safe

    @pytest.mark.skipif(
        not (REPO_ROOT / "annotations" / "ab_sample.parquet").exists(),
        reason="ab_sample.parquet not generated yet",
    )
    def test_ab_sample_matches_toy(self):
        df = pd.read_parquet(self.dir / "ab_sample.parquet")
        assert len(df) == 1383
        assert df["has_key_entity"].dtype == bool

"""Validate immutable resources and the fixed manuscript experiment contracts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_contracts(root=ROOT):
    data = pd.read_csv(root / "figures/figure2b/source_data/input_scored_runs.csv")
    contract = json.loads((root / "figures/figure2b/figure_contract.json").read_text())["F2b"]
    assert len(data) == 255 and data.method.nunique() == 17
    assert not data.duplicated(["method", "cell", "seed"]).any()
    assert set(data.method) == set(contract["methods"])
    assert set(data.seed) == {52, 229, 564}
    reference = pd.read_csv(root / "figures/figure2b/source_data/F2b_overall_ranking.csv")
    top = reference.iloc[0]
    assert top["method"] == "GRAFT" and int(top.first_place_count) == 16
    assert abs(top.mean_rank - 26 / 21) < 1e-12
    external = pd.read_csv(root / "figures/figure3/source_data/c_overall_ranking.csv")
    external_top = external.iloc[0]
    assert len(external) == 10 and external_top["method"] == "B"
    assert int(external_top.first_place_count) == 9
    assert abs(external_top.mean_rank - 17 / 12) < 1e-12
    assert abs(external_top.r20 - 0.60619973738475) < 1e-12
    axes = None
    drug_sets = []
    for cell in ("A375", "A549", "HT29", "MCF7", "PC3"):
        file = root / f"data/processed/sources/{cell}.npz"
        if not file.is_file():
            continue
        with np.load(file, allow_pickle=False) as z:
            genes = z["genes"].astype(str)
            assert len(genes) == len(set(genes)) == 976
            if axes is not None:
                np.testing.assert_array_equal(axes, genes)
            axes = genes
            names = z["chemical_names"].astype(str)
            assert z["chemical_native"].shape == z["weights_rest"].shape == (len(names), 976)
            np.testing.assert_allclose(z["weights_rest"].sum(1), 1, atol=1e-5)
            drug_sets.append(set(names))
    if len(drug_sets) == 5:
        assert len(set.intersection(*drug_sets)) == 3220
    splits = root / "data/processed/adaptation/splits"
    orders = sorted(splits.glob("*/anchor_order.csv"))
    assert len(orders) == 10
    for file in orders:
        frame = pd.read_csv(file)
        assert frame["draw"].nunique() == 30
    return {"internal_configurations": 17, "internal_run_records": 255, "external_configurations": 10,
            "gene_panel": 976, "adaptation_contexts": 10, "nested_anchor_lists_per_context": 30}


def main():
    manifest = json.loads((ROOT / "resources_manifest.json").read_text())
    missing, failed, checked = [], [], 0
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            missing.append(row["path"])
        elif path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            failed.append(row["path"])
        else:
            checked += 1
    if missing or failed:
        raise SystemExit(json.dumps({"missing": missing, "checksum_mismatch": failed}, indent=2))
    record = {"files_checked": checked, "all_hashes_match": True, **verify_contracts()}
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

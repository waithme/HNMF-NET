"""Precompute molecular region tensors for refined molecule datasets.

The generated cache is keyed by canonical SMILES and can be passed to
data_loader.get_dataloaders(..., region_cache_path=...).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

from data_loader import (
    get_label_columns,
    get_row_smiles,
    resolve_refined_dataset_path,
)
from molecular_regions import build_molecular_regions


DEFAULT_DATASETS = (
    "FreeSolv",
    "ESOL",
    "Lipophilicity",
    "BACE",
    "BBBP",
    "SIDER",
    "Tox21",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute overlapping molecular region tensors."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--root", default="./dataset")
    parser.add_argument("--output_dir", default="./dataset/molecular_regions")
    parser.add_argument("--max_regions", type=int, default=32)
    parser.add_argument(
        "--remove_nan_labels",
        action="store_true",
        help="Skip rows with NaN labels. By default all valid molecules are cached.",
    )
    return parser.parse_args()


def tensor_to_cpu(fields):
    return {
        key: value.detach().cpu()
        for key, value in fields.items()
        if torch.is_tensor(value)
    }


def precompute_dataset(dataset_name, root, output_dir, max_regions, remove_nan_labels):
    csv_path = resolve_refined_dataset_path(dataset_name, root)
    dataframe = pd.read_csv(csv_path)
    label_columns = get_label_columns(dataset_name, dataframe)

    regions = {}
    invalid_smiles = 0
    skipped_nan = 0
    duplicate_smiles = 0

    for _, row in dataframe.iterrows():
        label_values = row[label_columns].to_numpy(dtype=np.float32)
        if remove_nan_labels and np.isnan(label_values).any():
            skipped_nan += 1
            continue

        smiles = get_row_smiles(row)
        if smiles is None:
            invalid_smiles += 1
            continue

        rdBase.DisableLog("rdApp.error")
        try:
            mol = Chem.MolFromSmiles(smiles)
        finally:
            rdBase.EnableLog("rdApp.error")

        if mol is None or mol.GetNumAtoms() == 0:
            invalid_smiles += 1
            continue

        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical_smiles in regions:
            duplicate_smiles += 1
            continue

        fields = build_molecular_regions(mol, max_regions=max_regions)
        regions[canonical_smiles] = tensor_to_cpu(fields)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_regions_max{max_regions}.pt"
    payload = {
        "dataset": dataset_name,
        "source_csv": str(csv_path),
        "max_regions": max_regions,
        "num_molecules": len(regions),
        "invalid_smiles": invalid_smiles,
        "skipped_nan_labels": skipped_nan,
        "duplicate_canonical_smiles": duplicate_smiles,
        "regions": regions,
    }
    torch.save(payload, output_path)
    print(
        f"{dataset_name}: saved {len(regions)} molecules to {output_path} "
        f"(skipped_nan={skipped_nan}, invalid={invalid_smiles}, duplicates={duplicate_smiles})"
    )
    return output_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    for dataset_name in args.datasets:
        precompute_dataset(
            dataset_name=dataset_name,
            root=args.root,
            output_dir=output_dir,
            max_regions=args.max_regions,
            remove_nan_labels=args.remove_nan_labels,
        )


if __name__ == "__main__":
    main()

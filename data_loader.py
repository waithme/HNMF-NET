import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
from rdkit.Chem.BRICS import FindBRICSBonds

try:
    from molecular_regions import build_molecular_regions
except Exception:
    build_molecular_regions = None


REGION_CACHE_FIELDS = (
    "region_atom_index",
    "n_regions",
    "region_type",
    "region_attr",
    "region_edge_index",
    "brics_num_regions",
    "region_degenerate",
)


# ============================================================
# HiGNN original-style featurization
# ============================================================

def onehot_encoding_unk(x, allowable_set):
    """
    Maps inputs not in allowable_set to the last element.
    Same style as HiGNN source/dataset.py.
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


fun_smarts = {
    "Hbond_donor": "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
    "Hbond_acceptor": "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
    "Basic": "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]",
    "Acid": "[C,S](=[O,S,P])-[O;H1,-1]",
    "Halogen": "[F,Cl,Br,I]",
}

FunQuery = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in fun_smarts.items()
}


def tag_pharmacophore(mol):
    for fungrp, qmol in FunQuery.items():
        matches = mol.GetSubstructMatches(qmol)
        match_idxes = []

        for mat in matches:
            match_idxes.extend(mat)

        for i, atom in enumerate(mol.GetAtoms()):
            tag = "1" if i in match_idxes else "0"
            atom.SetProp(fungrp, tag)

    return mol


def tag_scaffold_atoms(mol):
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        match_idxes = mol.GetSubstructMatch(core)
    except Exception:
        match_idxes = []

    for i, atom in enumerate(mol.GetAtoms()):
        tag = "1" if i in match_idxes else "0"
        atom.SetProp("Scaffold", tag)

    return mol


def hignn_atom_attr(mol, explicit_H=False, use_chirality=True, pharmaco=True, scaffold=True):
    """
    Original HiGNN atom features.
    Expected dimension: 46.
    """
    if pharmaco:
        mol = tag_pharmacophore(mol)

    if scaffold:
        mol = tag_scaffold_atoms(mol)

    feat = []

    for atom in mol.GetAtoms():
        results = (
            onehot_encoding_unk(
                atom.GetSymbol(),
                ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl",
                 "As", "Se", "Br", "Te", "I", "At", "other"],
            )
            + onehot_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, "other"])
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + onehot_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                    "other",
                ],
            )
            + [atom.GetIsAromatic()]
        )

        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        if use_chirality:
            try:
                results = (
                    results
                    + onehot_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
                    + [atom.HasProp("_ChiralityPossible")]
                )
            except Exception:
                results = results + [0, 0] + [atom.HasProp("_ChiralityPossible")]

        if pharmaco:
            results = (
                results
                + [int(atom.GetProp("Hbond_donor"))]
                + [int(atom.GetProp("Hbond_acceptor"))]
                + [int(atom.GetProp("Basic"))]
                + [int(atom.GetProp("Acid"))]
                + [int(atom.GetProp("Halogen"))]
            )

        if scaffold:
            results = results + [int(atom.GetProp("Scaffold"))]

        feat.append(results)

    return np.asarray(feat, dtype=np.float32)


def bond_to_feature(bond, use_chirality=True):
    bt = bond.GetBondType()

    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    if use_chirality:
        bond_feats = bond_feats + onehot_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
        )

    return bond_feats


def hignn_bond_attr(mol, use_chirality=True):
    """
    Original HiGNN directed bond features.
    Expected edge feature dimension: 10.
    """
    feat = []
    index = []

    n = mol.GetNumAtoms()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            bond = mol.GetBondBetweenAtoms(i, j)

            if bond is None:
                continue

            feat.append(bond_to_feature(bond, use_chirality=use_chirality))
            index.append([i, j])

    if len(index) == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 10), dtype=np.float32),
        )

    return np.asarray(index, dtype=np.int64), np.asarray(feat, dtype=np.float32)


def hignn_bond_break(mol):
    """
    BRICS bond break used by HiGNN.

    Returns:
        fra_edge_index
        fra_edge_attr
        cluster_index
    """
    try:
        results = np.asarray(sorted(list(FindBRICSBonds(mol))), dtype=object)
    except Exception:
        results = np.asarray([], dtype=object)

    if results.size == 0:
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(mol, asMols=True, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = hignn_bond_attr(mol)
    else:
        bond_to_break = []

        for item in results:
            atom_pair = item[0]
            bond_to_break.append([int(atom_pair[0]), int(atom_pair[1])])

        rwmol = Chem.RWMol(mol)

        for a, b in bond_to_break:
            try:
                rwmol.RemoveBond(a, b)
            except Exception:
                pass

        rwmol = rwmol.GetMol()

        cluster_idx = []
        Chem.rdmolops.GetMolFrags(
            rwmol,
            asMols=True,
            sanitizeFrags=False,
            frags=cluster_idx,
        )

        fra_edge_index, fra_edge_attr = hignn_bond_attr(rwmol)

    cluster_idx = torch.LongTensor(cluster_idx)

    return fra_edge_index, fra_edge_attr, cluster_idx


def convert_to_hignn_data(raw_data):
    """
    Convert a PyG MoleculeNet data object to HiGNN-compatible data.

    This overwrites x / edge_index / edge_attr with HiGNN original-style
    atom/bond features and adds:
        fra_edge_index
        fra_edge_attr
        cluster_index
    """
    smiles = get_smiles(raw_data)
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        node_attr = hignn_atom_attr(mol)
        edge_index, edge_attr = hignn_bond_attr(mol)
        fra_edge_index, fra_edge_attr, cluster_index = hignn_bond_break(mol)

        data = MolHierData()

        data.x = torch.FloatTensor(node_attr)
        data.edge_index = torch.LongTensor(edge_index).t().contiguous()
        data.edge_attr = torch.FloatTensor(edge_attr)

        data.fra_edge_index = torch.LongTensor(fra_edge_index).t().contiguous()
        data.fra_edge_attr = torch.FloatTensor(fra_edge_attr)
        data.cluster_index = torch.LongTensor(cluster_index)

        data.y = raw_data.y
        data.smiles = smiles

        # Keep your own hierarchical fields too, so your HVB model still works.
        atom_to_motif, motif_edge_index, n_motifs = build_motif_info_from_smiles(smiles)
        data.atom_to_motif = atom_to_motif
        data.motif_edge_index = motif_edge_index
        data.n_motifs = torch.tensor([n_motifs], dtype=torch.long)

        return data

    except Exception:
        return None


# ============================================================
# Custom PyG Data for motif graph batching
# ============================================================

class MolHierData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "motif_edge_index":
            return int(self.n_motifs.item())

        if key == "region_atom_index":
            n_regions = int(self.n_regions.view(-1).sum().item())
            return torch.tensor([[self.num_nodes], [n_regions]])

        if key == "region_edge_index":
            return int(self.n_regions.view(-1).sum().item())

        # HiGNN fragment graph edge index is still atom-level after BRICS breaking.
        if key == "fra_edge_index":
            return self.x.size(0)

        # cluster_index is fragment id, should be incremented by number of clusters.
        if key == "cluster_index":
            if self.cluster_index is None or self.cluster_index.numel() == 0:
                return 0
            return int(self.cluster_index.max().item()) + 1

        return super().__inc__(key, value, *args, **kwargs)


# ============================================================
# Basic utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_smiles(data):
    if hasattr(data, "smiles"):
        return data.smiles
    if hasattr(data, "smile"):
        return data.smile
    raise ValueError("Cannot find SMILES in data object.")


# ============================================================
# Robust scaffold split
# ============================================================

def generate_scaffold(smiles: str):
    """
    Robust Murcko scaffold generation.

    Some MoleculeNet molecules may trigger RDKit stereo errors.
    We remove stereochemistry before converting scaffold to SMILES.
    """
    try:
        with rdBase.BlockLogs():
            return MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smiles,
                includeChirality=True,
            )
    except Exception:
        return ""


def scaffold_split(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=0,
):
    """
    Random scaffold split following splits.py.

    Molecules are grouped by chirality-aware Murcko scaffold. Scaffold
    groups are shuffled with ``seed`` and then assigned, as whole groups,
    to validation, test, and finally training.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    scaffolds = defaultdict(list)

    for idx, data in enumerate(dataset):
        smiles = get_smiles(data)
        with rdBase.BlockLogs():
            mol = (
                Chem.MolFromSmiles(str(smiles))
                if smiles is not None
                else None
            )

        if mol is None:
            scaffold = f"__invalid_smiles_{idx}"
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol,
                    includeChirality=True,
                )
            except ValueError:
                scaffold = f"__invalid_scaffold_{idx}"

        scaffolds[scaffold].append(idx)

    rng = np.random.RandomState(seed)
    scaffold_values = list(scaffolds.values())
    scaffold_sets = [
        scaffold_values[index]
        for index in rng.permutation(len(scaffold_values))
    ]

    val_target = int(np.floor(val_ratio * len(dataset)))
    test_target = int(np.floor(test_ratio * len(dataset)))
    train_idx, val_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(val_idx) + len(scaffold_set) <= val_target:
            val_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= test_target:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == len(dataset)

    return train_idx, val_idx, test_idx


def random_split(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=0,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n_total = len(dataset)
    indices = list(range(n_total))

    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


# ============================================================
# Motif construction
# ============================================================

def find_brics_bond_atom_pairs(mol):
    """
    Find BRICS breakable bonds.

    Returns:
        set of atom-index pairs, e.g. {(1, 5), (3, 8)}
    """
    brics_pairs = set()

    try:
        for item in BRICS.FindBRICSBonds(mol):
            atom_pair = item[0]
            a, b = int(atom_pair[0]), int(atom_pair[1])
            brics_pairs.add(tuple(sorted((a, b))))
    except Exception:
        pass

    return brics_pairs

def build_motif_info_from_smiles(smiles):
    """
    Build BRICS motif information from SMILES.
    """
    return build_brics_motif_info_from_smiles(smiles)

def build_brics_motif_info_from_smiles(smiles):
    """
    Build motif information from SMILES.

    Motifs are connected components after removing BRICS breakable bonds.

    Returns:
        atom_to_motif: LongTensor [num_atoms]
        motif_edge_index: LongTensor [2, num_motif_edges]
        n_motifs: int
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        atom_to_motif = torch.zeros(1, dtype=torch.long)
        motif_edge_index = torch.empty((2, 0), dtype=torch.long)
        return atom_to_motif, motif_edge_index, 1

    num_atoms = mol.GetNumAtoms()

    if num_atoms == 0:
        atom_to_motif = torch.zeros(1, dtype=torch.long)
        motif_edge_index = torch.empty((2, 0), dtype=torch.long)
        return atom_to_motif, motif_edge_index, 1

    brics_pairs = find_brics_bond_atom_pairs(mol)

    adjacency = [[] for _ in range(num_atoms)]
    cut_edges = []

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        pair = tuple(sorted((a, b)))

        if pair in brics_pairs:
            cut_edges.append((a, b))
        else:
            adjacency[a].append(b)
            adjacency[b].append(a)

    visited = [False] * num_atoms
    atom_to_motif_list = [-1] * num_atoms
    motif_id = 0

    for start in range(num_atoms):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True

        while stack:
            u = stack.pop()
            atom_to_motif_list[u] = motif_id

            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        motif_id += 1

    n_motifs = max(motif_id, 1)

    motif_edges = set()

    for a, b in cut_edges:
        ma = atom_to_motif_list[a]
        mb = atom_to_motif_list[b]

        if ma != mb:
            motif_edges.add((ma, mb))
            motif_edges.add((mb, ma))

    if len(motif_edges) == 0:
        motif_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        motif_edge_index = torch.tensor(
            list(motif_edges),
            dtype=torch.long,
        ).t().contiguous()

    atom_to_motif = torch.tensor(atom_to_motif_list, dtype=torch.long)

    return atom_to_motif, motif_edge_index, n_motifs


def convert_to_hier_data(raw_data):
    """
    Convert PyG MoleculeNet Data to MolHierData and attach motif information.
    """
    smiles = get_smiles(raw_data)

    atom_to_motif, motif_edge_index, n_motifs = build_motif_info_from_smiles(smiles)

    data = MolHierData()

    for key, value in raw_data:
        data[key] = value

    data.smiles = smiles
    data.atom_to_motif = atom_to_motif
    data.motif_edge_index = motif_edge_index
    data.n_motifs = torch.tensor([n_motifs], dtype=torch.long)

    return data


# ============================================================
# Dataset processing
# ============================================================

LABEL_COLUMNS = {
    "BACE": ["class"],
    "BBBP": ["class"],
    "HIV": ["class"],
    "ClinTox": ["CT_TOX", "FDA_APPROVED"],
    "ESOL": ["measured"],
    "FreeSolv": ["measured"],
    "Lipophilicity": ["measured"],
    "Lipo": ["measured"],
    "Tox21": [
        "NR-AhR",
        "NR-AR-LBD",
        "NR-AR",
        "NR-Aromatase",
        "NR-ER-LBD",
        "NR-ER",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
}

DATASET_FILE_NAMES = {
    "BACE": "refined_BACE.csv",
    "BBBP": "refined_BBBP.csv",
    "ClinTox": "refined_ClinTox.csv",
    "ESOL": "refined_ESOL.csv",
    "FreeSolv": "refined_FreeSolv.csv",
    "HIV": "refined_HIV.csv",
    "Lipophilicity": "refined_Lipophilicity.csv",
    "Lipo": "refined_Lipophilicity.csv",
    "SIDER": "refined_SIDER.csv",
    "Tox21": "refined_Tox21.csv",
}

NON_LABEL_COLUMNS = {
    "CID",
    "SMILES",
    "smiles",
    "mol_id",
    "activity",
}


def resolve_refined_dataset_path(dataset_name, root):
    file_name = DATASET_FILE_NAMES.get(
        dataset_name,
        f"refined_{dataset_name}.csv",
    )

    candidates = [
        Path(root) / file_name,
        Path(root) / "dataset" / file_name,
        Path("dataset") / file_name,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find {file_name}. Looked in: "
        + ", ".join(str(path) for path in candidates)
    )


def get_label_columns(dataset_name, dataframe):
    if dataset_name in LABEL_COLUMNS:
        return LABEL_COLUMNS[dataset_name]

    return [
        column
        for column in dataframe.columns
        if column not in NON_LABEL_COLUMNS
    ]


def get_row_smiles(row):
    if "SMILES" in row and not pd.isna(row["SMILES"]):
        return row["SMILES"]
    if "smiles" in row and not pd.isna(row["smiles"]):
        return row["smiles"]
    return None


def mol_to_local_graph(mol, smiles, y, featurizer="pyg"):
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    if featurizer not in {"pyg", "hignn"}:
        raise ValueError(f"Unsupported featurizer: {featurizer}.")

    try:
        node_attr = hignn_atom_attr(mol)
        edge_index, edge_attr = hignn_bond_attr(mol)

        data = MolHierData()
        data.x = torch.FloatTensor(node_attr)
        data.edge_index = torch.LongTensor(edge_index).t().contiguous()
        data.edge_attr = torch.FloatTensor(edge_attr)

        if featurizer == "hignn":
            fra_edge_index, fra_edge_attr, cluster_index = hignn_bond_break(mol)
            data.fra_edge_index = torch.LongTensor(fra_edge_index).t().contiguous()
            data.fra_edge_attr = torch.FloatTensor(fra_edge_attr)
            data.cluster_index = torch.LongTensor(cluster_index)

        data.y = torch.FloatTensor(y).view(1, -1)
        data.smiles = smiles

        atom_to_motif, motif_edge_index, n_motifs = build_motif_info_from_smiles(
            smiles
        )
        data.atom_to_motif = atom_to_motif
        data.motif_edge_index = motif_edge_index
        data.n_motifs = torch.tensor([n_motifs], dtype=torch.long)

        return data

    except Exception:
        return None


def load_molecular_region_cache(region_cache_path):
    if region_cache_path is None:
        return None

    path = Path(region_cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find molecular region cache: {path}")

    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")

    if not isinstance(cache, dict) or "regions" not in cache:
        raise ValueError(
            f"Invalid molecular region cache format: {path}. "
            "Expected a dict with a 'regions' field."
        )
    return cache


def attach_cached_molecular_regions(data, canonical_smiles, region_cache, mol=None):
    if region_cache is None:
        return data

    fields = region_cache["regions"].get(canonical_smiles)
    if fields is None:
        if mol is None or build_molecular_regions is None:
            return data

        max_regions = int(region_cache.get("max_regions", 32))
        try:
            fields = build_molecular_regions(mol, max_regions=max_regions)
            fields = {
                key: value.detach().cpu()
                for key, value in fields.items()
                if torch.is_tensor(value)
            }
            region_cache["regions"][canonical_smiles] = fields
        except Exception:
            return data

    for key in REGION_CACHE_FIELDS:
        if key in fields:
            setattr(data, key, fields[key].clone())
    return data


def remove_invalid_labels(dataset):
    """
    Remove samples whose labels contain NaN.

    For BACE / BBBP usually not a big issue, but this makes the loader safer.
    """
    new_dataset = []

    for data in dataset:
        y = data.y

        if y is None:
            continue

        if torch.isnan(y.float()).any():
            continue

        new_dataset.append(data)

    return new_dataset


def load_molecule_dataset(
    dataset_name="BACE",
    root="./data",
    remove_nan_labels=True,
    featurizer="pyg",
    region_cache_path=None,
):
    csv_path = resolve_refined_dataset_path(dataset_name, root)
    dataframe = pd.read_csv(csv_path)
    label_columns = get_label_columns(dataset_name, dataframe)
    region_cache = load_molecular_region_cache(region_cache_path)

    dataset = []

    for _, row in dataframe.iterrows():
        smiles = get_row_smiles(row)

        if smiles is None:
            continue

        label_values = row[label_columns].to_numpy(dtype=np.float32)

        if remove_nan_labels and np.isnan(label_values).any():
            continue

        rdBase.DisableLog("rdApp.error")
        try:
            mol = Chem.MolFromSmiles(smiles)
        finally:
            rdBase.EnableLog("rdApp.error")

        if mol is None:
            continue

        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        data = mol_to_local_graph(
            mol=mol,
            smiles=canonical_smiles,
            y=label_values,
            featurizer=featurizer,
        )

        if data is not None:
            data = attach_cached_molecular_regions(
                data=data,
                canonical_smiles=canonical_smiles,
                region_cache=region_cache,
                mol=mol,
            )
            dataset.append(data)

    return dataset


# ============================================================
# Main public function
# ============================================================

def get_dataloaders(
    dataset_name="BACE",
    root="./data",
    batch_size=64,
    split_type="scaffold",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=0,
    num_workers=0,
    remove_nan_labels=True,
    featurizer="pyg",
    region_cache_path=None,
):
    set_seed(seed)

    dataset = load_molecule_dataset(
        dataset_name=dataset_name,
        root=root,
        remove_nan_labels=remove_nan_labels,
        featurizer=featurizer,
        region_cache_path=region_cache_path,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset {dataset_name} is empty after preprocessing.")

    if split_type == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            dataset=dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    elif split_type == "random":
        train_idx, val_idx, test_idx = random_split(
            dataset=dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unsupported split_type: {split_type}. "
            f"Use 'scaffold' or 'random'."
        )

    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    node_dim = dataset[0].x.size(-1)
    edge_dim = dataset[0].edge_attr.size(-1)
    num_tasks = dataset[0].y.view(1, -1).size(-1)

    info = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_tasks": num_tasks,
        "split_type": split_type,
        "seed": seed,
    }

    return train_loader, val_loader, test_loader, info

# ============== test =============
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="BACE")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split_type", type=str, default="scaffold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    train_loader, val_loader, test_loader, info = get_dataloaders(
        dataset_name=args.dataset,
        root=args.root,
        batch_size=args.batch_size,
        split_type=args.split_type,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print("=" * 80)
    print("Dataset Loaded")
    print("=" * 80)

    for k, v in info.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print("Check one training batch")
    print("=" * 80)

    batch = next(iter(train_loader))

    print(batch)
    print("x:", batch.x.shape)
    print("edge_index:", batch.edge_index.shape)
    print("edge_attr:", batch.edge_attr.shape)
    print("y:", batch.y.shape)
    print("batch:", batch.batch.shape)

    print("atom_to_motif:", batch.atom_to_motif.shape)
    print("motif_edge_index:", batch.motif_edge_index.shape)
    print("n_motifs:", batch.n_motifs.shape)
    print("num_graphs:", batch.num_graphs)

    print("=" * 80)
    print("Done")
    print("=" * 80)


#if __name__ == "__main__":
#    main()

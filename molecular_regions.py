"""
Multi-source overlapping molecular region proposal construction.

This replaces BRICS-only disjoint partitioning with overlapping region proposals:
    - BRICS fragments, when BRICS produces more than one component
    - ring systems
    - functional groups via SMARTS
    - atom-centered radius-1 and radius-2 regions
    - one global molecule token

The produced fields are designed for MFP-GINE:
    region_atom_index: [2, num_memberships]
    n_regions: [1] per graph before batching, [B] after batching
    region_type: [num_regions]
    region_attr: [num_regions, 9]
    region_edge_index: [2, num_region_edges]

Important: overlapping regions require a bipartite membership tensor, not the old
atom_to_motif one-to-one partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS
except Exception as exc:  # pragma: no cover
    Chem = None
    BRICS = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    Data = object


# Region type ids. Keep these consistent with MFPGINE(num_region_types=6).
REGION_TYPE = {
    "brics": 0,
    "ring": 1,
    "functional_group": 2,
    "radius1": 3,
    "radius2": 4,
    "global": 5,
}

# Higher priority regions are kept first when max_regions is reached.
REGION_PRIORITY = {
    "functional_group": 0,
    "ring": 1,
    "brics": 2,
    "radius1": 3,
    "radius2": 4,
    "global": 99,
}

# Compact but useful SMARTS set for ESOL/FreeSolv-style small molecules.
# These are region proposals, not labels; overlap and redundancy are allowed before deduplication.
FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H][#6]",
    "phenol": "[OX2H][c]",
    "ether": "[#6][OX2][#6]",
    "primary_amine": "[NX3H2][#6]",
    "secondary_amine": "[NX3H1]([#6])[#6]",
    "tertiary_amine": "[NX3]([#6])([#6])[#6]",
    "nitrile": "[CX2]#N",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfone": "[SX4](=O)(=O)([#6])[#6]",
    "sulfoxide": "[SX3](=O)([#6])[#6]",
    "thiol": "[SX2H][#6]",
    "thioether": "[#6][SX2][#6]",
    "halogen": "[F,Cl,Br,I]",
    "phosphate": "[PX4](=O)([OX2,OX1-])([OX2,OX1-])[OX2,OX1-]",
}


@dataclass(frozen=True)
class RegionCandidate:
    atoms: Tuple[int, ...]
    source: str
    radius: int = 0

    @property
    def type_id(self) -> int:
        return REGION_TYPE[self.source]

    @property
    def priority(self) -> int:
        return REGION_PRIORITY[self.source]


class MolecularRegionData(Data):
    """
    PyG Data subclass with correct batching increments for overlapping regions.

    Use this class when constructing data objects offline. If your current
    dataset already returns vanilla Data objects, you can still attach the
    tensors and call ensure_batch_regions(...) before moving a batch to device.
    """

    def __inc__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key == "region_atom_index":
            n_regions = int(self.n_regions.view(-1).sum().item()) if hasattr(self, "n_regions") else 0
            return torch.tensor([[self.num_nodes], [n_regions]])
        if key == "region_edge_index":
            n_regions = int(self.n_regions.view(-1).sum().item()) if hasattr(self, "n_regions") else 0
            return n_regions
        return super().__inc__(key, value, *args, **kwargs)


# ============================================================
# RDKit helpers
# ============================================================


def _require_rdkit() -> None:
    if Chem is None or BRICS is None:
        raise ImportError(f"RDKit is required for molecular region construction: {_RDKIT_IMPORT_ERROR}")


def mol_from_smiles(smiles: str):
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def _connected_components_after_cut(mol, cut_edges: set[Tuple[int, int]]) -> List[Tuple[int, ...]]:
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (min(a, b), max(a, b))
        if key in cut_edges:
            continue
        adj[a].append(b)
        adj[b].append(a)

    visited = [False] * n
    comps: List[Tuple[int, ...]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(tuple(sorted(comp)))
    return comps


def get_brics_regions(mol) -> List[RegionCandidate]:
    _require_rdkit()
    n = mol.GetNumAtoms()
    brics_bonds = []
    for item in BRICS.FindBRICSBonds(mol):
        # item is commonly ((a, b), (label_a, label_b))
        atoms = item[0]
        if len(atoms) != 2:
            continue
        a, b = int(atoms[0]), int(atoms[1])
        brics_bonds.append((min(a, b), max(a, b)))

    if not brics_bonds:
        return []

    comps = _connected_components_after_cut(mol, set(brics_bonds))
    # If BRICS does not truly split the molecule, do not add a whole-molecule BRICS region.
    if len(comps) <= 1:
        return []

    return [RegionCandidate(atoms=c, source="brics") for c in comps if 0 < len(c) < n]


def get_ring_system_regions(mol) -> List[RegionCandidate]:
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if not rings:
        return []

    # Merge fused/overlapping rings into ring systems.
    merged: List[set[int]] = []
    for ring in rings:
        placed = False
        for system in merged:
            if ring & system:
                system.update(ring)
                placed = True
                break
        if not placed:
            merged.append(set(ring))

    # One more transitive merge pass.
    changed = True
    while changed:
        changed = False
        new_systems: List[set[int]] = []
        while merged:
            base = merged.pop(0)
            i = 0
            while i < len(merged):
                if base & merged[i]:
                    base.update(merged.pop(i))
                    changed = True
                else:
                    i += 1
            new_systems.append(base)
        merged = new_systems

    return [RegionCandidate(atoms=tuple(sorted(s)), source="ring") for s in merged]


def get_functional_group_regions(mol) -> List[RegionCandidate]:
    _require_rdkit()
    regions: List[RegionCandidate] = []
    for _, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            atoms = tuple(sorted(set(int(i) for i in match)))
            if atoms:
                regions.append(RegionCandidate(atoms=atoms, source="functional_group"))
    return regions


def get_radius_regions(mol, radii: Sequence[int] = (1, 2)) -> List[RegionCandidate]:
    n = mol.GetNumAtoms()
    if n == 0:
        return []
    dist = Chem.GetDistanceMatrix(mol)
    regions: List[RegionCandidate] = []
    for r in radii:
        source = f"radius{r}"
        if source not in REGION_TYPE:
            continue
        for center in range(n):
            atoms = tuple(sorted(int(i) for i in range(n) if dist[center, i] <= r))
            # Skip singletons and whole-molecule duplicates. Global token handles whole molecule.
            if 1 < len(atoms) < n:
                regions.append(RegionCandidate(atoms=atoms, source=source, radius=r))
    return regions


def get_global_region(mol) -> RegionCandidate:
    return RegionCandidate(atoms=tuple(range(mol.GetNumAtoms())), source="global")


# ============================================================
# Region graph and features
# ============================================================


def _canonical_atoms(atoms: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(int(a) for a in atoms)))


def deduplicate_and_limit_regions(
    candidates: Sequence[RegionCandidate],
    n_atoms: int,
    max_regions: int = 32,
    include_global: bool = True,
) -> List[RegionCandidate]:
    """Deduplicate atom sets, keep high-priority regions, always keep global if requested."""
    if max_regions < 1:
        raise ValueError("max_regions must be >= 1")

    global_candidate: Optional[RegionCandidate] = None
    unique: Dict[Tuple[int, ...], RegionCandidate] = {}

    for cand in candidates:
        atoms = _canonical_atoms(cand.atoms)
        if not atoms:
            continue
        fixed = RegionCandidate(atoms=atoms, source=cand.source, radius=cand.radius)
        if cand.source == "global":
            global_candidate = fixed
            continue
        # Avoid non-global whole-molecule regions; global token represents that.
        if len(atoms) >= n_atoms:
            continue
        old = unique.get(atoms)
        if old is None or fixed.priority < old.priority:
            unique[atoms] = fixed

    regions = list(unique.values())
    regions.sort(key=lambda c: (c.priority, len(c.atoms), c.atoms))

    if include_global:
        if global_candidate is None:
            global_candidate = RegionCandidate(atoms=tuple(range(n_atoms)), source="global")
        regions = regions[: max_regions - 1]
        regions.append(global_candidate)
    else:
        regions = regions[:max_regions]

    if not regions:
        regions = [RegionCandidate(atoms=tuple(range(n_atoms)), source="global")]
    return regions


def compute_region_attr(mol, regions: Sequence[RegionCandidate]) -> torch.Tensor:
    """
    Region attributes, 9 dims:
        0 size / n_atoms
        1 log(1+size) / log(1+n_atoms)
        2 topological diameter / max(1, n_atoms-1)
        3 is_brics
        4 is_ring
        5 is_functional_group
        6 is_radius1
        7 is_radius2
        8 is_global
    """
    n = max(1, mol.GetNumAtoms())
    dist = Chem.GetDistanceMatrix(mol) if mol.GetNumAtoms() > 0 else None
    attrs: List[List[float]] = []
    for r in regions:
        atoms = list(r.atoms)
        size = len(atoms)
        if dist is not None and size > 1:
            diam = max(float(dist[i, j]) for i in atoms for j in atoms)
        else:
            diam = 0.0
        attrs.append(
            [
                size / n,
                math.log1p(size) / max(math.log1p(n), 1e-6),
                diam / max(n - 1, 1),
                1.0 if r.source == "brics" else 0.0,
                1.0 if r.source == "ring" else 0.0,
                1.0 if r.source == "functional_group" else 0.0,
                1.0 if r.source == "radius1" else 0.0,
                1.0 if r.source == "radius2" else 0.0,
                1.0 if r.source == "global" else 0.0,
            ]
        )
    return torch.tensor(attrs, dtype=torch.float32)


def build_region_atom_index(regions: Sequence[RegionCandidate]) -> torch.Tensor:
    atom_ids: List[int] = []
    region_ids: List[int] = []
    for rid, r in enumerate(regions):
        for a in r.atoms:
            atom_ids.append(int(a))
            region_ids.append(int(rid))
    if not atom_ids:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([atom_ids, region_ids], dtype=torch.long)


def _has_bond_between(mol, a_set: set[int], b_set: set[int]) -> bool:
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in a_set and b in b_set) or (b in a_set and a in b_set):
            return True
    return False


def build_region_edge_index(mol, regions: Sequence[RegionCandidate]) -> torch.Tensor:
    edges: List[Tuple[int, int]] = []
    sets = [set(r.atoms) for r in regions]
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            overlap = bool(sets[i] & sets[j])
            adjacent = _has_bond_between(mol, sets[i], sets[j])
            contained = sets[i].issubset(sets[j]) or sets[j].issubset(sets[i])
            if overlap or adjacent or contained:
                edges.append((i, j))
                edges.append((j, i))
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ============================================================
# Public API
# ============================================================


def build_molecular_regions(
    mol,
    max_regions: int = 32,
    include_brics: bool = True,
    include_rings: bool = True,
    include_functional_groups: bool = True,
    include_radius_regions: bool = True,
    include_global: bool = True,
) -> Dict[str, torch.Tensor]:
    """Construct all tensors required by MFP-GINE for one molecule."""
    _require_rdkit()
    n_atoms = mol.GetNumAtoms()
    candidates: List[RegionCandidate] = []

    brics_regions = get_brics_regions(mol) if include_brics else []
    ring_regions = get_ring_system_regions(mol) if include_rings else []
    fg_regions = get_functional_group_regions(mol) if include_functional_groups else []
    radius_regions = get_radius_regions(mol, radii=(1, 2)) if include_radius_regions else []

    candidates.extend(fg_regions)
    candidates.extend(ring_regions)
    candidates.extend(brics_regions)
    candidates.extend(radius_regions)
    if include_global:
        candidates.append(get_global_region(mol))

    regions = deduplicate_and_limit_regions(
        candidates=candidates,
        n_atoms=n_atoms,
        max_regions=max_regions,
        include_global=include_global,
    )

    region_atom_index = build_region_atom_index(regions)
    region_type = torch.tensor([r.type_id for r in regions], dtype=torch.long)
    region_attr = compute_region_attr(mol, regions)
    region_edge_index = build_region_edge_index(mol, regions)
    n_regions = torch.tensor([len(regions)], dtype=torch.long)

    # Useful diagnostics.
    brics_num_regions = torch.tensor([len(brics_regions)], dtype=torch.long)
    num_whole_like = torch.tensor([int(len(regions) == 1 or all(len(r.atoms) == n_atoms for r in regions))], dtype=torch.long)

    return {
        "region_atom_index": region_atom_index,
        "n_regions": n_regions,
        "region_type": region_type,
        "region_attr": region_attr,
        "region_edge_index": region_edge_index,
        "brics_num_regions": brics_num_regions,
        "region_degenerate": num_whole_like,
    }


def build_molecular_regions_from_smiles(smiles: str, max_regions: int = 32) -> Dict[str, torch.Tensor]:
    mol = mol_from_smiles(smiles)
    return build_molecular_regions(mol, max_regions=max_regions)


def attach_molecular_regions_to_data(data, mol=None, smiles: Optional[str] = None, max_regions: int = 32):
    """Attach molecular region tensors to a single PyG Data object."""
    if mol is None:
        if smiles is None:
            if hasattr(data, "smiles"):
                smiles = data.smiles
            elif hasattr(data, "smi"):
                smiles = data.smi
            else:
                raise ValueError("Need either `mol`, `smiles`, or data.smiles/data.smi.")
        mol = mol_from_smiles(smiles)

    fields = build_molecular_regions(mol, max_regions=max_regions)
    for key, value in fields.items():
        setattr(data, key, value)
    return data


def ensure_batch_regions(batch, smiles_attr: str = "smiles", max_regions: int = 32):
    """
    Attach batched region tensors on the fly if they are absent.

    This is useful when the existing data loader cannot be modified yet. It
    requires the batch to carry a list-like SMILES attribute and a PyG `ptr`.
    Prefer offline preprocessing for speed in final experiments.
    """
    if hasattr(batch, "region_atom_index") and hasattr(batch, "n_regions"):
        return batch

    if not hasattr(batch, smiles_attr):
        # Let the model fall back to global regions if SMILES are unavailable.
        return batch

    smiles_list = getattr(batch, smiles_attr)
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    if not isinstance(smiles_list, (list, tuple)):
        return batch

    if not hasattr(batch, "ptr"):
        return batch

    device = batch.x.device if hasattr(batch, "x") else torch.device("cpu")
    ptr = batch.ptr.detach().cpu().long().tolist()

    all_memberships: List[torch.Tensor] = []
    all_region_types: List[torch.Tensor] = []
    all_region_attrs: List[torch.Tensor] = []
    all_region_edges: List[torch.Tensor] = []
    n_regions_list: List[int] = []
    region_offset = 0

    for gid, smiles in enumerate(smiles_list):
        fields = build_molecular_regions_from_smiles(smiles, max_regions=max_regions)
        node_offset = ptr[gid]
        region_atom_index = fields["region_atom_index"].clone()
        if region_atom_index.numel() > 0:
            region_atom_index[0] += node_offset
            region_atom_index[1] += region_offset
            all_memberships.append(region_atom_index)

        region_edge_index = fields["region_edge_index"].clone()
        if region_edge_index.numel() > 0:
            region_edge_index += region_offset
            all_region_edges.append(region_edge_index)

        all_region_types.append(fields["region_type"])
        all_region_attrs.append(fields["region_attr"])
        n_r = int(fields["n_regions"].item())
        n_regions_list.append(n_r)
        region_offset += n_r

    batch.region_atom_index = (
        torch.cat(all_memberships, dim=1).to(device)
        if all_memberships
        else torch.empty(2, 0, dtype=torch.long, device=device)
    )
    batch.region_edge_index = (
        torch.cat(all_region_edges, dim=1).to(device)
        if all_region_edges
        else torch.empty(2, 0, dtype=torch.long, device=device)
    )
    batch.region_type = torch.cat(all_region_types, dim=0).to(device)
    batch.region_attr = torch.cat(all_region_attrs, dim=0).to(device)
    batch.n_regions = torch.tensor(n_regions_list, dtype=torch.long, device=device)
    return batch


def summarize_region_degeneracy(smiles_list: Sequence[str], max_regions: int = 32) -> Dict[str, float]:
    """Quick diagnostic for BRICS degeneracy and generated region counts."""
    if len(smiles_list) == 0:
        return {
            "num_molecules": 0,
            "brics_singleton_ratio": float("nan"),
            "mean_regions": float("nan"),
            "median_regions": float("nan"),
        }

    brics_single = []
    region_counts = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        brics_regions = get_brics_regions(mol)
        brics_single.append(1 if len(brics_regions) <= 1 else 0)
        fields = build_molecular_regions(mol, max_regions=max_regions)
        region_counts.append(int(fields["n_regions"].item()))

    counts = torch.tensor(region_counts, dtype=torch.float32)
    return {
        "num_molecules": len(smiles_list),
        "brics_singleton_ratio": float(sum(brics_single) / len(brics_single)),
        "mean_regions": float(counts.mean().item()),
        "median_regions": float(counts.median().item()),
        "min_regions": float(counts.min().item()),
        "max_regions": float(counts.max().item()),
    }

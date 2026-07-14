# the best version
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter


# ============================================================
# Pooling
# ============================================================

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, batch):
        """
        x:     [N, D]
        batch: [N]
        return: [B, D]
        """
        raw_score = self.score(x).view(-1)

        max_score = scatter(raw_score, batch, dim=0, reduce="max")[batch]
        exp_score = torch.exp(raw_score - max_score)
        denom = scatter(exp_score, batch, dim=0, reduce="sum")[batch].clamp(min=1e-9)
        alpha = exp_score / denom

        pooled = scatter(x * alpha.view(-1, 1), batch, dim=0, reduce="sum")
        return pooled


# ============================================================
# Atom-level GINE encoder
# ============================================================

class GINEHierEncoder(nn.Module):
    """
    Shared multi-layer GINE encoder.

    It returns three node-level representations:
        h1: output of layer 1
        h2: output of layer 2
        h3: output of the final layer
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        for _ in range(3):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        edge_attr = edge_attr.float()

        h = self.node_encoder(x)
        layer_outputs = []

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res
            layer_outputs.append(h)

        h1 = layer_outputs[0]
        h2 = layer_outputs[1]
        h3 = layer_outputs[-1]
        return h1, h2, h3


# ============================================================
# Motif-level GNN
# ============================================================

def make_norm(hidden_dim, norm_type):
    if norm_type == "batch":
        return nn.BatchNorm1d(hidden_dim)
    if norm_type == "layer":
        return nn.LayerNorm(hidden_dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class SimpleMotifGNN(nn.Module):
    """
    Motif/region-level message passing over a batched motif graph.

    motif_x:          [num_total_motifs_or_regions, hidden_dim]
    motif_edge_index: [2, num_motif_or_region_edges]
    """

    def __init__(self, hidden_dim=128, dropout=0.2, norm_type="layer"):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            make_norm(hidden_dim, norm_type),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, motif_x, motif_edge_index):
        if motif_x.size(0) == 0:
            return motif_x
        if motif_edge_index.numel() == 0:
            return self.norm(motif_x)

        src, dst = motif_edge_index[0], motif_edge_index[1]
        msg = self.msg(motif_x[src])
        agg = scatter(msg, dst, dim=0, dim_size=motif_x.size(0), reduce="mean")
        out = self.update(torch.cat([motif_x, agg], dim=-1))
        out = self.norm(motif_x + out)
        return out


# ============================================================
# Basic MLP head
# ============================================================

class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        dropout=0.2,
        out_dim=1,
        norm_type="layer",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            make_norm(hidden_dim, norm_type),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualGate(nn.Module):
    """
    h_base + gate * delta([h_base, h_aux])
    Used for node-motif fusion inside each layer and for residual final fusion.
    """

    def __init__(self, hidden_dim=128, dropout=0.2, norm_type="layer"):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            make_norm(hidden_dim, norm_type),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_base, h_aux):
        h_in = torch.cat([h_base, h_aux], dim=-1)
        delta = self.delta(h_in)
        gate = self.gate(h_in)
        return h_base + gate * delta, gate


# ============================================================
# One layer: node feature + motif/region feature
# ============================================================

class NodeMotifLayerBlock(nn.Module):
    """
    For one GINE layer output h_l:
        1) node readout: h_l -> node graph feature
        2) motif/region readout: atom-to-region pooling -> region GNN -> region pooling
        3) residual gated fusion: node + gated motif/region correction
    """

    def __init__(self, hidden_dim=128, dropout=0.2, norm_type="layer"):
        super().__init__()
        self.node_pool = AttentionPool(hidden_dim)
        self.motif_gnn = SimpleMotifGNN(
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.motif_pool = AttentionPool(hidden_dim)
        self.node_motif_fuse = ResidualGate(
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, h, data, build_motif_batch_fn):
        h_node = self.node_pool(h, data.batch)
        motif_x, motif_batch, motif_edge_index = build_motif_batch_fn(data, h)
        motif_x = self.motif_gnn(motif_x, motif_edge_index)
        h_motif = self.motif_pool(motif_x, motif_batch)

        h_fused, motif_gate = self.node_motif_fuse(h_node, h_motif)

        return {
            "h": h_fused,
            "h_node": h_node,
            "h_motif": h_motif,
            "motif_gate": motif_gate,
        }


# ============================================================
# Main model
# ============================================================

class HierNodeMotifGNN(nn.Module):
    """
    Hierarchical node-motif GNN without KL / variational bottleneck.

    Final v3 uses all three layers, and every layer always combines atom-level
    graph readout with molecular-region readout.
    Forward returns final_logit plus auxiliary logits from h1/h2/h3.
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=128,
        dropout=0.2,
        num_tasks=1,
        task_type="classification",
    ):
        super().__init__()
        assert task_type in {"classification", "regression"}

        self.fusion_type = "concat"
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.norm_type = "batch" if task_type == "classification" else "layer"

        self.encoder = GINEHierEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.layer1_block = NodeMotifLayerBlock(
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=self.norm_type,
        )
        self.layer2_block = NodeMotifLayerBlock(
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=self.norm_type,
        )
        self.layer3_block = NodeMotifLayerBlock(
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=self.norm_type,
        )

        self.final_head = MLPHead(
            input_dim=hidden_dim * 3,
            hidden_dim=hidden_dim,
            dropout=dropout,
            out_dim=num_tasks,
            norm_type=self.norm_type,
        )

        # Auxiliary heads. They are used only by auxiliary loss.
        self.layer1_head = MLPHead(
            hidden_dim,
            hidden_dim,
            dropout,
            num_tasks,
            norm_type=self.norm_type,
        )
        self.layer2_head = MLPHead(
            hidden_dim,
            hidden_dim,
            dropout,
            num_tasks,
            norm_type=self.norm_type,
        )
        self.layer3_head = MLPHead(
            hidden_dim,
            hidden_dim,
            dropout,
            num_tasks,
            norm_type=self.norm_type,
        )

    def build_motif_batch(self, data, h):
        """
        Build batched motif/region representations from atom embeddings.

        Preferred fields for precomputed overlapping molecular regions:
            data.region_atom_index: [2, num_memberships], atom ids to region ids
            data.n_regions:         number of regions for each graph
            data.region_edge_index: batched region graph edge_index

        Fallback fields for old disjoint BRICS motifs:
            data.atom_to_motif:     local motif id for each atom
            data.n_motifs:          number of motifs for each graph
            data.motif_edge_index:  batched motif graph edge_index

        Region and motif edge indices are expected to be already batched/offset
        correctly by data_loader.MolHierData.__inc__.
        """
        device = h.device

        if hasattr(data, "region_atom_index") and hasattr(data, "n_regions"):
            n_regions_per_graph = data.n_regions.view(-1).to(device)
            total_regions = int(n_regions_per_graph.sum().item())
            region_atom_index = data.region_atom_index.to(device)

            if total_regions == 0:
                num_graphs = getattr(data, "num_graphs", None)
                if num_graphs is None:
                    num_graphs = int(data.batch.max().item()) + 1
                region_x = h.new_zeros((num_graphs, h.size(-1)))
                region_batch = torch.arange(num_graphs, device=device, dtype=torch.long)
                region_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
                return region_x, region_batch, region_edge_index

            atom_idx = region_atom_index[0].long()
            region_idx = region_atom_index[1].long()
            motif_x = scatter(
                h[atom_idx],
                region_idx,
                dim=0,
                dim_size=total_regions,
                reduce="mean",
            )

            motif_batch = torch.repeat_interleave(
                torch.arange(n_regions_per_graph.size(0), device=device, dtype=torch.long),
                n_regions_per_graph,
            )

            if hasattr(data, "region_edge_index"):
                motif_edge_index = data.region_edge_index.to(device)
            else:
                motif_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            return motif_x, motif_batch, motif_edge_index

        n_motifs_per_graph = data.n_motifs.view(-1).to(device)

        motif_offsets = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(n_motifs_per_graph, dim=0)[:-1],
            ],
            dim=0,
        )

        atom_to_motif_local = data.atom_to_motif.to(device)
        atom_graph_id = data.batch.to(device)
        atom_to_motif_global = atom_to_motif_local + motif_offsets[atom_graph_id]

        total_motifs = int(n_motifs_per_graph.sum().item())

        motif_x = scatter(
            h,
            atom_to_motif_global,
            dim=0,
            dim_size=total_motifs,
            reduce="mean",
        )

        motif_batch = torch.repeat_interleave(
            torch.arange(n_motifs_per_graph.size(0), device=device, dtype=torch.long),
            n_motifs_per_graph,
        )

        motif_edge_index = data.motif_edge_index.to(device)
        return motif_x, motif_batch, motif_edge_index

    def forward(self, data):
        h1_node, h2_node, h3_node = self.encoder(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )

        # ----------------------------------------------------
        # Each layer has node feature + optional motif feature.
        # ----------------------------------------------------
        out1 = self.layer1_block(
            h1_node,
            data,
            self.build_motif_batch,
        )
        out2 = self.layer2_block(
            h2_node,
            data,
            self.build_motif_batch,
        )
        out3 = self.layer3_block(
            h3_node,
            data,
            self.build_motif_batch,
        )

        h1 = out1["h"]
        h2 = out2["h"]
        h3 = out3["h"]

        # ----------------------------------------------------
        # Final prediction from concatenated layer features.
        # ----------------------------------------------------
        h_concat = torch.cat([h1, h2, h3], dim=-1)
        final_logit = self.final_head(h_concat)

        # Auxiliary predictions from each layer representation.
        layer1_logit = self.layer1_head(h1)
        layer2_logit = self.layer2_head(h2)
        layer3_logit = self.layer3_head(h3)

        return {
            "final_logit": final_logit,
            "layer1_logit": layer1_logit,
            "layer2_logit": layer2_logit,
            "layer3_logit": layer3_logit,
            # Backward-compatible aliases for old losses/scripts
            "local_logit": layer1_logit,
            "motif_logit": layer2_logit,
            "global_logit": layer3_logit,
            # Keep h_fused as a backward-compatible alias for old scripts.
            "h_fused": h_concat,
            "h_concat": h_concat,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "layer_info": {
                "layer1": out1,
                "layer2": out2,
                "layer3": out3,
                "use_node_layers": (True, True, True),
                "use_motif_layers": (True, True, True),
            },
            "fusion_info": {
                "fusion_type": "concat",
                "att_weight": None,
            },
        }


# ============================================================
# Optional helper for loading pretrained checkpoint
# ============================================================

def load_pretrained_encoder(model, ckpt_path, device="cpu", strict=False):
    """
    Load matching pretrained weights only.
    Descriptor heads or unmatched modules from pretraining are ignored automatically.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    pretrained_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model_state = model.state_dict()
    matched_state = {}

    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched_state[k] = v

    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=strict)
    print(f"Loaded pretrained parameters: {len(matched_state)} / {len(model_state)}")
    return model

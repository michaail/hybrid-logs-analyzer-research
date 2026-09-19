"""Attribute-Aware Graph Autoencoder (AttributeAwareGAE).

Architecture extracted from ``6_GAE_Training_BGL_fixed.ipynb`` and corrected
so structure reconstruction matches a *directed*, *per-graph* adjacency:

Encoder
    * ``raw_node_norm`` — BatchNorm1d on input node features.
    * ``node_proj`` + ``edge_proj`` — linear projections to ``hidden_dim``.
    * ``encoder_conv`` — GINEConv with configurable aggregation.

Decoder (multi-task)
    1. Structure — directed concat-MLP (default) or inner product. Trained
       with BCE on observed edges and *in-graph* non-edges (never cross-graph
       pairs from a PyG mini-batch).
    2. Node feature reconstruction — 2-layer MLP from latent Z onto the
       reconstruction columns (full ``x`` or TF-IDF+extras when
       ``node_reconstruct=without_sbert``).
    3. Edge attribute reconstruction — 2-layer MLP from ⟨Z_i ∥ Z_j⟩.

``raw_edge_norm`` is intentionally absent: BGL's ``log1p(td_std)`` is ~0 for
most edges and in-model BatchNorm would explode. Edges are pre-normalised
on the training split with std clamped ≥ 0.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import negative_sampling, scatter

FULL_STRUCTURE_MAX_NODES = 256
NODE_EXTRA_DIM = 9


def node_recon_index_tensor(
    node_dim: int,
    *,
    sbert_dim: int,
    extra_dim: int = NODE_EXTRA_DIM,
) -> torch.Tensor:
    """Column indices for the node decoder when skipping the SBERT block."""
    if sbert_dim <= 0:
        return torch.arange(node_dim, dtype=torch.long)
    embed_dim = node_dim - extra_dim
    if embed_dim < sbert_dim:
        raise ValueError(
            f"sbert_dim={sbert_dim} exceeds embedding width {embed_dim} "
            f"(node_dim={node_dim}, extra_dim={extra_dim})."
        )
    tfidf_dim = embed_dim - sbert_dim
    lexical = torch.arange(tfidf_dim, dtype=torch.long)
    extras = torch.arange(embed_dim, node_dim, dtype=torch.long)
    return torch.cat([lexical, extras])


class AttributeAwareGAE(nn.Module):
    """Multi-task Graph Autoencoder with a GINEConv encoder.

    Parameters
    ----------
    structure_decoder : str
        ``"mlp"`` (directed concat-MLP, default) or ``"inner_product"``
        (symmetric ⟨z_i, z_j⟩, Family B ablation).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        gine_aggregation: str = "sum",
        node_transformation: str = "mlp",
        structure_decoder: str = "mlp",
        node_recon_index: torch.Tensor | None = None,
        tfidf_dim: int = 0,
        sbert_dim: int = 0,
        fusion_mode: str = "concat",
        modality_projection_dim: int = 64,
        node_loss_mode: str = "global",
        node_block_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if structure_decoder not in {"mlp", "inner_product"}:
            raise ValueError(
                f"structure_decoder must be 'mlp' or 'inner_product', got {structure_decoder!r}"
            )
        self.structure_decoder_kind = structure_decoder
        self.latent_dim = latent_dim
        self.node_dim = int(node_dim)
        self.tfidf_dim = int(tfidf_dim)
        self.sbert_dim = int(sbert_dim)
        self.extra_dim = int(node_dim - self.tfidf_dim - self.sbert_dim)
        if self.extra_dim < 0:
            raise ValueError("tfidf_dim + sbert_dim cannot exceed node_dim.")
        if fusion_mode not in {"concat", "projected_gated"}:
            raise ValueError("fusion_mode must be 'concat' or 'projected_gated'.")
        if node_loss_mode not in {"global", "block_balanced"}:
            raise ValueError("node_loss_mode must be 'global' or 'block_balanced'.")
        self.fusion_mode = fusion_mode
        self.node_loss_mode = node_loss_mode
        self.node_block_weights = {
            "tfidf": 1.0,
            "sbert": 1.0,
            "extras": 1.0,
            **(node_block_weights or {}),
        }
        if node_recon_index is None:
            recon_index = torch.arange(node_dim, dtype=torch.long)
        else:
            recon_index = torch.as_tensor(node_recon_index, dtype=torch.long).reshape(-1)
        if recon_index.numel() == 0:
            raise ValueError("node_recon_index must contain at least one column.")
        # Not a state-dict buffer: old packages must keep the historical key set
        # when recon_dim == node_dim. The index is saved on the training checkpoint.
        self.node_recon_index = recon_index
        recon_dim = int(recon_index.numel())

        self.raw_node_norm = nn.BatchNorm1d(node_dim, affine=False)

        self.modality_projectors = nn.ModuleDict()
        self.modality_gate_logits = None
        node_projection_input = node_dim
        if fusion_mode == "projected_gated":
            blocks = self._input_block_ranges()
            if len(blocks) < 2:
                raise ValueError("projected_gated fusion requires at least two non-empty modalities.")
            for name, (start, stop) in blocks.items():
                self.modality_projectors[name] = nn.Sequential(
                    nn.Linear(stop - start, modality_projection_dim),
                    nn.LayerNorm(modality_projection_dim),
                    nn.ReLU(),
                )
            self.modality_gate_logits = nn.Parameter(torch.zeros(len(blocks)))
            node_projection_input = modality_projection_dim * len(blocks)

        self.node_proj = nn.Linear(node_projection_input, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        if node_transformation == "mlp":
            nn_module: nn.Module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            nn_module = nn.Linear(hidden_dim, latent_dim)

        self.encoder_conv = GINEConv(nn_module, edge_dim=hidden_dim, aggr=gine_aggregation)

        self.structure_decoder = None
        if structure_decoder == "mlp":
            self.structure_decoder = nn.Sequential(
                nn.Linear(latent_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, recon_dim),
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def node_reconstruction_target(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Select the decoder target columns from BatchNorm-scaled node features."""
        index = self.node_recon_index
        if index.device != x_norm.device:
            index = index.to(device=x_norm.device)
            self.node_recon_index = index
        return x_norm.index_select(1, index)

    def _input_block_ranges(self) -> dict[str, tuple[int, int]]:
        ranges: dict[str, tuple[int, int]] = {}
        cursor = 0
        if self.tfidf_dim:
            ranges["tfidf"] = (cursor, cursor + self.tfidf_dim)
            cursor += self.tfidf_dim
        if self.sbert_dim:
            ranges["sbert"] = (cursor, cursor + self.sbert_dim)
            cursor += self.sbert_dim
        if self.extra_dim:
            ranges["extras"] = (cursor, cursor + self.extra_dim)
        return ranges

    def project_node_inputs(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Legacy concat or gated, equally sized modality projections."""
        if self.fusion_mode == "concat":
            return x_norm
        assert self.modality_gate_logits is not None
        gates = torch.softmax(self.modality_gate_logits, dim=0)
        projected = []
        for gate, (name, (start, stop)) in zip(
            gates, self._input_block_ranges().items(), strict=True
        ):
            projected.append(gate * self.modality_projectors[name](x_norm[:, start:stop]))
        return torch.cat(projected, dim=1)

    def node_block_errors(
        self, x_rec: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-node MSE for each reconstructed modality block."""
        squared = F.mse_loss(x_rec, target, reduction="none")
        original = self.node_recon_index.to(squared.device)
        result: dict[str, torch.Tensor] = {}
        for name, (start, stop) in self._input_block_ranges().items():
            positions = torch.nonzero(
                (original >= start) & (original < stop), as_tuple=False
            ).reshape(-1)
            if positions.numel():
                result[name] = squared.index_select(1, positions).mean(dim=1)
        return result

    def node_reconstruction_loss(
        self, x_rec: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.node_loss_mode == "global":
            return F.mse_loss(x_rec, target)
        blocks = self.node_block_errors(x_rec, target)
        weighted = [
            float(self.node_block_weights[name]) * values.mean()
            for name, values in blocks.items()
            if float(self.node_block_weights[name]) > 0
        ]
        weight_sum = sum(
            float(self.node_block_weights[name])
            for name in blocks
            if float(self.node_block_weights[name]) > 0
        )
        if not weighted or weight_sum <= 0:
            raise ValueError("At least one reconstructed node block must have positive weight.")
        return torch.stack(weighted).sum() / weight_sum

    def node_anomaly_error(self, x_rec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-node counterpart of the configured training objective."""
        if self.node_loss_mode == "global":
            return F.mse_loss(x_rec, target, reduction="none").mean(dim=1)
        blocks = self.node_block_errors(x_rec, target)
        weighted = [
            float(self.node_block_weights[name]) * values
            for name, values in blocks.items()
            if float(self.node_block_weights[name]) > 0
        ]
        weight_sum = sum(
            float(self.node_block_weights[name])
            for name in blocks
            if float(self.node_block_weights[name]) > 0
        )
        return torch.stack(weighted).sum(dim=0) / weight_sum

    def standardize_inputs(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply node BN; pass edge features through unchanged (pre-normalised)."""
        x_norm = self.raw_node_norm(x)
        if edge_attr is not None and edge_attr.numel() > 0 and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        return x_norm, edge_attr

    def encode(
        self,
        x_norm: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        x_h = self.node_proj(self.project_node_inputs(x_norm))
        edge_h = self.edge_proj(edge_attr_norm) if edge_attr_norm is not None else None
        return self.encoder_conv(x_h, edge_index, edge_h)

    def decode_structure(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Directed (MLP) or symmetric (inner-product) logits for edge pairs."""
        src, dst = edge_index
        if self.structure_decoder is None:
            return (z[src] * z[dst]).sum(dim=1)
        return self.structure_decoder(torch.cat([z[src], z[dst]], dim=-1)).squeeze(-1)

    def decode_node_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.node_decoder(z)

    def decode_edge_attributes(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return self.edge_decoder(torch.cat([z[src], z[dst]], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return ``(z, x_norm, edge_attr_norm)`` for use in loss computation."""
        x_norm, edge_attr_norm = self.standardize_inputs(x, edge_attr)
        z = self.encode(x_norm, edge_index, edge_attr_norm)
        return z, x_norm, edge_attr_norm


def per_graph_negative_sampling(edge_index: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Sample one non-edge per observed edge, restricted to that graph's nodes.

    Mini-batch ``negative_sampling(..., num_nodes=batch.num_nodes)`` treats the
    disjoint union as one graph and yields trivial cross-graph negatives.
    """
    if edge_index.size(1) == 0 or ptr.numel() < 2:
        return edge_index.new_zeros((2, 0))
    src, dst = edge_index[0], edge_index[1]
    chunks: list[torch.Tensor] = []
    n_graphs = int(ptr.numel() - 1)
    for graph in range(n_graphs):
        lo = int(ptr[graph].item())
        hi = int(ptr[graph + 1].item())
        n_nodes = hi - lo
        mask = (src >= lo) & (src < hi)
        pos = edge_index[:, mask]
        if pos.size(1) == 0 or n_nodes <= 0:
            continue
        local = pos - lo
        neg_local = negative_sampling(
            local,
            num_nodes=n_nodes,
            num_neg_samples=pos.size(1),
        )
        if neg_local.numel() == 0:
            continue
        chunks.append(neg_local + lo)
    if not chunks:
        return edge_index.new_zeros((2, 0))
    return torch.cat(chunks, dim=1)


def complete_directed_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    """All directed pairs including self-loops (collapsed graphs may loop)."""
    src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
    return torch.stack([src, dst], dim=0)


def graph_ptr(batch, num_graphs: int, device: torch.device) -> torch.Tensor:
    """Node-offset pointer tensor, reconstructed from ``batch.batch`` if needed."""
    ptr = getattr(batch, "ptr", None)
    if ptr is not None:
        return ptr
    counts = torch.bincount(batch.batch, minlength=num_graphs)
    out = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    out[1:] = torch.cumsum(counts, dim=0)
    return out


def _structure_terms(
    model: AttributeAwareGAE,
    z: torch.Tensor,
    pos_index: torch.Tensor,
    neg_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean pos BCE, mean neg BCE) as logits; zeros when a set is empty."""
    device = z.device
    pos_loss = torch.tensor(0.0, device=device)
    neg_loss = torch.tensor(0.0, device=device)
    if pos_index.size(1) > 0:
        pos_logits = model.decode_structure(z, pos_index)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
    if neg_index is not None and neg_index.size(1) > 0:
        neg_logits = model.decode_structure(z, neg_index)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )
    return pos_loss, neg_loss


def _structure_error_vector(
    model: AttributeAwareGAE,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    *,
    include_non_edges: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-graph structure error: pos BCE (+ in-graph non-edge BCE when requested)."""
    device = z.device
    num_graphs = int(ptr.numel() - 1)
    scores = torch.zeros(num_graphs, device=device)
    src = edge_index[0]
    for graph in range(num_graphs):
        lo = int(ptr[graph].item())
        hi = int(ptr[graph + 1].item())
        n_nodes = hi - lo
        mask = (src >= lo) & (src < hi) if edge_index.size(1) else None
        pos = edge_index[:, mask] if mask is not None else edge_index.new_zeros((2, 0))
        neg = None
        if include_non_edges and n_nodes > 0:
            neg = _in_graph_non_edges(pos, lo, n_nodes, device, generator)
        pos_loss, neg_loss = _structure_terms(model, z, pos, neg)
        scores[graph] = pos_loss + (neg_loss if include_non_edges else torch.tensor(0.0, device=device))
    return scores


def _in_graph_non_edges(
    pos: torch.Tensor,
    lo: int,
    n_nodes: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Non-edges inside one graph: full digraph when small, else sampled."""
    del generator  # sampling uses PyG's RNG; seed the process for campaigns
    if n_nodes <= 0:
        return pos.new_zeros((2, 0))
    if n_nodes <= FULL_STRUCTURE_MAX_NODES:
        complete = complete_directed_index(n_nodes, device)
        adj = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)
        if pos.size(1):
            adj[pos[0] - lo, pos[1] - lo] = True
        keep = ~adj[complete[0], complete[1]]
        return complete[:, keep] + lo
    local = pos - lo if pos.size(1) else pos.new_zeros((2, 0))
    n_pos = max(int(pos.size(1)), 1)
    neg_local = negative_sampling(
        local,
        num_nodes=n_nodes,
        num_neg_samples=n_pos,
        force_undirected=False,
    )
    if neg_local.numel() == 0:
        return pos.new_zeros((2, 0))
    return neg_local + lo


def train_epoch(
    model: AttributeAwareGAE,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> tuple[float, float, float, float]:
    """Run one full training epoch with per-graph structure negatives."""
    model.train()
    total_loss = total_str = total_node = total_edge = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        z, x_norm, edge_attr_norm = model(batch.x, batch.edge_index, batch.edge_attr)
        num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        ptr = graph_ptr(batch, num_graphs, device)

        loss_str = torch.tensor(0.0, device=device)
        if batch.edge_index.size(1) > 0:
            pos_logits = model.decode_structure(z, batch.edge_index)
            neg_edge = per_graph_negative_sampling(batch.edge_index, ptr)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            if neg_edge.size(1) > 0:
                neg_logits = model.decode_structure(z, neg_edge)
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits)
                )
            else:
                neg_loss = torch.tensor(0.0, device=device)
            loss_str = pos_loss + neg_loss

        x_rec = model.decode_node_features(z)
        loss_node = model.node_reconstruction_loss(
            x_rec, model.node_reconstruction_target(x_norm)
        )

        loss_edge = torch.tensor(0.0, device=device)
        if edge_attr_norm is not None and edge_attr_norm.size(0) > 0:
            edge_rec = model.decode_edge_attributes(z, batch.edge_index)
            loss_edge = F.mse_loss(edge_rec, edge_attr_norm)

        loss = alpha * loss_str + beta * loss_node + gamma * loss_edge
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = batch.num_graphs
        total_loss += loss.item() * n
        total_str += loss_str.item() * n
        total_node += loss_node.item() * n
        total_edge += loss_edge.item() * n

    ng = len(loader.dataset)
    if ng == 0:
        return 0.0, 0.0, 0.0, 0.0
    return total_loss / ng, total_str / ng, total_node / ng, total_edge / ng


@torch.no_grad()
def compute_anomaly_scores(
    model: AttributeAwareGAE,
    loader,
    device: torch.device,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    return_components: bool = False,
    include_structure_non_edges: bool = True,
) -> tuple:
    """Compute per-graph anomaly scores (weighted reconstruction error).

    Structure error matches training: BCE on observed edges plus in-graph
    non-edges (full directed adjacency on small collapsed graphs). Combined
    scores stay ``α·str + β·node + γ·edge``.
    """
    model.eval()
    all_scores, all_labels = [], []
    all_structure, all_node, all_edge = [], [], []
    all_tfidf, all_sbert, all_extras = [], [], []
    graph_ids: list[str] = []

    for batch in loader:
        batch = batch.to(device)
        z, x_norm, edge_attr_norm = model(batch.x, batch.edge_index, batch.edge_attr)
        num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        ptr = graph_ptr(batch, num_graphs, device)

        g_str = _structure_error_vector(
            model,
            z,
            batch.edge_index,
            ptr,
            include_non_edges=include_structure_non_edges,
        )

        x_rec = model.decode_node_features(z)
        target = model.node_reconstruction_target(x_norm)
        node_errors = model.node_anomaly_error(x_rec, target)
        g_node = scatter(
            node_errors, batch.batch, dim=0, reduce="mean", dim_size=num_graphs
        )

        g_edge = torch.zeros(num_graphs, device=device)
        if edge_attr_norm is not None and edge_attr_norm.size(0) > 0:
            ea_errors = F.mse_loss(
                model.decode_edge_attributes(z, batch.edge_index),
                edge_attr_norm,
                reduction="none",
            ).mean(dim=1)
            edge_batch = batch.batch[batch.edge_index[0]]
            g_edge = scatter(
                ea_errors, edge_batch, dim=0, reduce="mean", dim_size=num_graphs
            )

        g_str = torch.nan_to_num(g_str, 0.0)
        g_edge = torch.nan_to_num(g_edge, 0.0)

        total = alpha * g_str + beta * g_node + gamma * g_edge
        all_scores.append(total.cpu())
        all_labels.append(batch.y.cpu())
        if return_components:
            all_structure.append(g_str.cpu())
            all_node.append(g_node.cpu())
            all_edge.append(g_edge.cpu())
            block_errors = model.node_block_errors(x_rec, target)
            for name, destination in (
                ("tfidf", all_tfidf),
                ("sbert", all_sbert),
                ("extras", all_extras),
            ):
                values = block_errors.get(name)
                if values is None:
                    destination.append(torch.zeros(num_graphs))
                else:
                    destination.append(
                        scatter(
                            values, batch.batch, dim=0, reduce="mean", dim_size=num_graphs
                        ).cpu()
                    )
            graph_ids.extend(_batch_graph_ids(batch, num_graphs))

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    if not return_components:
        return scores, labels
    zeros = scores * 0.0
    components = {
        "structure": torch.cat(all_structure).numpy() if all_structure else zeros,
        "node": torch.cat(all_node).numpy() if all_node else zeros,
        "edge": torch.cat(all_edge).numpy() if all_edge else zeros,
        "tfidf": torch.cat(all_tfidf).numpy() if all_tfidf else zeros,
        "sbert": torch.cat(all_sbert).numpy() if all_sbert else zeros,
        "extras": torch.cat(all_extras).numpy() if all_extras else zeros,
    }
    return scores, labels, components, graph_ids


def _batch_graph_ids(batch, num_graphs: int) -> list[str]:
    """Return per-graph identifiers from a PyG batch when available."""
    for attribute in ("block_id", "window_id", "sequence_id", "graph_id"):
        if hasattr(batch, attribute):
            raw = getattr(batch, attribute)
            if raw is None:
                continue
            if hasattr(raw, "tolist"):
                values = raw.tolist()
            elif isinstance(raw, (list, tuple)):
                values = list(raw)
            else:
                values = [raw]
            if len(values) == num_graphs:
                return [str(value) for value in values]
    return [str(index) for index in range(num_graphs)]

"""Attribute-Aware Graph Autoencoder (AttributeAwareGAE).

Architecture extracted from ``6_GAE_Training_BGL_fixed.ipynb``:

Encoder
    * ``raw_node_norm`` — BatchNorm1d on input node features (TF-IDF is
      high-dimensional and sparse; BN stabilises training without affecting
      semantics).
    * ``node_proj`` + ``edge_proj`` — linear projections to ``hidden_dim``.
    * ``encoder_conv`` — GINEConv with configurable aggregation and a node
      transformation MLP (or linear) in the GIN neighbourhood function.

Decoder (multi-task)
    1. Structure reconstruction — inner product ⟨Z_i, Z_j⟩, trained with
       BCE against positive edges and negative samples.
    2. Node feature reconstruction — 2-layer MLP from latent Z to input dim.
    3. Edge attribute reconstruction — 2-layer MLP from ⟨Z_i ∥ Z_j⟩ to
       edge-feature dim.

Design notes
------------
``raw_edge_norm`` is intentionally absent (BGL FIX #2): BGL's
``log1p(td_std)`` edge feature is ~0 for 99% of edges; in-model BatchNorm
would amplify the rare non-zero values and cause gradient explosion.  Edge
features are instead pre-normalised once before training using global
mean/std computed over the training split, with std clamped ≥ 0.1.

Ablation toggles (passed to __init__)
    ``gine_aggregation``    — ``"sum"`` | ``"mean"`` | ``"max"``
    ``node_transformation`` — ``"mlp"`` (2-layer BN+ReLU) | ``"linear"``

Training helpers
----------------
    train_epoch(model, loader, optimizer, device, *, alpha, beta, gamma)
    compute_anomaly_scores(model, loader, device, *, alpha, beta, gamma, return_components)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import negative_sampling, scatter


class AttributeAwareGAE(nn.Module):
    """Multi-task Graph Autoencoder with a GINEConv encoder.

    Parameters
    ----------
    node_dim : int
        Input node feature dimension.
    edge_dim : int
        Input (pre-normalised) edge feature dimension.
    hidden_dim : int
        Intermediate projection dimension.
    latent_dim : int
        Latent embedding dimension output by the encoder.
    gine_aggregation : str
        Neighbourhood aggregation for GINEConv: ``"sum"``, ``"mean"``, or
        ``"max"``.
    node_transformation : str
        Node MLP inside GINEConv: ``"mlp"`` (two linear layers separated by
        BatchNorm1d + ReLU) or ``"linear"`` (single linear layer, no
        activation).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        gine_aggregation: str = "sum",
        node_transformation: str = "mlp",
    ) -> None:
        super().__init__()

        # ── Input standardisation ─────────────────────────────────────────────
        self.raw_node_norm = nn.BatchNorm1d(node_dim, affine=False)
        # raw_edge_norm intentionally absent — BGL FIX #2 (pre-normalised upstream)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        if node_transformation == "mlp":
            nn_module: nn.Module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:  # "linear"
            nn_module = nn.Linear(hidden_dim, latent_dim)

        self.encoder_conv = GINEConv(nn_module, edge_dim=hidden_dim, aggr=gine_aggregation)

        # ── Decoders ──────────────────────────────────────────────────────────
        # 1. Structure: inner-product (no extra parameters)

        # 2. Node feature reconstruction
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # 3. Edge attribute reconstruction
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    # ── Forward passes ────────────────────────────────────────────────────────

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
        x_h = self.node_proj(x_norm)
        edge_h = self.edge_proj(edge_attr_norm) if edge_attr_norm is not None else None
        return self.encoder_conv(x_h, edge_index, edge_h)

    def decode_structure(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Inner-product decoder; returns raw BCE logits."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def decode_node_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.node_decoder(z)

    def decode_edge_attributes(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
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


# ── Training helpers ──────────────────────────────────────────────────────────


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
    """Run one full training epoch.

    Parameters
    ----------
    alpha, beta, gamma : float
        Loss weights for structure, node-feature, and edge-attribute
        reconstruction respectively.

    Returns
    -------
    tuple[float, float, float, float]
        Per-graph mean (total, structure, node, edge) losses.
    """
    model.train()
    total_loss = total_str = total_node = total_edge = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        z, x_norm, edge_attr_norm = model(batch.x, batch.edge_index, batch.edge_attr)

        # 1. Structure loss (guard against empty-edge batches)
        loss_str = torch.tensor(0.0, device=device)
        if batch.edge_index.size(1) > 0:
            pos_logits = model.decode_structure(z, batch.edge_index)
            neg_edge = negative_sampling(
                batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1),
            )
            neg_logits = model.decode_structure(z, neg_edge)
            loss_str = (
                F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            )

        # 2. Node feature loss
        x_rec = model.decode_node_features(z)
        loss_node = F.mse_loss(x_rec, x_norm)

        # 3. Edge attribute loss
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
) -> tuple:
    """Compute per-graph anomaly scores (weighted reconstruction error).

    Parameters
    ----------
    alpha, beta, gamma : float
        Same loss weights used during training.
    return_components : bool
        When True, also return unweighted structure/node/edge errors and
        graph identifiers. Combined scores stay ``α·str + β·node + γ·edge``.

    Returns
    -------
    scores : numpy.ndarray, shape (N,)
    labels : numpy.ndarray, shape (N,)
    When *return_components* is True, also:
    components : dict[str, numpy.ndarray]
    graph_ids : list[str]
    """

    model.eval()
    all_scores, all_labels = [], []
    all_structure, all_node, all_edge = [], [], []
    graph_ids: list[str] = []

    for batch in loader:
        batch = batch.to(device)
        z, x_norm, edge_attr_norm = model(batch.x, batch.edge_index, batch.edge_attr)
        num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else 1

        # Structure error (per graph)
        g_str = torch.zeros(num_graphs, device=device)
        if batch.edge_index.size(1) > 0:
            pos_probs = torch.sigmoid(model.decode_structure(z, batch.edge_index))
            edge_err = F.binary_cross_entropy(
                pos_probs, torch.ones_like(pos_probs), reduction="none"
            )
            edge_batch = batch.batch[batch.edge_index[0]]
            g_str = scatter(edge_err, edge_batch, dim=0, reduce="mean", dim_size=num_graphs)

        # Node error (per graph)
        x_rec = model.decode_node_features(z)
        node_errors = F.mse_loss(x_rec, x_norm, reduction="none").mean(dim=1)
        g_node = scatter(
            node_errors, batch.batch, dim=0, reduce="mean", dim_size=num_graphs
        )

        # Edge error (per graph)
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

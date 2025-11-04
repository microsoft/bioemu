# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#  Code taken from the MFMP-Protein/distributional graphormer codebase, with some modifications.

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .chemgraph import ChemGraph


class FeedForward(nn.Module):
    """Standard single hidden layer MLP with dropout and GELU activations."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class DiffHead(nn.Module):
    """
    MLP network that takes the invariant sequence representation as input and outputs translation
    and inverse rotation vectors which are used to model the score.
    """

    def __init__(self, ninp: int):
        super().__init__()
        self.fc_t = nn.Sequential(
            nn.LayerNorm(ninp),
            nn.Linear(ninp, ninp),
            nn.ReLU(),
            nn.Linear(ninp, 3),
        )
        self.fc_eps = nn.Sequential(
            nn.LayerNorm(ninp),
            nn.Linear(ninp, ninp),
            nn.ReLU(),
            nn.Linear(ninp, 3),  # Dimension three for axis-angle representation.
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T_eps = self.fc_t(x)
        IR_eps = self.fc_eps(x)
        return (T_eps, IR_eps)


class SAAttention(nn.Module):
    """DiG version of the invariant point attention module. See AF2 supplement Alg 22.
    I believe SA might stand for "Structural Attention", see App B.3 in the DiG paper.

    The forward pass of this module is identical to IPA as described in Alg 22 in AF2 supplement,
    with the following changes:
        1. An extra linear map is applied to the pair representation.
        2. Dropout is applied to the output. (In AF2 it is applied outside of IPA. This may be
            equivalent.)


    Args:
        d_model: Dimension of attention dot product * number of heads.
        d_pair: Dimension of the pair representation.
        n_head: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_pair: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("The hidden size is not a multiple of the number of attention heads.")
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.scalar_query = nn.Linear(d_model, d_model, bias=False)
        self.scalar_key = nn.Linear(d_model, d_model, bias=False)
        self.scalar_value = nn.Linear(d_model, d_model, bias=False)
        self.pair_bias = nn.Linear(d_pair, n_head, bias=False)
        self.point_query = nn.Linear(
            d_model, n_head * 3 * 4, bias=False
        )  # 4 is N_query_points in Alg 22.
        self.point_key = nn.Linear(
            d_model, n_head * 3 * 4, bias=False
        )  # 4 is N_query_points in Alg 22.
        self.point_value = nn.Linear(
            d_model, n_head * 3 * 8, bias=False
        )  # 8 is N_point_values in Alg 22.

        self.scalar_weight = 1.0 / math.sqrt(3 * self.d_k)  # Alg 22 line 7, w_L / sqrt(d_k).
        self.point_weight = 1.0 / math.sqrt(3 * 4 * 9 / 2)  # Alg 22 line 7, w_C * w_L.
        self.trained_point_weight = nn.Parameter(
            torch.rand(n_head)
        )  # gamma^h, AF2 Supp Section 1.8.2.
        self.pair_weight = 1.0 / math.sqrt(3)  # Alg 22 line 7, w_L.

        self.pair_value = nn.Linear(
            d_pair, d_model, bias=False
        )  # NOTE: AF2 IPA does not have this.

        self.fc_out = nn.Linear(d_model * 2 + n_head * 8 * 4, d_model, bias=True)  # Alg 22 line 11.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x1d: torch.Tensor,
        x2d: torch.Tensor,
        pose: tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
    ) -> torch.Tensor:

        """Forward pass of the SAAttention module.

        Args:
            x1d: Invariant sequence representation.
            x2d: Invariant pair representation.
            pose: Tuple of translation and inverse rotation vectors.
            bias: Pair bias, used to encode masking.
        """
        T, R = pose[0], pose[1].transpose(
            -1, -2
        )  # Transpose to go back to rotations from inverse rotations.

        # Compute scalar attention queries keys and values.
        # Alg 22 line 1, shape [B, L, nhead, C].
        q_scalar = self.scalar_query(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)
        k_scalar = self.scalar_key(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)
        v_scalar = self.scalar_value(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)

        # Perform scalar dot product attention.
        # Alg 22 line 7, shape [B, nhead, L, L]
        scalar_attn = torch.einsum("bihc,bjhc->bhij", q_scalar * self.scalar_weight, k_scalar)

        # Compute point attention queries keys and values.
        # Alg 22 line 2-3, shape [B, L, nhead, num_points, 3]
        q_point_local = self.point_query(x1d).reshape(*x1d.shape[:-1], self.n_head, -1, 3)
        k_point_local = self.point_key(x1d).reshape(*x1d.shape[:-1], self.n_head, -1, 3)
        v_point_local = self.point_value(x1d).reshape(*x1d.shape[:-1], self.n_head, -1, 3)

        def apply_affine(point: torch.Tensor, T: torch.Tensor, R: torch.Tensor):
            """Apply affine transformation (T, R) to point x. Acts as x -> R @ x + T. This follows
            AF2 Supplement Section 1.1.

            Args:
                point: Point to transform.
                T: Translation vector.
                R: Rotation matrix.

            Returns:
                Transformed point.
            """
            return (
                torch.matmul(R[:, :, None, None], point.unsqueeze(-1)).squeeze(-1)
                + T[:, :, None, None]
            )

        # Apply the frames to the attention points.
        # Alg 22 lines 7 and 10, shape [B, L, nhead, num_points, 3]
        q_point_global = apply_affine(q_point_local, T, R)
        k_point_global = apply_affine(k_point_local, T, R)
        v_point_global = apply_affine(v_point_local, T, R)

        # Compute squared distances between transformed points.
        # Alg 22 line 7, shape [B, L, L, nhead, num]
        point_attn = torch.norm(q_point_global.unsqueeze(2) - k_point_global.unsqueeze(1), dim=-1)
        point_weight = self.point_weight * F.softplus(
            self.trained_point_weight
        )  # w_L * w_C * gamma^h
        point_attn = (
            -0.5 * point_weight[:, None, None] * torch.sum(point_attn, dim=-1).permute(0, 3, 1, 2)
        )

        # Alg 22 line 4.
        pair_attn = self.pair_weight * self.pair_bias(x2d).permute(0, 3, 1, 2)

        # Compute attention logits, Alg 22 line 7.
        attn_logits = scalar_attn + point_attn + pair_attn + bias  # [B, nhead, L, L]

        # Compute attention weights.
        # Alg 22 line 7, shape [B, nhead, L, L]
        attn = torch.softmax(attn_logits, dim=-1)

        # Alg 22 line 9.
        out_scalar = torch.einsum("bhij,bjhc->bihc", attn, v_scalar)
        out_scalar = out_scalar.reshape(*out_scalar.shape[:2], -1)

        # Alg 22 line 10.
        with torch.amp.autocast("cuda", enabled=False):
            out_point_global = torch.einsum(
                "bhij,bjhcp->bihcp", attn.float(), v_point_global.float()
            )
        # Inverse affine transformation, as per Alg 22 line 10, and AF2 Supplement Section 1.1.
        out_point_local = torch.matmul(
            R.transpose(-1, -2)[:, :, None, None],
            (out_point_global - T[:, :, None, None]).unsqueeze(-1),
        ).squeeze(-1)

        # Alg 22 line 11.
        out_point_norm = torch.norm(out_point_local, dim=-1)
        out_point_norm = out_point_norm.reshape(*out_point_norm.shape[:2], -1)
        out_point_local = out_point_local.reshape(*out_point_local.shape[:2], -1)

        # NOTE: AF2 IPA does not project x2d as in here, i.e., v_pair = x2d in AF2.
        v_pair = self.pair_value(x2d).reshape(*x2d.shape[:-1], self.n_head, -1)

        # Alg 22 line 8.
        out_pair = torch.einsum("bhij,bijhc->bihc", attn, v_pair)
        out_pair = out_pair.reshape(*out_pair.shape[:2], -1)

        # Alg 22 line 11.
        out_feat = torch.cat([out_scalar, out_point_local, out_pair, out_point_norm], dim=-1)

        # NOTE: AF2 includes dropout outside IPA, not inside. See AF2 Alg 22 line 6.
        x = self.dropout(self.fc_out(out_feat))
        return x  # [B, L, C]


class SAEncoderLayer(nn.Module):
    """IPA interleaved with layernorm and MLP."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        extra_residue_embeds: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SAAttention(d_model=d_model, d_pair=d_pair, n_head=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)
        self.extra_residue_embeds = extra_residue_embeds
        if self.extra_residue_embeds:
            # Define embeddings for 1D and 2D features. 1D is for residue types, we assume the number of residue types is not more than 30.
            # 2D is for pairwise combinations of residue types, so we assume n_residue_types^2 combinations, use 1000 for safety.
            self.x1d_mul = nn.Embedding(30, d_model)
            self.x1d_add = nn.Embedding(30, d_model)
            self.x2d_mul = nn.Embedding(1000, d_pair)
            self.x2d_add = nn.Embedding(1000, d_pair)

            # alpha_1d and alpha_2d control the contribution of the extra residue embeddings.
            # When zero, the extra embeddings have no effect.
            self.alpha_1d = nn.Parameter(torch.tensor(0.0))
            self.alpha_2d = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x1d: torch.Tensor,
        x2d: torch.Tensor,
        pose: tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor,
        node_labels: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # Add additional residue embeddings if specified
        if self.extra_residue_embeds:
            assert node_labels is not None
            node_emb_mul_batch = self.x1d_mul(node_labels)
            node_emb_mul, _ = to_dense_batch(
                node_emb_mul_batch, batch_index
            )  # [B, L, 384], embedding of the node features.
            node_emb_add_batch = self.x1d_add(node_labels)
            node_emb_add, _ = to_dense_batch(node_emb_add_batch, batch_index)

            # Create edge_labels from pair-wise node labels
            assert edge_index is not None
            edge_labels = node_labels[edge_index[0]] * 30 + node_labels[edge_index[1]]

            edge_emb_mul_batch = self.x2d_mul(edge_labels)
            edge_emb_mul = to_dense_adj(  # [B, L, L, 128]
                edge_index, batch_index, edge_attr=edge_emb_mul_batch
            ).to(torch.float32)
            edge_emb_add_batch = self.x2d_add(edge_labels)
            edge_emb_add = to_dense_adj(edge_index, batch_index, edge_attr=edge_emb_add_batch).to(
                torch.float32
            )

            x1d = x1d + self.alpha_1d * (x1d * node_emb_mul + node_emb_add)
            x2d = x2d + self.alpha_2d * (x2d * edge_emb_mul + edge_emb_add)

        x1d = x1d + self.attn(self.norm1(x1d), x2d, pose, bias)
        x1d = x1d + self.ffn(self.norm2(x1d))
        return x1d


class SAEncoder(nn.Module):
    """Stack of IPA layers interleaved with layernorm and MLPs."""

    def __init__(self, n_layer: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([SAEncoderLayer(**kwargs) for _ in range(n_layer)])

    def forward(
        self,
        x1d: torch.Tensor,
        x2d: torch.Tensor,
        pose: tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
        context: ChemGraph,
    ) -> torch.Tensor:
        batch_index = context.batch
        edge_index = context.edge_index
        node_labels = context.node_labels  # node_labels is used for extra residue embeddings.

        for module in self.layers:
            x1d = module(
                x1d,
                x2d,
                pose,
                bias,
                batch_index=batch_index,
                edge_index=edge_index,
                node_labels=node_labels,
            )
        return x1d


class StructureModule(nn.Module):
    """Network that predicts translation and rotation score from input translations and rotations."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.encoder = SAEncoder(d_model=d_model, **kwargs)
        self.diff_head = DiffHead(ninp=d_model)

    def forward(
        self,
        pose: tuple[torch.Tensor, torch.Tensor],
        x1d: torch.Tensor,
        x2d: torch.Tensor,
        bias: torch.Tensor,
        context: ChemGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1d = self.encoder(x1d, x2d, pose, bias, context=context)
        return self.diff_head(x1d)

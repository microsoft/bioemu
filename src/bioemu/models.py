# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .chemgraph import ChemGraph
from .structure_module import StructureModule

# Number of dimensions for the nodes and edges of Evoformer embeddings
EVOFORMER_NODE_DIM: int = 384
EVOFORMER_EDGE_DIM: int = 128


class SinusoidalPositionEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        min_input: float = 0.0,
        max_input: float = 1000.0,  # 1000 is the DiG default maximum diffusion time value.
    ) -> None:
        """
        Sinusoidal embedding for encoding a scalar input.
        See e.g.
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Args:
            dim: Dimension of embedding.
            max_period: Maximum period of embedding.
            min_input: The minimum expected range of the scalar input variable.
            max_input: The maximum expected range of the scalar input variable.
        """
        super().__init__()
        self.dim = dim
        self.half_dim = self.dim // 2
        self.min_input = min_input
        self.max_input = max_input

        self.embedding_factor = -math.log(max_period) / (self.half_dim - 1)

        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Convert diffusion times to the corresponding embeddings.

        Args:
            time: Tensor of times.

        Returns:
            Time embeddings.
        """
        # Rescale the input scalar so that it is in the range [0, 1000.]. This matches the
        # behaviour of the timestep embedder in DiG.
        time = (time - self.min_input) * 1000.0 / (self.max_input - self.min_input)

        device = time.device
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * self.embedding_factor)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int = 64, max_distance: int = 256, out_dim: int = 2) -> None:
        """
        Module for computing embeddings based on the relative position between residues in the
        sequence. Distances are categorized into different buckets and each is assigned a learnable
        embedding.

        NOTE: The algorithm here for computing the embeddings is distinct from both AF2 and FrameDiff. It is taken from DiG.
        The resolution of the buckets gets lower the
        further you go from the diagonal.

        Args:
            num_buckets: Number of buckets used for binning pair distances.
            max_distance: Maximum distance considered for embedding.
            out_dim: Dimension of output embeddings.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, out_dim)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor, num_buckets: int, max_distance: int
    ) -> torch.Tensor:
        """
        Obtain embedding index based on current relative positions.

        Args:
            relative_position: Relative positions in sequence.
            num_buckets: Number of buckets used for binning.
            max_distance: Maximum distance considered for embedding.

        Returns:
            Index corresponding to the embedding vector of each pairwise distance.
        """
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Get relative position embeddings.

        Args:
            relative_position: Tensor of relative sequence positions.

        Returns:
            Embedding vectors corresponding to relative positions.
        """
        rp_bucket = self._relative_position_bucket(  # [L, L] integers
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(
            rp_bucket
        )  # [L, L, C] embeddings of each relative sequence distance
        return rp_bias


class DistributionalGraphormer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        dim_pair: int = 256,
        num_layers: int = 8,
        num_heads: int = 32,
        dim_single_rep: int = 64,
        dim_hidden: int = 1024,
        num_buckets: int = 64,
        max_distance_relative: int = 128,
        dropout: float = 0.1,
    ):
        """
        Basic distributional graphormer model. For architecture details, please refer to:
        https://arxiv.org/abs/2306.05445

        Args:
            dim_model: Number of features used for main model representation.
            dim_pair: Number of features used for pair representations.
            num_layers: Number of layers used in the structure model.
            num_heads: Number of heads used for attention.
            dim_single_rep: Number of features used for single sequence embeddings.
            dim_pair_rep: Number of features used for pair embeddings.
            dim_hidden: Number of nodes used in hidden layers.
            num_buckets: Number of buckets used in relative positional encoding.
            max_distance_relative: Maximum distance considered in relative positional encoding.
            dropout: _description_. Defaults to 0.1.

            Let (T_in, R_in) be the input frames, and let (T, R) be an arbitrary rotation and translation.
            Let T_out, R_out = model(T_in, R_in). And let T_out_transformed, R_out_transformed = model((T, R) * (T_in, R_in)), where frames are
            composed according to Section 1.1 of the AF2 Supplementary Material.

            The model has the following equivariance properties:
                T_out_transformed = R * T_out
                R_out_transformed = R_out
            This makes it suitable for regressing the "invariant score", as in the DiG paper.
        """

        super().__init__()

        self.d_model = dim_model

        # Set the dimensions for EVOFORMER feature embedding
        dim_single_rep = EVOFORMER_NODE_DIM
        self.dim_pair_rep = EVOFORMER_EDGE_DIM

        self.step_emb = SinusoidalPositionEmbedder(dim=self.d_model)
        self.x1d_proj = nn.Sequential(
            nn.LayerNorm(dim_single_rep),
            nn.Linear(dim_single_rep, self.d_model, bias=False),
        )
        self.x2d_proj = nn.Sequential(
            nn.LayerNorm(self.dim_pair_rep),
            nn.Linear(self.dim_pair_rep, dim_pair, bias=False),
        )
        self.rp_proj = RelativePositionBias(
            num_buckets=num_buckets, max_distance=max_distance_relative, out_dim=dim_pair
        )

        self.st_module = StructureModule(
            d_pair=dim_pair,
            n_layer=num_layers,
            d_model=self.d_model,
            n_head=num_heads,
            dim_feedforward=dim_hidden,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        node_orientations: torch.Tensor,
        batch_index: torch.Tensor,
        t: torch.Tensor,
        context: ChemGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict translation and rotation scores based on noisy positions and node orientations.
        This function takes and returns quantities in sparse batch format.

        Args:
            x: [N, 3] node positions.
            node_orientations: Rotation matrices encoding orientations [N, 3, 3].
            batch_index: Sparse batch indices.
            t: Current diffusion time steps.
            context: Current batch in ChemGraph format. Used for embeddings, labels and masking.

        Returns:
            Predicted translation and rotation scores, both in vector representation. The
            translation score will be equivariant and the rotation score invariant to global pose
            rotations.
        """
        batch_index = context.batch
        # Convert from sparse graph representation to dense representation used by DiG.
        T_perturbed, mask = to_dense_batch(
            x, batch_index
        )  # [B, L, 3], [B, L], L is maximum number of residues in a batch
        IR_perturbed, _ = to_dense_batch(node_orientations, batch_index)  # [B, L, 3, 3]

        # Transform node and edge embeddings from sparse to dense.
        single_repr, _ = to_dense_batch(context.single_embeds, batch_index)
        single_repr = single_repr.to(torch.float32)  # [B, L, 384]
        pair_repr = to_dense_adj(  # [B, L, L, 128]
            context.edge_index, batch_index, edge_attr=context.pair_embeds
        ).to(torch.float32)

        # Deal with masking that comes from the data itself, not just the dense batch representation.
        # There are two sources of masking - the first is the mask that comes from the data itself,
        # which represents residues where the position/orientation are unknown. This is captured in
        # `pos_is_known`. The second is masking due to uneven length proteins in the dense batch
        # representation. This is captured in `mask``.
        # attn_mask is True if the element should be masked out in attention, following DiG conventions.
        if "pos_is_known" in context:
            pos_is_known = context.pos_is_known
            pos_is_known_dense = to_dense_batch(pos_is_known, batch_index)[0].bool()  # [B, L]
            attn_mask = ~(mask & pos_is_known_dense)  # [B, L]
        else:
            attn_mask = ~mask  # [B, L]

        t, _ = to_dense_batch(t, batch_index)  # [B, L]
        t = t[:, 0]  # [B], each residue experiences the same timestep.

        # Embed single_repr, noise level "t" and the conditionings from context into x1d.
        x1d = self.x1d_proj(single_repr) + self.step_emb(t)[:, None]  # [B, L, C]

        x2d = self.x2d_proj(pair_repr)  # [B, L, L, C]

        pos_sequence = torch.arange(
            T_perturbed.shape[1], device=x1d.device
        )  # [L], integer sequence index

        pos_sequence = pos_sequence.unsqueeze(1) - pos_sequence.unsqueeze(
            0
        )  # [L, L], integer relative sequence positions.
        x2d = x2d + self.rp_proj(pos_sequence)[None]

        z = (~attn_mask).long().sum(-1, keepdims=True)  # [B, 1], number of unmasked elements.
        filled_mask = attn_mask.masked_fill(
            z == 0, False  # z == 0 means no unmasked elements, i.e., all elements are masked.
        )  # [B, L], fill in False if all elements are masked. This is to avoid the case where all elements are masked, which would cause an error in the softmax I think.

        bias = filled_mask.float().masked_fill(filled_mask, float("-inf"))[:, None, :, None]
        bias = bias.permute(
            0, 3, 1, 2
        )  # [B, 1, 1, L], this has the value -inf for elements that are masked.

        # Change in translation and rotation. At this point, both quantities are invariant to global
        # SE(3) transformations.
        (
            T_eps,
            IR_eps,
        ) = self.st_module(  # st_module plays an equivalent role to BackboneUpdate in the Algorithm 20 of AF2 supplement.
            (T_perturbed, IR_perturbed), x1d, x2d, bias
        )

        # Introduce orientation dependence of the translation score.
        T_eps = torch.matmul(IR_perturbed.transpose(-1, -2), T_eps.unsqueeze(-1)).squeeze(-1)
        T_out, R_out = T_eps, IR_eps

        # Return back to sparse graph representation
        T_out = T_out[mask]  # [N, 3]
        R_out = R_out[mask]  # [N, 3], in axis angle representation.

        return (
            T_out,
            R_out,
        )

    def __str__(self) -> str:
        """
        Model prints with number of trainable parameters.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params}"


class DiGConditionalScoreModel(torch.nn.Module):
    """Wrapper to convert the DiG nn.Module neural network that operates directly on position
    and rotation tensors into a ScoreModel that operates on ChemGraph objects.
    """

    def __init__(
        self,
        dim_model: int = 512,
        dim_pair: int = 256,
        num_layers: int = 8,
        num_heads: int = 32,
        dim_single_rep: int = 64,
        dim_hidden: int = 1024,
        num_buckets: int = 64,
        max_distance_relative: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args: all passed through to DistributionalGraphormer
        """
        super().__init__()
        self.model_nn = DistributionalGraphormer(
            dim_model=dim_model,
            dim_pair=dim_pair,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_single_rep=dim_single_rep,
            dim_hidden=dim_hidden,
            num_buckets=num_buckets,
            max_distance_relative=max_distance_relative,
            dropout=dropout,
        )

    def forward(self, x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        # NOTE: the DiG structure model uses a time embedding intended for integer time
        # steps between 0 and num_timesteps (1000 by default). The SDE gets times between
        # 0 and 1, where the filter used in the embedding is less expressive. To avoid
        # this, t is scaled by the timesteps before passing to model.
        assert hasattr(x, "batch"), "batch of ChemGraphs must have a 'batch' attribute."
        time_effective = t[x.batch] * 1000
        # NOTE: DiG takes in inverse rotations as input. To be consistent
        # with frame conventions and sampling in the rest of the code, frames are transposed
        # / inverted here.
        node_orientations_effective = x.node_orientations.swapaxes(-1, -2)

        context = x.replace(pos=None, node_orientations=None)

        # pos is the translation score, node_orientations is the rotation score in axis
        # angle representation.
        pos_effective = x.pos
        (pos, node_orientations) = self.model_nn(
            x=pos_effective,
            node_orientations=node_orientations_effective,
            batch_index=x.batch,
            t=time_effective,
            context=context,
        )

        return x.replace(pos=pos, node_orientations=node_orientations)

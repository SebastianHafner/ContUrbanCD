import torch
import torch.nn as nn
from torch import Tensor
import einops

import numpy as np

# https://pgmpy.org/models/markovnetwork.html
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

from typing import Tuple, Sequence, Callable

from joblib import Parallel, delayed

import os
os.environ["PYTHONWARNINGS"] = "ignore"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TFRModule(nn.Module):
    def __init__(self, t: int, d_model: int, n_heads: int, d_hid: int, activation: str, n_layers: int):
        super().__init__()
        # Generate relative temporal encodings
        self.register_buffer('temporal_encodings', self.get_relative_encodings(t, d_model), persistent=False)

        # Define a transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_hid, batch_first=True,
            activation=activation
        )
        # Create module
        self.temporal_feature_refinement = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, features: Tensor) -> Tensor:
        B, T, D, H, W = features.size()

        # Reshape to tokens
        tokens = einops.rearrange(features, 'B T D H W -> (B H W) T D')

        # Adding relative temporal encodings
        tokens = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)

        # Feature refinement with self-attention
        features_hat = self.temporal_feature_refinement(tokens)

        # Reshape to original shape
        features_hat = einops.rearrange(features_hat, '(B H W) T D -> B T D H W', B=B, H=H)

        return features_hat

    @staticmethod
    def get_relative_encodings(sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result


class CFModule(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(features: Sequence[Tensor], edges: Sequence[Tuple[int, int]]) -> Sequence[Tensor]:
        # compute urban change detection features
        features_ch = []
        for feature in features:
            B, T, _, H, W = feature.size()
            feature_ch = []
            for t1, t2 in edges:
                feature_ch.append(feature[:, t2] - feature[:, t1])
            # n: number of combinations
            feature_ch = torch.stack(feature_ch)
            features_ch.append(feature_ch)
        return features_ch


class MTIModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, o_ch: Tensor, o_seg: Tensor, edges: Sequence[Tuple[int, int]]) -> Tensor:

        B, T, _, H, W = o_seg.size()

        # Get processing function by defining the Markov network
        process_pixel = self.markov_network(T, edges)

        # Reshape
        o_ch = einops.rearrange(o_ch, 'B N C H W -> (B H W) N C').cpu().numpy()
        o_seg = einops.rearrange(o_seg, 'B T C H W -> (B H W) T C').cpu().numpy()

        # Find optimal building time series using Markov network
        o_seg_mrf = Parallel(n_jobs=-1)(delayed(process_pixel)(p_seg, p_ch) for p_seg, p_ch, in zip(o_seg, o_ch))

        # Reshape
        o_seg_mrf = torch.Tensor(o_seg_mrf)
        o_seg_mrf = einops.rearrange(o_seg_mrf, '(B H W) (T C) -> B T C H W', H=H, W=W, T=T)

        return o_seg_mrf

    @staticmethod
    def markov_network(n: int, edges: Sequence[Tuple[int, int]]) -> Callable:
        # Define a function to process a single pixel
        def process_pixel(y_hat_seg: Sequence[float], y_hat_ch: Sequence[float]):
            model = MarkovNetwork()

            for t in range(len(y_hat_seg)):
                model.add_node(f'N{t}')
                # Cardinality: number of potential values (i.e., 2: 0/1)
                # Potential Values for node: P(urban=True), P(urban=False)
                urban_value = float(y_hat_seg[t])
                factor = DiscreteFactor([f'N{t}'], cardinality=[2], values=[1 - urban_value, urban_value])
                model.add_factors(factor)

            # add adjacent edges w/o potentials
            for t in range(n - 1):
                model.add_edge(f'N{t}', f'N{t + 1}')

            # add edges with potentials
            for i, (t1, t2) in enumerate(edges):
                model.add_edge(f'N{t1}', f'N{t2}')
                # [P(A=False, B=False), P(A=False, B=True), P(A=True, B=False), P(A=True, B=True)]
                change_value = float(y_hat_ch[i])
                edge_values = [1 - change_value, change_value, change_value, 1 - change_value]

                factor = DiscreteFactor([f'N{t1}', f'N{t2}'], cardinality=[2, 2], values=edge_values)
                model.add_factors(factor)

            # Create an instance of BeliefPropagation algorithm
            bp = BeliefPropagation(model)

            # Compute the most probable state of the MRF
            state = bp.map_query()
            states_list = [state[f'N{t}'] for t in range(n)]
            return states_list

        return process_pixel

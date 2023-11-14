import math

import numpy as np
import pandas as pd

import torch
from torch import nn

# http://d2l.ai/chapter_recurrent-modern/seq2seq.html
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value
    return X


# http://d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html
def masked_softmax(X, valid_lens=None):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# http://d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.attention_weights, values)


# https://www.kaggle.com/bminixhofer/a-validation-framework-impact-of-the-random-seed
class SelfAttention(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self.feature_dim = feature_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x, valid_lens=None):
        feature_dim = self.feature_dim
        step_dim = x.shape[1]

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim
        )

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if valid_lens is not None:
            a = sequence_mask(a.clone(), valid_lens, value=0)

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1), a


class ADModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
        bidirectional,
        self_attention=True,
        average_pool=False,
        last_step=False,
        return_att_weight=False,
    ):
        super().__init__()
        self.num_classes = num_classes

        ratio = 2 if bidirectional else 1
        embed_size = hidden_size * ratio

        self.linear_input = nn.LazyLinear(hidden_size * 2)
        self.bn_input = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.rnn = nn.GRU(
            hidden_size * 2,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.bn_rnn = nn.LazyBatchNorm1d()

        if self_attention:
            self.attention = SelfAttention(embed_size)
        self.self_attention = self_attention

        self.average_pool = average_pool
        self.last_step = last_step

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LazyLinear(num_classes),
        )

        self.return_att_weight = return_att_weight

    def forward(self, X_static, X_dynamic, valid_lens=None):
        if valid_lens is not None:
            x_seq = torch.concat(
                (
                    X_static.unsqueeze(1).repeat_interleave(
                        int(valid_lens.max()), axis=1
                    ),
                    X_dynamic[:, : int(valid_lens.max()), :],
                ),
                axis=2,
            )
        else:
            x_seq = torch.concat(
                (
                    X_static.unsqueeze(1).repeat_interleave(
                        X_dynamic.shape[1], axis=1
                    ),
                    X_dynamic,
                ),
                axis=2,
            )

        # (batch_size, seq_len, embed_size)
        x_seq = self.linear_input(x_seq)
        # Permute to fit batchnorm: (batch_size, embed_size, seq_len)
        x_seq = x_seq.permute(0, 2, 1)
        x_seq = self.bn_input(x_seq)
        x_seq = x_seq.permute(0, 2, 1)
        x_seq = self.dropout(self.relu(x_seq))

        # (seq_len, batch_size, embed_size)
        x_seq = x_seq.permute(1, 0, 2)
        self.rnn.flatten_parameters()
        # (seq_len, batch_size, embed_size)
        rnn_output, _ = self.rnn(x_seq)
        # Permute to fit batchnorm: (batch_size, embed_size, seq_len)
        rnn_output = rnn_output.permute(1, 2, 0)
        rnn_output = self.bn_rnn(rnn_output)
        rnn_output = rnn_output.permute(0, 2, 1)

        if self.self_attention:
            # (batch_size, embed_size)
            att_output, att_weight = self.attention(rnn_output, valid_lens=valid_lens)
            att_output = att_output.squeeze(1)
        if self.average_pool:
            att_output = rnn_output.mean(1).squeeze(1)
        if self.last_step:
            att_output = rnn_output[:, -1, :].squeeze(1)

        output = self.mlp(att_output)

        # output: (batch_size, ) or (batch_size, num_classes(>1))
        if self.num_classes == 1:
            output = output.squeeze(-1)

        if self.return_att_weight:
            return output, att_weight
        return output


class BaselineModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
    ):
        super().__init__()
        self.num_classes = num_classes

        module_list = []
        for _ in range(num_layers):
            module_list.extend(
                [
                    nn.LazyLinear(hidden_size),
                    nn.LazyBatchNorm1d(),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        module_list.append(nn.LazyLinear(num_classes))

        self.mlp = nn.Sequential(*module_list)

    def forward(self, X):
        output = self.mlp(X)
        # (batch_size, ) or (batch_size, num_classes(>1))
        return output.squeeze(-1) if self.num_classes == 1 else output

import torch
from torch import nn as nn

from cerml.encoder_decoder_networks import Encoder


class NoActionEncoder(Encoder):
    def __init__(self,
                 state_dim,
                 action_dim,
                 reward_dim,
                 net_complex_enc_dec,
                 encoder_type,
                 latent_dim,
                 batch_size,
                 num_classes,
                 time_steps=None,
                 merge_mode=None
                 ):
        super().__init__(state_dim, 0, reward_dim, net_complex_enc_dec, encoder_type, latent_dim, batch_size,
                         num_classes, time_steps, merge_mode)
        self.original_input_dim = state_dim + action_dim + reward_dim + state_dim
        self.modified_input_dim = state_dim + reward_dim + state_dim
        # exclude the action indices, keep the rest
        self.relevant_input_indices = list(range(state_dim)) + \
                                      list(range(state_dim + action_dim, self.original_input_dim))

    def exclude_actions(self, x):
        if x.shape[-1] == self.original_input_dim:
            return x[..., self.relevant_input_indices]
        elif x.shape[-1] == self.modified_input_dim:
            return x
        else:
            raise ValueError(f'Unexpected input dimension {x.shape[-1]}')

    def forward(self, x):
        return super().forward(self.exclude_actions(x))

    def encode(self, x):
        return super().encode(self.exclude_actions(x))


class SpecialOmissionEncoder(Encoder):
    def __init__(self,
                 state_dim,
                 action_dim,
                 reward_dim,
                 net_complex_enc_dec,
                 encoder_type,
                 latent_dim,
                 batch_size,
                 num_classes,
                 time_steps=None,
                 merge_mode=None
                 ):
        super().__init__(1, 0, reward_dim, net_complex_enc_dec, encoder_type, latent_dim, batch_size,
                         num_classes, time_steps, merge_mode)
        self.original_input_dim = state_dim + action_dim + reward_dim + state_dim
        self.modified_input_dim = 3
        # keep only reward and position
        self.relevant_input_indices = [8, 26, 35]

    def exclude_indices(self, x):
        if x.shape[-1] == self.original_input_dim:
            return x[..., self.relevant_input_indices]
        elif x.shape[-1] == self.modified_input_dim:
            return x
        else:
            raise ValueError(f'Unexpected input dimension {x.shape[-1]}')

    def forward(self, x):
        return super().forward(self.exclude_indices(x))

    def encode(self, x):
        return super().encode(self.exclude_indices(x))


class NoOpEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.latent_dim = 1

    def forward(self, x):
        if len(x.shape) == 2:
            return torch.zeros((1, self.latent_dim), device=x.device), \
                   torch.zeros(1, device=x.device)
        else:
            return torch.zeros((x.shape[0], self.latent_dim), device=x.device), \
                   torch.zeros(x.shape[0], device=x.device)

    def encode(self, x):
        raise NotImplementedError('encode should not be called on NoOpEncoder')

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random"):
        raise NotImplementedError('sample_z should not be called on NoOpEncoder')

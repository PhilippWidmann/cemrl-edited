import torch
from torch import nn as nn

from cerml.encoder_decoder_networks import Encoder, DecoderMDP


class NoActionEncoder(Encoder):
    def __init__(self,
                 state_dim,
                 action_dim,
                 reward_dim,
                 net_complex_enc_dec,
                 encoder_type,
                 encoder_exclude_padding,
                 latent_dim,
                 batch_size,
                 num_classes,
                 time_steps=None,
                 merge_mode=None
                 ):
        super().__init__(state_dim, 0, reward_dim, net_complex_enc_dec, encoder_type, encoder_exclude_padding,
                         latent_dim, batch_size, num_classes, time_steps, merge_mode)
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

    def forward(self, x, padding_mask=None, return_distributions=False, exclude_padding=False):
        return super().forward(self.exclude_actions(x), padding_mask=padding_mask,
                               return_distributions=return_distributions, exclude_padding=exclude_padding)

    def encode(self, x, padding_mask=None, exclude_padding=False):
        return super().encode(self.exclude_actions(x), padding_mask=padding_mask, exclude_padding=exclude_padding)


class SpecialOmissionEncoder(Encoder):
    def __init__(self,
                 state_dim,
                 action_dim,
                 reward_dim,
                 net_complex_enc_dec,
                 encoder_type,
                 encoder_exclude_padding,
                 latent_dim,
                 batch_size,
                 num_classes,
                 time_steps=None,
                 merge_mode=None,
                 relevant_input_indices=None
                 ):

        if relevant_input_indices is not None:
            self.relevant_input_indices = relevant_input_indices
        else:
            self.relevant_input_indices = [8, 26, 35]
        super().__init__(0, len(self.relevant_input_indices), 0, net_complex_enc_dec, encoder_type,
                         encoder_exclude_padding, latent_dim, batch_size, num_classes, time_steps, merge_mode)
        self.original_input_dim = state_dim + action_dim + reward_dim + state_dim
        self.modified_input_dim = len(relevant_input_indices)

    def exclude_indices(self, x):
        if x.shape[-1] == self.original_input_dim:
            return x[..., self.relevant_input_indices]
        elif x.shape[-1] == self.modified_input_dim:
            return x
        else:
            raise ValueError(f'Unexpected input dimension {x.shape[-1]}')

    def forward(self, x, padding_mask=None, return_distributions=False, exclude_padding=False):
        return super().forward(self.exclude_indices(x), padding_mask=padding_mask,
                               return_distributions=return_distributions, exclude_padding=exclude_padding)

    def encode(self, x, padding_mask=None, exclude_padding=False):
        return super().encode(self.exclude_indices(x), padding_mask=padding_mask, exclude_padding=exclude_padding)


class NoOpEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.latent_dim = 1

    def forward(self, x, return_distributions=False):
        if len(x.shape) == 2:
            first_dim = 1
        else:
            first_dim = x.shape[0]
        if return_distributions:
            return torch.zeros((first_dim, self.latent_dim), device=x.device), \
                   torch.zeros(first_dim, device=x.device), \
                   None
        else:
            return torch.zeros((first_dim, self.latent_dim), device=x.device), \
                   torch.zeros(first_dim, device=x.device)


    def encode(self, x):
        raise NotImplementedError('encode should not be called on NoOpEncoder')

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random"):
        raise NotImplementedError('sample_z should not be called on NoOpEncoder')


class SpecialOmissionDecoder(DecoderMDP):
    def __init__(self,
                 action_dim,
                 state_dim,
                 reward_dim,
                 z_dim,
                 net_complex,
                 state_reconstruction_clip,
                 use_state_decoder):
        super().__init__(action_dim, 1, reward_dim, z_dim, net_complex, state_reconstruction_clip, use_state_decoder)
        # keep only reward and position
        self.relevant_state_indices = [8]

    def forward(self, state, action, next_state, z):
        return super().forward(state[..., self.relevant_state_indices],
                               torch.zeros(action.shape, device=action.device),
                               None,  # next_state is unused
                               z)

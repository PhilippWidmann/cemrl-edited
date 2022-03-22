import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from cerml.utils import process_gaussian_parameters
from cerml.shared_encoder_variants import SharedEncoderTimestepMLP, SharedEncoderTrajectoryMLP, SharedEncoderGRU, \
    SharedEncoderConv, SharedEncoderFCN, SharedEncoderMeanTimestepMLP


class ClassEncoder(nn.Module):
    def __init__(self, shared_dim, num_classes):
        super(ClassEncoder, self).__init__()

        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(self.shared_dim, self.num_classes)

    def forward(self, m):
        return F.softmax(self.linear(m), dim=-1)


class PriorPz(nn.Module):
    def __init__(self,
                 num_classes,
                 latent_dim
                 ):
        super(PriorPz, self).__init__()
        self.latent_dim = latent_dim
        # feed cluster number y as one-hot, get mu_sigma out
        self.linear = nn.Linear(num_classes, self.latent_dim * 2)

    def forward(self, m):
        return self.linear(m)


class Encoder(nn.Module):
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
                 **kwargs
                 ):
        super(Encoder, self).__init__()

        self.encoder_type = encoder_type
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.merge_mode = merge_mode
        self.always_exclude_padding = encoder_exclude_padding

        if encoder_type == 'TimestepMLP':
            self.shared_encoder = SharedEncoderTimestepMLP(state_dim, action_dim, reward_dim, net_complex_enc_dec)
        elif encoder_type == 'MeanTimestepMLP':
            self.shared_encoder = SharedEncoderMeanTimestepMLP(state_dim, action_dim, reward_dim, net_complex_enc_dec)
        elif encoder_type == 'TrajectoryMLP':
            self.shared_encoder = SharedEncoderTrajectoryMLP(state_dim, action_dim, reward_dim, net_complex_enc_dec,
                                                             time_steps)
        elif encoder_type == 'GRU':
            self.shared_encoder = SharedEncoderGRU(state_dim, action_dim, reward_dim, net_complex_enc_dec)
        elif encoder_type == 'Conv':
            self.shared_encoder = SharedEncoderConv(state_dim, action_dim, reward_dim, net_complex_enc_dec)
        elif encoder_type == 'FCN':
            self.shared_encoder = SharedEncoderFCN(state_dim, action_dim, reward_dim, net_complex_enc_dec)
        else:
            raise ValueError(f'Unknown encoder type "{encoder_type}"')

        self.class_encoder = ClassEncoder(self.shared_encoder.shared_dim, num_classes)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_encoder.shared_dim, self.latent_dim * 2)
                                                 for _ in range(self.num_classes)])

        if self.merge_mode == 'linear':
            self.pre_class_encoder = nn.Linear(self.time_steps * self.shared_encoder.shared_dim,
                                               self.shared_encoder.shared_dim)
        elif self.merge_mode == 'mlp':
            self.pre_class_encoder = Mlp(input_size=self.time_steps * self.shared_encoder.shared_dim,
                                         hidden_sizes=[self.shared_encoder.shared_dim],
                                         output_size=self.shared_encoder.shared_dim)

    def forward(self, x, padding_mask=None, return_distributions=False, exclude_padding=False):
        """
        Encode the provided context
        :param x: context of the form (batch_size, time_steps, obs + action + reward + next_obs)
        :param return_distributions: If true, also return the distribution objects, not just a sampled data point
        :return: z - task indicator [batch_size, latent_dim]
                 y - base task indicator [batch_size]
        """
        y_distribution, z_distributions = self.encode(x, padding_mask=padding_mask, exclude_padding=exclude_padding)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        z, y = self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")
        if return_distributions:
            distribution = {
                'y_probs': y_distribution.probs.detach().cpu().numpy().squeeze(),
                'z_means': [z.loc.detach().cpu().numpy().squeeze() for z in z_distributions],
                'z_stds': [z.scale.detach().cpu().numpy().squeeze() for z in z_distributions]
            }
            return z, y, distribution
        else:
            return z, y

    def encode(self, x, padding_mask=None, exclude_padding=False):
        z_combination_mode = 'multiplication' if self.shared_encoder.returns_timestep_encodings else None
        if self.always_exclude_padding or exclude_padding:
            if padding_mask is None:
                raise ValueError('Padding should be excluded, but no padding_mask is specified.')

            y = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float, device=ptu.device)
            final_mu_sigma = [torch.zeros((x.shape[0], 2*self.latent_dim), dtype=torch.float, device=ptu.device)
                              for _ in range(self.num_classes)]

            padding_lengths = padding_mask.sum(axis=1)
            unique_padding_lengths = np.unique(padding_lengths)

            # Process samples separately, excluding the padding from the input
            # For performance, we can combine all samples with the same number of available timesteps into a batch
            for length in unique_padding_lengths:
                relevant_samples = padding_lengths == length
                relevant_ind_per_sample = ~padding_mask[relevant_samples][0]
                # If we truly have no data (in the first step of the rollout), we give one of the padding-0's as input
                if not np.any(relevant_ind_per_sample):
                    relevant_ind_per_sample[-1] = True
                x_length = x[relevant_samples][:, relevant_ind_per_sample]
                y_temp, all_mu_sigma_temp = self._encode_partial(x_length)
                final_mu_sigma_temp = [process_gaussian_parameters(mu_sigma, self.latent_dim, mode=z_combination_mode)
                                       for mu_sigma in all_mu_sigma_temp]
                y[relevant_samples] = y_temp
                for i in range(self.num_classes):
                    final_mu_sigma[i][relevant_samples] = final_mu_sigma_temp[i]
        else:
            # Just encode everything at once; treats padding as valid data points
            y, all_mu_sigma = self._encode_partial(x)
            final_mu_sigma = [process_gaussian_parameters(mu_sigma, self.latent_dim, mode=z_combination_mode)
                              for mu_sigma in all_mu_sigma]

        # Construct the categorical and Normal distributions
        y_distribution = torch.distributions.categorical.Categorical(probs=y)
        z_distributions = [torch.distributions.normal.Normal(*torch.split(final_mu_sigma[i], split_size_or_sections=self.latent_dim, dim=-1))
                           for i in range(self.num_classes)]
        return y_distribution, z_distributions

    def _encode_partial(self, x):
        # Compute shared encoder forward pass
        m = self.shared_encoder(x)

        # Compute class probabilities
        # If the shared encoder produces separate encodings per timestep, we have to merge
        if self.shared_encoder.returns_timestep_encodings:
            y = self.merge_y(m)
        else:
            y = self.class_encoder(m)

        # Compute every gauss_encoder forward pass
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))

        return y, all_mu_sigma

    def merge_y(self, shared_encoding):
        if self.merge_mode == 'linear' or self.merge_mode == 'mlp':
            flat = torch.flatten(shared_encoding, start_dim=1)
            pre_class = self.pre_class_encoder(flat)
            y = self.class_encoder(pre_class)
        # Variant 2: Add logits
        elif self.merge_mode == "add":
            y = self.class_encoder(shared_encoding)
            y = y.sum(dim=-2) / y.shape[1]  # add the outcome of individual samples, scale down
        elif self.merge_mode == "add_softmax":
            y = self.class_encoder(shared_encoding)
            y = F.softmax(y.sum(dim=-2), dim=-1)  # add the outcome of individual samples, softmax
        # Variant 2: Multiply logits
        elif self.merge_mode == "multiply":
            y = self.class_encoder(shared_encoding)
            y = F.softmax(y.prod(dim=-2), dim=-1)  # multiply the outcome of individual samples
        elif self.merge_mode is None:
            raise ValueError('The shared encoder returns timestep encodings, but no merge mode is specified.')
        else:
            raise ValueError(f'The specified merge mode "{self.merge_mode}" is unknown.')

        return y

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random"):
        """
        Sample from the latent Gaussian mixture model

        :param y_distribution: Categorical distribution of the classes
        :param z_distributions: List of Gaussian distributions
        :param y_usage: 'most_likely' to sample from the most likely class per batch
                'specific' to sample from the class specified in param y for all batches
        :param y: class to sample from if y_usage=='specific'
        :param sampler: 'random' for actual sampling, 'mean' to return the mean of the Gaussian
        :return: z - task indicator [batch_size, latent_dim]
                 y - base task indicator [batch_size]
        """
        # Select from which Gaussian to sample
        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(self.batch_size, dtype=torch.long) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.latent_dim)

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].rsample(), 0) for i in range(self.num_classes)], dim=0)

        elif sampler == "mean":
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].mean, 0) for i in range(self.num_classes)], dim=0)

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        z = torch.squeeze(torch.gather(permute, 1, mask), 1)
        return z, y


class DecoderMDP(nn.Module):
    '''
    Uses data (state, action, reward, task_hypothesis z) from the replay buffer or online
    and computes estimates for the next state and reward.
    Through that it reconstructs the MDP and gives gradients back to the task hypothesis.
    '''
    def __init__(self,
                 action_dim,
                 state_dim,
                 reward_dim,
                 z_dim,
                 net_complex,
                 state_reconstruction_clip,
                 use_state_decoder,
                 use_next_state_for_reward=False):
        super(DecoderMDP, self).__init__()

        self.state_decoder_input_size = state_dim + action_dim + z_dim
        self.state_decoder_hidden_size = int(self.state_decoder_input_size * net_complex)
        
        self.use_next_state_for_reward = use_next_state_for_reward
        self.reward_decoder_input_size = state_dim + action_dim + z_dim
        if self.use_next_state_for_reward:
            self.reward_decoder_input_size = self.reward_decoder_input_size + state_dim
        self.reward_decoder_hidden_size = int(self.reward_decoder_input_size * net_complex)
        self.state_reconstruction_clip = state_reconstruction_clip
        self.use_state_decoder = use_state_decoder

        if use_state_decoder:
            self.net_state_decoder = Mlp(
                hidden_sizes=[self.state_decoder_hidden_size, self.state_decoder_hidden_size],
                input_size=self.state_decoder_input_size,
                output_size=self.state_reconstruction_clip
            )
        else:
            self.net_state_decoder = None

        self.net_reward_decoder = Mlp(
            hidden_sizes=[self.reward_decoder_hidden_size, self.reward_decoder_hidden_size],
            input_size=self.reward_decoder_input_size,
            output_size=reward_dim
        )

    def forward(self, state, action, next_state, z, padding_mask=None):
        if self.use_state_decoder:
            state_estimate = self.net_state_decoder(torch.cat([state, action, z], dim=-1))
        else:
            state_estimate = None
            
        if self.use_next_state_for_reward:
            reward_estimate = self.net_reward_decoder(torch.cat([state, action, next_state, z], dim=-1))
        else:
            reward_estimate = self.net_reward_decoder(torch.cat([state, action, z], dim=-1))

        # Ignore all the padding in the input (if present) and set the corresponding estimates to 0
        # Cannot ignore them sooner since the input size has to be the same for all batch samples
        if padding_mask is not None:
            reward_estimate[padding_mask] = 0
            if state_estimate is not None:
                state_estimate[padding_mask] = 0

        return state_estimate, reward_estimate

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger


class OptimizationEncoder(nn.Module):
    def __init__(
            self,
            decoder,
            state_dim,
            action_dim,
            reward_dim,
            latent_dim,
            batch_size,
            num_classes,
            time_steps,
            lr_encoder,
            reconstruct_all_steps,
            state_reconstruction_clip,
            first_order,
            z_update_steps,
            optimizer_class=optim.Adam,
            **kwargs
    ):
        super().__init__()

        self.decoder = decoder
        self.reward_dim = reward_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.first_order = first_order
        self.z_update_steps = z_update_steps
        self.lr_encoder = lr_encoder
        self.reconstruct_all_steps = reconstruct_all_steps
        self.state_reconstruction_clip = state_reconstruction_clip
        self.optimizer_class = optimizer_class

        if self.num_classes > 1:
            raise NotImplementedError('OptimizationEncoder does not support Gaussian mixtures yet')
        if not self.reconstruct_all_steps:
            raise ValueError('The OptimizationEncoder is only sensible when reconstructing all timesteps.')

    def forward(self, x, return_distributions=False):
        """
        Encode the provided context
        :param x: context of the form (batch_size, time_steps, obs + action + reward + next_obs)
        :param return_distributions: If true, also return the distribution objects, not just a sampled data point
        :return: z - task indicator [batch_size, latent_dim]
                 y - base task indicator [batch_size]
        """

        if return_distributions:
            warnings.warn('OptimizationEncoder is not probabilistic yet. '
                                      'forward(..., return_distribution=True) is not supported.')
            return *self.encode(x), None
        return self.encode(x)

    def encode(self, x):
        state = x[:, :, :self.state_dim]
        action = x[:, :, self.state_dim:(self.state_dim + self.action_dim)]
        reward = x[:, :, (self.state_dim + self.action_dim):(self.state_dim + self.action_dim + self.reward_dim)]
        next_state = x[:, :, (self.state_dim + self.action_dim + self.reward_dim):
                             (self.state_dim + self.action_dim + self.reward_dim + self.state_dim)]

        z = torch.zeros((x.shape[0], self.latent_dim),
                        device=x.device,
                        requires_grad=True)

        #self._set_decoder_requires_grad(False)
        for i in range(self.z_update_steps):
            z_rep = z.unsqueeze(dim=1).repeat([1, self.time_steps, 1])
            next_state_pred, reward_pred = self.decoder(state, action, next_state, z_rep)
            loss = F.mse_loss(reward_pred, reward, reduction='sum')
            if next_state_pred is not None:
                loss += F.mse_loss(next_state_pred, next_state[..., :self.state_reconstruction_clip], reduction='sum')

            task_gradients = torch.autograd.grad(loss, z, create_graph=not self.first_order)[0]
            z = z - self.lr_encoder * task_gradients
        #self._set_decoder_requires_grad(True)

        #z.requires_grad = False
        return z, torch.zeros(self.batch_size, device=x.device)

    def _set_decoder_requires_grad(self, requires_grad):
        for param in self.decoder.parameters():
            param.requires_grad = requires_grad

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
        raise NotImplementedError('OptimizationEncoder is not probabilistic yet. sample_z should not be called.')


class OptimizationReconstructionTrainer():
    def __init__(self,
                 encoder,
                 decoder,
                 prior_pz_layer,
                 replay_buffer,
                 batch_size,
                 validation_batch_size,
                 num_classes,
                 latent_dim,
                 timesteps,
                 lr_decoder,
                 lr_encoder,
                 alpha_kl_z,
                 beta_kl_y,
                 use_state_diff,
                 component_constraint_learning,
                 state_reconstruction_clip,
                 use_state_decoder,
                 use_data_normalization,
                 train_val_percent,
                 eval_interval,
                 early_stopping_threshold,
                 experiment_log_dir,
                 temp_dir,
                 prior_mode,
                 prior_sigma,
                 data_usage_reconstruction,
                 reconstruct_all_steps,
                 optimizer_class=optim.Adam,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior_pz_layer = prior_pz_layer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.lr_decoder = lr_decoder
        self.lr_encoder = lr_encoder
        self.alpha_kl_z = alpha_kl_z
        self.beta_kl_y = beta_kl_y
        self.use_state_diff = use_state_diff
        self.component_constraint_learning = component_constraint_learning
        self.state_reconstruction_clip = state_reconstruction_clip
        self.use_state_decoder = use_state_decoder
        self.use_data_normalization = use_data_normalization
        self.train_val_percent = train_val_percent
        self.eval_interval = eval_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.experiment_log_dir = experiment_log_dir
        self.temp_dir = temp_dir
        self.prior_mode = prior_mode
        self.prior_sigma = prior_sigma
        self.data_usage_reconstruction = data_usage_reconstruction
        self.reconstruct_all_steps = reconstruct_all_steps

        self.factor_state_loss = 1
        self.factor_reward_loss = self.state_reconstruction_clip

        self.loss_weight_state = self.factor_state_loss / (self.factor_state_loss + self.factor_reward_loss)
        self.loss_weight_reward = self.factor_reward_loss / (self.factor_state_loss + self.factor_reward_loss)

        self.lowest_loss = np.inf
        self.lowest_loss_epoch = 0

        self.temp_path = os.path.join(self.temp_dir, self.experiment_log_dir.split('/')[-1])
        self.encoder_path = os.path.join(self.temp_path, 'encoder.pth')
        self.decoder_path = os.path.join(self.temp_path, 'decoder.pth')
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.optimizer_class = optimizer_class

        self.loss_state_decoder = nn.MSELoss()
        self.loss_reward_decoder = nn.MSELoss()

        self.optimizer_decoder = self.optimizer_class(
            self.decoder.parameters(),
            lr=self.lr_decoder,
        )

    def train(self, epochs):
        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        if self.data_usage_reconstruction == "tree_sampling":
            train_indices = np.random.permutation(train_indices)
            val_indices = np.random.permutation(val_indices)

        train_overall_losses = []
        train_state_losses = []
        train_reward_losses = []

        train_val_state_losses = []
        train_val_reward_losses = []

        val_state_losses = []
        val_reward_losses = []

        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        epoch = 0
        for epoch in range(epochs):
            overall_loss, state_loss, reward_loss = self.training_step(train_indices, epoch)
            train_overall_losses.append(overall_loss)
            train_state_losses.append(state_loss)
            train_reward_losses.append(reward_loss)

            #Evaluate with validation set for early stopping
            if epoch % self.eval_interval == 0:
                val_state_loss, val_reward_loss = self.validate(val_indices)
                val_state_losses.append(val_state_loss)
                val_reward_losses.append(val_reward_loss)
                train_val_state_loss, train_val_reward_loss = self.validate(train_indices)
                train_val_state_losses.append(train_val_state_loss)
                train_val_reward_losses.append(train_val_reward_loss)

                # change loss weighting
                weight_factors = np.ones(2)
                weights = np.array([train_val_state_loss * self.factor_state_loss, train_val_reward_loss * self.factor_reward_loss])
                for i in range(weights.shape[0]):
                    weight_factors[i] = weights[i] / np.sum(weights)
                self.loss_weight_state = weight_factors[0]
                self.loss_weight_reward = weight_factors[1]
                if int(os.environ['DEBUG']) == 1:
                    print("weight factors: " + str(weight_factors))

                # Debug printing
                if int(os.environ['DEBUG']) == 1:
                    print("\nEpoch: " + str(epoch))
                    print("Overall loss: " + str(train_overall_losses[-1]))
                    print("Train Validation loss (state, reward): " + str(train_val_state_losses[-1]) + ' , ' + str(train_val_reward_losses[-1]))
                    print("Validation loss (state, reward): " + str(val_state_losses[-1]) + ' , ' + str(val_reward_losses[-1]))
                if self.early_stopping(epoch, val_state_loss + val_reward_loss):
                    print("Early stopping at epoch " + str(epoch))
                    break

        # load the least loss encoder
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.decoder.load_state_dict(torch.load(self.decoder_path))

        # for logging
        validation_train = self.validate(train_indices)
        validation_val = self.validate(val_indices)

        logger.record_tabular("Reconstruction_train_val_state_loss", validation_train[0])
        logger.record_tabular("Reconstruction_train_val_reward_loss", validation_train[1])
        logger.record_tabular("Reconstruction_val_state_loss", validation_val[0])
        logger.record_tabular("Reconstruction_val_reward_loss", validation_val[1])
        logger.record_tabular("Reconstruction_epochs", epoch + 1)

    def training_step(self, indices, step):
        '''
        Computes a forward pass to encoder and decoder with sampling at the encoder.
        The overall objective due to the generative model is:
        parameter* = arg max ELBO
        ELBO = sum_k q(y=k | x) * [ log p(x|z_k) - KL ( q(z, x,y=k) || p(z|y=k) ) ] - KL ( q(y|x) || p(y) )
        '''

        # get data from replay buffer
        # TODO: for validation data use all data --> batch size == validation size
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size,
                                                               normalize=self.use_data_normalization)

        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        # prepare for usage in decoder
        decoder_action = ptu.from_numpy(data['actions'])
        decoder_state = ptu.from_numpy(data['observations'])
        decoder_next_state = ptu.from_numpy(data['next_observations'])
        decoder_reward = ptu.from_numpy(data['rewards'])
        true_task = data['true_tasks']

        if not self.reconstruct_all_steps:
            # Reconstruct only the current timestep
            decoder_action = decoder_action[:, -1, :]
            decoder_state = decoder_state[:, -1, :]
            decoder_next_state = decoder_next_state[:, -1, :]
            decoder_reward = decoder_reward[:, -1, :]
            true_task = true_task[:, -1, :]

        if self.use_state_diff:
            decoder_state_target = (decoder_next_state - decoder_state)[..., :self.state_reconstruction_clip]
        else:
            decoder_state_target = decoder_next_state[..., :self.state_reconstruction_clip]

        # Local optimization of z
        z, _ = self.encoder(encoder_input)
        if self.reconstruct_all_steps:
            z = z.unsqueeze(1).repeat(1, self.timesteps + 1, 1)

        # put in decoder to get likelihood
        state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z)
        reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=-1)
        if self.reconstruct_all_steps:
            reward_loss = torch.mean(reward_loss, dim=1)
        if self.use_state_decoder:
            state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=-1)
            if self.reconstruct_all_steps:
                state_loss = torch.mean(state_loss, dim=1)
            loss = self.loss_weight_state * state_loss + self.loss_weight_reward * reward_loss
        else:
            state_loss = torch.zeros(1)
            loss = self.loss_weight_reward * reward_loss

        loss = torch.sum(loss)
        self.optimizer_decoder.zero_grad()
        loss.backward()
        self.optimizer_decoder.step()

        return ptu.get_numpy(loss) / self.batch_size, \
               ptu.get_numpy(torch.sum(state_loss)) / self.batch_size, \
               ptu.get_numpy(torch.sum(reward_loss)) / self.batch_size

    def validate(self, indices):
        # get data from replay buffer
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.validation_batch_size,
                                                               normalize=self.use_data_normalization)

        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.validation_batch_size)
        # prepare for usage in decoder
        decoder_action = ptu.from_numpy(data['actions'])[:, -1, :]
        decoder_state = ptu.from_numpy(data['observations'])[:, -1, :]
        decoder_next_state = ptu.from_numpy(data['next_observations'])[:, -1, :]
        decoder_reward = ptu.from_numpy(data['rewards'])[:, -1, :]

        if self.use_state_diff:
            decoder_state_target = (decoder_next_state - decoder_state)[:, :self.state_reconstruction_clip]
        else:
            decoder_state_target = decoder_next_state[:, :self.state_reconstruction_clip]

        z, y = self.encoder(encoder_input)
        state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z)
        reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)
        if self.use_state_decoder:
            state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
        else:
            state_loss = torch.tensor(0)

        return ptu.get_numpy(torch.sum(state_loss)) / self.validation_batch_size, \
               ptu.get_numpy(torch.sum(reward_loss)) / self.validation_batch_size

    def early_stopping(self, epoch, loss):
        if loss < self.lowest_loss:
            if int(os.environ['DEBUG']) == 1:
                print("Found new minimum at Epoch " + str(epoch))
            self.lowest_loss = loss
            self.lowest_loss_epoch = epoch
            torch.save(self.encoder.state_dict(), self.encoder_path)
            torch.save(self.decoder.state_dict(), self.decoder_path)
        if epoch - self.lowest_loss_epoch > self.early_stopping_threshold:
            return True
        else:
            return False

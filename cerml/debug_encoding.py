import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.core import logger


class DecoderOnlyLatent(nn.Module):
    '''
    Predict the reward only from the latent space, not the current state and action. This should produce
    significantly worse results if the latent space does not just represent the reward.
    '''
    def __init__(self,
                 action_dim,
                 state_dim,
                 reward_dim,
                 z_dim,
                 net_complex):
        super(DecoderOnlyLatent, self).__init__()

        true_reward_decoder_input_size = state_dim + action_dim + z_dim
        self.decoder_hidden_size = int(true_reward_decoder_input_size * net_complex)
        self.decoder_input_size = z_dim

        self.net_reward_decoder = Mlp(
            hidden_sizes=[self.decoder_hidden_size, self.decoder_hidden_size],
            input_size=self.decoder_input_size,
            output_size=reward_dim
        )

    def forward(self, z):
        reward_estimate = self.net_reward_decoder(z)
        return reward_estimate


class EncodingDebugger:

    def __init__(self,
                 action_dim,
                 state_dim,
                 reward_dim,
                 z_dim,
                 net_complex,
                 encoder,
                 replay_buffer,
                 batch_size,
                 num_classes,
                 lr_decoder,
                 state_reconstruction_clip,
                 train_val_percent,
                 eval_interval,
                 early_stopping_threshold,
                 experiment_log_dir,
                 optimizer_class=optim.Adam
                 ):

        self.train_val_percent = train_val_percent
        self.batch_size = batch_size
        self.state_reconstruction_clip = state_reconstruction_clip
        self.num_classes = num_classes
        self.encoder = encoder
        self.experiment_log_dir = experiment_log_dir
        self.replay_buffer = replay_buffer
        self.eval_interval = eval_interval
        self.lr_decoder = lr_decoder
        self.optimizer_class = optimizer_class
        self.early_stopping_threshold = early_stopping_threshold

        self.factor_state_loss = 1
        self.factor_reward_loss = self.state_reconstruction_clip
        self.loss_weight_state = self.factor_state_loss / (self.factor_state_loss + self.factor_reward_loss)
        self.loss_weight_reward = self.factor_reward_loss / (self.factor_state_loss + self.factor_reward_loss)

        self.debug_decoder = DecoderOnlyLatent(action_dim, state_dim, reward_dim, z_dim, net_complex)

        self.optimizer_decoder = self.optimizer_class(
            self.debug_decoder.parameters(),
            lr=self.lr_decoder,
        )
        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        self.temp_path = os.path.join(os.getcwd(), '.temp', self.experiment_log_dir.split('/')[-1])
        self.decoder_path = os.path.join(self.temp_path, 'debugdecoder.pth')
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

    def record_debug_info(self, decoder_reconstruction_steps):
        self.train_decoder(decoder_reconstruction_steps)

    def train_decoder(self, epochs):
        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        train_losses, train_val_losses, val_losses = [], [], []

        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        for epoch in range(epochs):
            loss = self.decoder_training_step(train_indices, epoch)
            train_losses.append(loss)

            if epoch % self.eval_interval == 0:
                train_val_loss = self.decoder_validate(train_indices)
                val_loss = self.decoder_validate(val_indices)

                train_val_losses.append(train_val_loss)
                val_losses.append(val_loss)

                if self.decoder_early_stopping(epoch, val_loss):
                    print("Early stopping at epoch " + str(epoch))
                    break

        self.debug_decoder.load_state_dict(torch.load(self.decoder_path))

        validation_train = self.decoder_validate(train_indices)
        validation_val = self.decoder_validate(val_indices)

        logger.record_tabular("Only_latent_reconstruction_train_val_reward_loss", validation_train)
        logger.record_tabular("Only_latent_reconstruction_val_reward_loss", validation_val)
        logger.record_tabular("Only_latent_reconstruction_epochs", epoch + 1)

    def decoder_training_step(self, indices, epoch):
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=True)
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        decoder_reward = ptu.from_numpy(data['rewards'])[:, -1, :]

        self.debug_decoder.to(encoder_input.device)

        y_distribution, z_distributions = self.encoder.encode(encoder_input)

        reward_losses = ptu.zeros(self.batch_size, self.num_classes)

        for y in range(self.num_classes):
            z, _ = self.encoder.sample_z(y_distribution, z_distributions, y_usage="specific", y=y)

            # Important: Detach here, s.t. the gradients are NOT backpropagated into the encoder
            z = z.detach()

            reward_estimate = self.debug_decoder(z)
            reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)
            reward_losses[:, y] = self.loss_weight_reward * reward_loss

        loss = torch.sum(torch.mul(y_distribution.probs,  reward_losses))

        self.optimizer_decoder.zero_grad()
        loss.backward()
        self.optimizer_decoder.step()

        return ptu.get_numpy(loss)/self.batch_size

    def decoder_validate(self, indices):
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=True)
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        decoder_reward = ptu.from_numpy(data['rewards'])[:, -1, :]

        z, y = self.encoder(encoder_input)
        reward_estimate = self.debug_decoder(z)
        reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)
        return ptu.get_numpy(torch.sum(reward_loss)) / self.batch_size

    def decoder_early_stopping(self, epoch, loss):
        if loss < self.lowest_loss:
            if int(os.environ['DEBUG']) == 1:
                print("Found new minimum at Epoch " + str(epoch))
            self.lowest_loss = loss
            self.lowest_loss_epoch = epoch
            torch.save(self.debug_decoder.state_dict(), self.decoder_path)
        if epoch - self.lowest_loss_epoch > self.early_stopping_threshold:
            return True
        else:
            return False

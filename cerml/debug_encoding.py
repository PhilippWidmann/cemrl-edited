import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_categorical_dtype
import sklearn
from sklearn.neighbors import BallTree
from scipy.special import digamma

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.core import logger


def mutual_information(x, y, n_neighbors=5):
    if is_integer_dtype(x.dtype) or is_categorical_dtype(x.dtype):
        if is_integer_dtype(y.dtype) or is_categorical_dtype(y.dtype):
            return np.log2(np.e) * max(sklearn.metrics.mutual_info_score(x, y), 0.0)
        elif is_float_dtype(y.dtype):
            x = pd.array(x, dtype='category')
            return mutual_information_mixed(y, x, normalization=False, n_neighbors=n_neighbors)
        else:
            raise ValueError(f'Datatype {y.dtype} cannot be identified as continuous or discrete '
                             f'(for computing mutual information).')
    elif is_float_dtype(x.dtype):
        if is_integer_dtype(y.dtype) or is_categorical_dtype(y.dtype):
            y = pd.array(y, dtype='category')
            return mutual_information_mixed(x, y, normalization=False, n_neighbors=n_neighbors)
        elif is_float_dtype(y.dtype):
            return mutual_information_continuous(x, y, n_neighbors=n_neighbors)
        else:
            raise ValueError(f'Datatype {y.dtype} cannot be identified as continuous or discrete '
                             f'(for computing mutual information).')
    else:
        raise ValueError(f'Datatype {x.dtype} cannot be identified as continuous or discrete '
                         f'(for computing mutual information).')


def mutual_information_mixed(x, classes, normalization=False, n_neighbors=5):
    if classes.ndim != 1:
        raise ValueError("Discrete variable must be one-dimensional.")
    if classes.shape[0] != x.shape[0]:
        raise ValueError("First dimension of discrete and continuous variable must match.")

    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))

    nn_full = BallTree(x, metric='chebyshev')
    n = x.shape[0]
    d = np.zeros(classes.shape)
    n_classes = np.zeros(len(classes.categories))

    for i, c in enumerate(classes.categories):
        n_classes[i] = np.sum(classes == c)
        nn_class = BallTree(x[classes == c], metric='chebyshev')
        # BallTree considers a point its own closest neighbor, so use n_neighbors+1
        d_temp, _ = nn_class.query(x[classes == c], k=n_neighbors + 1, return_distance=True)
        d[classes == c] = d_temp[:, -1]

    # BallTree counts a point as its own neighbor. Thus subtract 1.
    m = nn_full.query_radius(x, r=d, count_only=True) - 1

    mutual_info = np.log2(np.e) * (digamma(n) - np.average(digamma(n_classes), weights=n_classes) +
                                   digamma(n_neighbors) - np.average(digamma(m)))

    if normalization:
        class_count = np.array([np.sum(classes == c) for c in classes.categories])
        class_prob = class_count / np.sum(class_count)
        discriminator_entropy = - np.sum(class_prob * np.log2(class_prob))
        mutual_info = mutual_info / discriminator_entropy

    return max(mutual_info, 0.0)


def mutual_information_continuous(x, y, n_neighbors=5):
    if x.shape[0] != y.shape[0]:
        raise ValueError("First dimension of variables must match.")
    n = x.shape[0]

    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))

    z = np.concatenate((x, y), axis=1)

    nn_z = BallTree(z, metric='chebyshev')
    nn_x = BallTree(x, metric='chebyshev')
    nn_y = BallTree(y, metric='chebyshev')

    dist, _ = nn_z.query(z, k=n_neighbors+1, return_distance=True)
    # Algorithm requires finding points with strictly smaller distance; -1E-16 is an approximation
    dist = dist[:, -1] - 1E-16
    n_x = nn_x.query_radius(x, r=dist, count_only=True) - 1
    n_y = nn_y.query_radius(y, r=dist, count_only=True) - 1

    mutual_info = np.log2(np.e) * (digamma(n_neighbors) - np.mean(digamma(n_x + 1)) - np.mean(digamma(n_y + 1))
                                   + digamma(n))
    return max(mutual_info, 0.0)


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
                 decoder,
                 replay_buffer,
                 batch_size,
                 validation_batch_size,
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
        self.decoder = decoder
        self.experiment_log_dir = experiment_log_dir
        self.replay_buffer = replay_buffer
        self.eval_interval = eval_interval
        self.lr_decoder = lr_decoder
        self.optimizer_class = optimizer_class
        self.early_stopping_threshold = early_stopping_threshold
        self.validation_batch_size = validation_batch_size

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
        self.validate_z_dependency()
        self.compute_mi_scores(repetitions=5)

    def compute_mi_scores(self, repetitions=1):
        all_ind, ind_temp = self.replay_buffer.get_train_val_indices(self.train_val_percent)
        all_ind = np.concatenate((all_ind, ind_temp))

        val_batch_size = min(self.validation_batch_size, len(all_ind))

        mi_z_base, mi_z_spec, mi_z_r = [], [], []
        mi_y_base, mi_y_spec, mi_y_r = [], [], []
        mi_r_base, mi_r_spec = [], []
        for i in range(repetitions):
            data = self.replay_buffer.sample_random_few_step_batch(all_ind, val_batch_size, normalize=True)

            encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
            y_distribution, z_distribution = self.encoder.encode(encoder_input)
            z, y = self.encoder.sample_z(y_distribution, z_distribution, y_usage='most_likely', sampler='mean')
            z, y = ptu.get_numpy(z), ptu.get_numpy(y)

            true_task_base = np.array([d[-1, 0]['base_task'] for d in data['true_tasks']])
            true_task_spec = pd.array([d[-1, 0]['specification'] for d in data['true_tasks']], dtype='category')
            r = data['rewards'][:, -1, :]

            mi_z_base.append(mutual_information(z, true_task_base))
            mi_z_spec.append(mutual_information(z, true_task_spec))
            mi_z_r.append(mutual_information(z, r))
            mi_y_base.append(mutual_information(y, true_task_base))
            mi_y_spec.append(mutual_information(y, true_task_spec))
            mi_y_r.append(mutual_information(y, r))
            mi_r_base.append(mutual_information(r, true_task_base))
            mi_r_spec.append(mutual_information(r, true_task_spec))

        logger.record_tabular("MI(z, true_task_base)", np.median(mi_z_base))
        logger.record_tabular("MI(z, true_task_spec)", np.median(mi_z_spec))
        logger.record_tabular("MI(z, reward)", np.median(mi_z_r))

        logger.record_tabular("MI(y, true_task_base)", np.median(mi_y_base))
        logger.record_tabular("MI(y, true_task_spec)", np.median(mi_y_spec))
        logger.record_tabular("MI(y, reward)", np.median(mi_y_r))

        logger.record_tabular("MI(r, true_task_base)", np.median(mi_r_base))
        logger.record_tabular("MI(r, true_task_spec)", np.median(mi_r_spec))

        if repetitions > 1:
            logger.record_tabular("ALL_MI(z, true_task_base)", np.sort(mi_z_base))
            logger.record_tabular("ALL_MI(z, true_task_spec)", np.sort(mi_z_spec))
            logger.record_tabular("ALL_MI(z, reward)", np.sort(mi_z_r))

            logger.record_tabular("ALL_MI(y, true_task_base)", np.sort(mi_y_base))
            logger.record_tabular("ALL_MI(y, true_task_spec)", np.sort(mi_y_spec))
            logger.record_tabular("ALL_MI(y, reward)", np.sort(mi_y_r))

            logger.record_tabular("ALL_MI(r, true_task_base)", np.sort(mi_r_base))
            logger.record_tabular("ALL_MI(r, true_task_spec)", np.sort(mi_r_spec))

    def validate_z_dependency(self):
        all_ind, ind_temp = self.replay_buffer.get_train_val_indices(self.train_val_percent)
        all_ind = np.concatenate((all_ind, ind_temp))
        np.random.shuffle(all_ind)

        reward_loss = torch.tensor(0.0, device=ptu.device)
        for i in range(int(np.ceil(all_ind.size / self.validation_batch_size))):
            upper = min(all_ind.size, (i+1) * self.validation_batch_size)
            ind = list(range(i * self.validation_batch_size, upper))
            data = self.replay_buffer.sample_data(ind)

            decoder_action = ptu.from_numpy(data['actions'])
            decoder_state = ptu.from_numpy(data['observations'])
            decoder_next_state = ptu.from_numpy(data['next_observations'])

            # To simulate random z from the distribution, just change order
            np.random.shuffle(data['task_indicators'])
            decoder_z = ptu.from_numpy(data['task_indicators'])

            decoder_reward = ptu.from_numpy(data['rewards'])

            self.decoder.to(decoder_action.device)
            _, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, decoder_z)
            reward_loss += torch.sum((reward_estimate - decoder_reward) ** 2)

        logger.record_tabular("Random_z_reconstruction_val_reward_loss",
                              ptu.get_numpy(reward_loss) / all_ind.size)

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

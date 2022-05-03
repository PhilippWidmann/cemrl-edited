import warnings
import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu


class StackedReplayBuffer:
    def __init__(self, max_replay_buffer_size,
                 encoder_time_steps,
                 decoder_time_window,
                 max_group_length,
                 observation_dim,
                 action_dim,
                 task_indicator_dim,
                 permute_samples,
                 sampling_mode=None):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._task_indicator_dim = task_indicator_dim
        self._max_replay_buffer_size = max_replay_buffer_size

        self._observations = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.float32)
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((max_replay_buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        # task indicator computed through encoder
        self._base_task_indicators = np.zeros(max_replay_buffer_size, dtype=np.float32)
        self._next_base_task_indicators = np.zeros(max_replay_buffer_size, dtype=np.float32)
        self._task_indicators = np.zeros((max_replay_buffer_size, task_indicator_dim), dtype=np.float32)
        self._next_task_indicators = np.zeros((max_replay_buffer_size, task_indicator_dim), dtype=np.float32)
        self._true_task = np.zeros((max_replay_buffer_size, 1), dtype=object)  # filled with dicts with keys 'base', 'specification'

        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self.encoder_time_steps = encoder_time_steps
        # Preprocess decoder window: Make endpoint inclusive, replace Inf with max_path_length
        self.max_group_length = max_group_length
        self.decoder_full_episode_window = decoder_time_window == [-np.inf, np.inf]
        decoder_time_window[0] = max(decoder_time_window[0], -max_group_length)
        decoder_time_window[1] = min(decoder_time_window[1] + 1, max_group_length + 1)
        self.decoder_time_window = tuple(decoder_time_window)
        self._top = 0
        self._top_group = 0
        self._size = 0

        # allowed points specify locations in the buffer, that, alone or together with the <self.time_step> last entries
        # can be sampled
        self._allowed_points = np.zeros(max_replay_buffer_size, dtype=bool)
        self._first_timestep = -np.ones(max_replay_buffer_size, dtype=int)
        self._first_timestep_group = -np.ones(max_replay_buffer_size, dtype=int)
        self._exploration_trajectory = np.zeros(max_replay_buffer_size, dtype=bool)

        self._train_indices = []
        self._val_indices = []
        self.stats_dict = None
        self.task_info_dict = {}

        self.permute_samples = permute_samples
        self.sampling_mode = sampling_mode

    def add_episode_group(self, episode_list, task_nr=None):
        for episode in episode_list[0]:
            self.add_episode(episode, task_nr)
        num_samples = episode_list[1]
        self._top_group = (self._top_group + num_samples) % self._max_replay_buffer_size
        return num_samples

    def add_episode(self, episode, task_nr=None):
        # Assume all array are same length (as they come from same rollout)
        length = episode['observations'].shape[0]

        # check, if whole episode fits into buffer
        if length >= self._max_replay_buffer_size:
            error_string = \
                "-------------------------------------------------------------------------------------------\n\n" \
                "ATTENTION:\n" \
                "The current episode was longer than the replay buffer and could not be fitted in.\n" \
                "Please consider decreasing the maximum episode length or increasing the task buffer size.\n\n" \
                "-------------------------------------------------------------------------------------------"
            print(error_string)
            return

        indices_list = np.array([(i + self._top) % self._max_replay_buffer_size for i in range(length)])

        self._observations[indices_list] = episode['observations']
        self._next_obs[indices_list] = episode['next_observations']
        self._actions[indices_list] = episode['actions']
        self._rewards[indices_list] = episode['rewards']
        self._task_indicators[indices_list] = episode['task_indicators']
        self._base_task_indicators[indices_list] = episode['base_task_indicators']
        self._next_task_indicators[indices_list] = episode['next_task_indicators']
        self._next_base_task_indicators[indices_list] = episode['next_base_task_indicators']
        self._terminals[indices_list] = episode['terminals']
        self._true_task[indices_list] = episode['true_tasks']

        # Update allowed points with new indices
        self._allowed_points[indices_list] = True
        self._first_timestep[indices_list] = self._top
        self._first_timestep_group[indices_list] = self._top_group
        self._exploration_trajectory[indices_list] = [episode['agent_infos'][i]['exploration_trajectory'] for i in range(len(indices_list))]
        # Reset start for next episode in buffer in case we overwrite the start
        next_index = (indices_list[-1] + 1) % self._max_replay_buffer_size
        if -1 < self._first_timestep[next_index]:
            # Todo: This should delete (or at least remove from availability) an episode group that has been partially overwritten
            self._first_timestep[self._first_timestep == self._first_timestep[next_index]] = next_index
            self._first_timestep_group[self._first_timestep_group == self._first_timestep_group[next_index]] = next_index

        # Increase buffer size and set _top to new end
        self._advance_multi(length)

        # Store info about task
        # Todo: task_nr parameter from Lukas' code currently unused. Is this necessary?
        if task_nr is not None:
            bt = episode['true_tasks'][0, 0]['base_task']
            if bt in self.task_info_dict.keys():
                if task_nr in self.task_info_dict[bt].keys():
                    self.task_info_dict[bt][task_nr].append(np.sum(episode['rewards']))
                else:
                    self.task_info_dict[bt][task_nr] = [np.sum(episode['rewards'])]
            else:
                self.task_info_dict[bt] = {task_nr: [np.sum(episode['rewards'])]}

    def _advance_multi(self, length):
        self._top = (self._top + length) % self._max_replay_buffer_size
        self._size = min(self._size + length, self._max_replay_buffer_size)

    def size(self):
        return self._size

    def get_allowed_points(self, include_exploration=True):
        if include_exploration:
            return np.where(self._allowed_points)[0]
        else:
            return np.where(self._allowed_points & ~self._exploration_trajectory)[0]

    def sample_indices(self, points, batch_size, prio=None):
        rng = np.random.default_rng()
        prio = self.sampling_mode if prio is None else prio

        if prio == 'linear':
            # prioritized version: later samples get more weight
            weights = np.linspace(0.9, 0.1, self._size)[(self._top - 1) - points]
            weights = weights / np.sum(weights)
            indices = rng.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False, p=weights)
        elif prio is None:
            indices = rng.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False)
        else:
            raise NotImplementedError(f'Sampling method {prio} has not been implemented yet.')

        return indices

    def get_data_batch(self, indices):
        return dict(
            observations=self._observations[indices],
            next_observations=self._next_obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            task_indicators=self._task_indicators[indices],
            base_task_indicators=self._base_task_indicators[indices],
            next_task_indicators=self._next_task_indicators[indices],
            next_base_task_indicators=self._next_base_task_indicators[indices],
            sparse_rewards=self._sparse_rewards[indices],
            terminals=self._terminals[indices],
            true_tasks=self._true_task[indices]
        )

    # Sequence sample functions
    def get_few_step_data_batch(self, points, batch_size, normalize=True, step_mode='encoder'):
        # the points in time together with their <time_step> many entries from before are sampled
        # check if the current point is still in the same 'episode' else take episode start
        if step_mode == 'encoder':
            all_indices = points[:, np.newaxis] + np.arange(-self.encoder_time_steps, 1)[np.newaxis, :]
        elif step_mode == 'decoder':
            all_indices = points[:, np.newaxis] + np.arange(*self.decoder_time_window)[np.newaxis, :]
        else:
            raise ValueError(f'step_mode={step_mode} is unknown. This should not happen and is probably a bug.')
        all_indices = all_indices % self._max_replay_buffer_size

        padding_mask = (self._first_timestep[all_indices] != self._first_timestep[points][:, np.newaxis])
        if step_mode == 'decoder' and self.decoder_full_episode_window:
            padding_mask = (self._first_timestep_group[all_indices] != self._first_timestep_group[points][:, np.newaxis])
            # Todo: This works only if we always fully overwrite episodes (never partially).
            # Todo: No problem as long as buffer_size = K * max_path_length, but fix nonetheless
            if np.sum(~padding_mask) != (batch_size * self.max_group_length):
                warnings.warn('Not all sample episodes had the same length. Reconstruction of the whole episode is unoptimized.')
            else:
                all_indices = np.reshape(all_indices[~padding_mask], (batch_size, self.max_group_length))
                padding_mask = np.zeros((batch_size, self.max_group_length), dtype=bool)

        data = self.get_data_batch(all_indices)

        if normalize:
            data = self.normalize_data(data)

        # TODO: Change in case keys are changed
        # Set data outside of trajectory to 0
        data['observations'][padding_mask] = 0.
        data['next_observations'][padding_mask] = 0.
        data['actions'][padding_mask] = 0.
        data['rewards'][padding_mask] = 0.
        data['task_indicators'][padding_mask] = 0.
        data['base_task_indicators'][padding_mask] = 0.
        data['next_task_indicators'][padding_mask] = 0.
        data['next_base_task_indicators'][padding_mask] = 0.
        data['terminals'][padding_mask] = 0.

        return data, padding_mask

    def sample_random_few_step_data_batch(self, points, batch_size, normalize=True, normalize_sac=False, prio=None,
                                          return_encoder_data=True, return_decoder_data=False, return_sac_data=False):
        ''' batch of unordered small sequences of transitions '''
        indices = self.sample_indices(points, batch_size, prio=prio)
        encoder_data, encoder_padding = None, None
        decoder_data, decoder_padding = None, None
        sac_data = None
        if return_encoder_data:
            encoder_data, encoder_padding = self.get_few_step_data_batch(indices, batch_size, normalize=normalize,
                                                                         step_mode='encoder')
        if return_decoder_data:
            decoder_data, decoder_padding = self.get_few_step_data_batch(indices, batch_size, normalize=normalize,
                                                                         step_mode='decoder')
        if return_sac_data:
            sac_data = self.get_data_batch(indices)
            if normalize_sac:
                sac_data = self.normalize_data(sac_data)

        res = [val for val in [encoder_data, encoder_padding, decoder_data, decoder_padding, sac_data]
               if val is not None]
        return tuple(res)

    # Relabeler util function
    def relabel_z(self, start, batch_size, z, next_z, y, next_y):
        points = self.get_allowed_points()[start:start + batch_size]
        self._task_indicators[points] = z
        self._next_task_indicators[points] = next_z
        self._base_task_indicators[points] = y
        self._next_base_task_indicators[points] = next_y

    def get_train_val_indices(self, train_val_percent):
        # Split all data from replay buffer into training and validation set
        # not very efficient but hopefully readable code in this function
        points = np.array(self.get_allowed_points())

        train_indices = np.array(self._train_indices)
        val_indices = np.array(self._val_indices)

        points = points[np.isin(points, train_indices, invert=True)]
        points = points[np.isin(points, val_indices, invert=True)]
        points = np.random.permutation(points)
        splitter = int(points.shape[0] * train_val_percent)
        new_train_indices = points[:splitter]
        new_val_indices = points[splitter:]
        self._train_indices += new_train_indices.tolist()
        self._val_indices += new_val_indices.tolist()
        self._train_indices.sort()
        self._val_indices.sort()

        return np.array(self._train_indices), np.array(self._val_indices)

    def make_encoder_data(self, data, batch_size, padding_mask=None, mode='multiply', exclude_last_timestep=True):
        # MLP encoder input: state of last timestep + state, action, reward of all timesteps before
        # input is in form [[t-N], ... [t-1], [t]]
        # therefore set action and reward of last timestep = 0
        # Returns: [batch_size, timesteps, obs+action+reward dim]
        # assumes, that a flat encoder flattens the data itself

        observations_encoder_input = torch.from_numpy(data['observations'])
        actions_encoder_input = torch.from_numpy(data['actions'])
        rewards_encoder_input = torch.from_numpy(data['rewards'])
        next_observations_encoder_input = torch.from_numpy((data['next_observations']))

        if exclude_last_timestep:
            observations_encoder_input = observations_encoder_input.detach().clone()[:, :-1, :]
            actions_encoder_input = actions_encoder_input.detach().clone()[:, :-1, :]
            rewards_encoder_input = rewards_encoder_input.detach().clone()[:, :-1, :]
            next_observations_encoder_input = next_observations_encoder_input.detach().clone()[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]
                # If we had sampled the very first step of an episode, no data remains after removing this point.
                # Only in this case, treat one step of padding as actual input to avoid implementation problems.
                samples_without_data = np.sum(~padding_mask, axis=1) == 0
                padding_mask[samples_without_data, -1] = False

        # size: [batch_size, time_steps, obs+action+reward]
        encoder_input = torch.cat(
            [observations_encoder_input, actions_encoder_input, rewards_encoder_input, next_observations_encoder_input],
            dim=-1)

        if self.permute_samples:
            perm = torch.randperm(encoder_input.shape[1]).long()
            encoder_input = encoder_input[:, perm]

        if self.encoder_time_steps == -1:
            raise NotImplementedError('The convention time_steps==-1 equals variable length input has not been implemented.')

        return encoder_input.to(ptu.device), padding_mask

    def get_stats(self):
        values_dict = dict(
            observations=self._observations[:self._size],
            next_observations=self._next_obs[:self._size],
            actions=self._actions[:self._size],
            rewards=self._rewards[:self._size],
        )
        stats_dict = dict(
            observations={},
            next_observations={},
            actions={},
            rewards={},
        )
        for key in stats_dict.keys():
            stats_dict[key]["max"] = values_dict[key].max(axis=0)
            stats_dict[key]["min"] = values_dict[key].min(axis=0)
            stats_dict[key]["mean"] = values_dict[key].mean(axis=0)
            stats_dict[key]["std"] = values_dict[key].std(axis=0)
        return stats_dict

    def sample_task_indicator(self):
        index_possible = ~self._exploration_trajectory & self._allowed_points
        if np.sum(index_possible) != 0:
            index = np.random.choice(range(self._max_replay_buffer_size), p=index_possible/index_possible.sum())
        else:
            index = 0
        return ptu.from_numpy(self._task_indicators[np.newaxis, index]), \
               ptu.from_numpy(self._base_task_indicators[np.newaxis, index])

    def normalize_data(self, data):
        for key in self.stats_dict.keys():
            data[key] = (data[key] - self.stats_dict[key]["mean"]) / (self.stats_dict[key]["std"] + 1e-8)
        return data

    def check_enc(self):

        indices = self.get_allowed_points()
        true_task_list = np.squeeze(self._true_task[indices])
        # Use arrays that are created once
        base_tasks_array = np.array([a['base_task'] for a in true_task_list])
        spec_tasks_array = np.array([a['specification'] for a in true_task_list])
        # Find unique base tasks
        base_tasks = np.unique(base_tasks_array)

        base_spec_dict = {}
        for base_task in base_tasks:
            # Find all unique specifications per base task
            spec_list = np.unique(spec_tasks_array[base_tasks_array == base_task])
            base_spec_dict[base_task] = spec_list

        encoding_storage = {}
        for base in base_spec_dict.keys():
            spec_encoding_dict = {}
            for i, spec in enumerate(base_spec_dict[base]):
                task_indices = np.where(np.logical_and(base_tasks_array == base, spec_tasks_array == spec))[0]

                # Get mean and std of estimated specs
                encodings = self._task_indicators[task_indices]
                mean = np.mean(encodings, axis=0)
                std = np.std(encodings, axis=0)
                # Get bincount of base tasks
                base_task_estimate = np.bincount(self._base_task_indicators[task_indices].astype(int))
                # Store estimated values in dict
                spec_encoding_dict[spec] = dict(mean=mean, std=std, base=base_task_estimate)

            encoding_storage[base] = spec_encoding_dict

        return encoding_storage

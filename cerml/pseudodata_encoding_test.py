import json
import os.path

import click
import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt

from cerml.encoder_decoder_networks import DecoderMDP, Encoder, PriorPz
from cerml.experimental_encoder_decoder_networks import SpecialOmissionDecoder, SpecialOmissionEncoder, NoActionEncoder, \
    NoOpEncoder
from cerml.reconstruction_trainer import ReconstructionTrainer
from cerml.stacked_replay_buffer import StackedReplayBuffer
from analysis.plot_episode import plot_per_episode
from configs.default import default_config
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu
from philipp_runner import deep_update_dict


class PseudodataGenerator:
    def __init__(
            self,
            min_target,
            max_target,
            episode_length,
    ):
        self.tasks = np.arange(min_target, max_target + 0.1, 0.1)
        self.num_tasks = len(self.tasks)
        self.episode_length = episode_length
        self.obs_dim = 1
        self.action_dim = 1

    def generate_episode(self, task_nr, target=None, modify_target=None, target_error=1.0, error_prob=1.0, mode='random_zigzag', dampen=1):
        states = np.zeros((self.episode_length + 1, self.obs_dim))
        actions = np.zeros((self.episode_length, self.action_dim))
        rewards = np.zeros((self.episode_length, 1))

        states[0] = 0.0
        modify_target = False if modify_target is None else modify_target
        if target is None:
            target = self.tasks[task_nr]
            modify_target = True
        true_target = target

        intermediate_target = None
        final_target = target
        if modify_target:
            if np.random.uniform() <= error_prob:
                if mode == 'random':
                    final_target = np.random.choice(self.tasks)
                elif mode == 'mirror':
                    final_target = -target
                elif mode == 'zigzag':
                    intermediate_target = -target
                    final_target = target
                elif mode == 'random_zigzag':
                    target = np.random.choice(self.tasks)
                    intermediate_target = -target
                    final_target = target
                elif mode == 'realistic':
                    intermediate_target = np.random.uniform(-7.5, 7.5)
                    final_target = target
                else:
                    raise ValueError('Typo in mode')
            final_target = dampen * np.random.uniform(final_target - target_error, final_target + target_error)

        target = intermediate_target if intermediate_target is not None else final_target
        for i in range(self.episode_length):
            actions[i] = self._generate_actions(states[i], target)
            states[i + 1] = self._generate_state(states[i], actions[i], target)
            rewards[i] = self._generate_reward(states[i], actions[i], states[i + 1], true_target)
            if (intermediate_target is not None) and (abs(states[i+1] - intermediate_target) <= 0.01):
                target = final_target
                intermediate_target = None

        episode = {
            'observations': states[:-1],
            'next_observations': states[1:],
            'rewards': rewards,
            'actions': actions,
            'task_indicators': np.zeros((self.episode_length, 1)),
            'base_task_indicators': np.zeros(self.episode_length),
            'next_task_indicators': np.zeros((self.episode_length, 1)),
            'next_base_task_indicators': np.zeros(self.episode_length),
            'terminals': np.zeros((self.episode_length, 1), dtype=bool),
            'true_tasks': np.array(
                [[{'base_task': 0, 'specification': true_target}] for _ in range(self.episode_length)],
                dtype=object)
        }
        return episode

    def _generate_state(self, state, action, target, itr_to_target=75):
        if target >= 0:
            next_state = min(state[0] + self.tasks[-1] / itr_to_target, target)
        else:
            next_state = max(state[0] + self.tasks[0] / itr_to_target, target)
        return next_state

    def _generate_actions(self, state, target):
        return 0

    def _generate_reward(self, state, action, next_state, target):
        # return -(next_state[0] - target) ** 2
        return - abs(next_state[0] - target)


def init_networks(variant, obs_dim, action_dim):
    reward_dim = 1
    variant['env_params']['state_reconstruction_clip'] = obs_dim
    encoding_save_epochs = [0]
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=None,
                                      base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='specific',
                                      snapshot_gap=variant['algo_params']['snapshot_gap'],
                                      snapshot_points=encoding_save_epochs)
    # if not os.path.exists(experiment_log_dir):
    #    os.makedirs(experiment_log_dir)

    # instantiate networks
    net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    num_classes = variant['reconstruction_params']['num_classes']

    # set parameters if not given
    if variant['algo_params']['decoder_time_window'] is None:
        variant['algo_params']['decoder_time_window'] = [-time_steps, 0]

    # encoder used: single transitions or trajectories
    if variant['algo_params']['encoder_type'] == 'NoEncoder':
        # debug case: completely omit any encoding and only do SAC training
        encoder_class = NoOpEncoder
    elif variant['algo_params']['encoder_omit_input'] == 'action':
        encoder_class = NoActionEncoder
    elif variant['algo_params']['encoder_omit_input'] == 'special':
        encoder_class = SpecialOmissionEncoder
        variant['algo_params']['encoder_omit_input'] = None
    elif isinstance(variant['algo_params']['encoder_omit_input'], list):
        encoder_class = SpecialOmissionEncoder
    else:
        encoder_class = Encoder
        variant['algo_params']['encoder_omit_input'] = None

    encoder = encoder_class(
        obs_dim,
        action_dim,
        reward_dim,
        net_complex_enc_dec,
        variant['algo_params']['encoder_type'],
        variant['algo_params']['encoder_exclude_padding'],
        latent_dim,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        time_steps,
        variant['algo_params']['encoder_merge_mode'],
        relevant_input_indices=variant['algo_params']['encoder_omit_input']
    )

    if variant['algo_params']['decoder_omit_input'] == 'special':
        decoder_class = SpecialOmissionDecoder
    else:
        decoder_class = DecoderMDP

    decoder = decoder_class(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
        variant['env_params']['use_state_decoder'],
        variant['reconstruction_params']['use_next_state_for_reward_decoder'],
    )

    replay_buffer = StackedReplayBuffer(
        variant['algo_params']['max_replay_buffer_size'],
        time_steps,
        variant['algo_params']['decoder_time_window'],
        variant['algo_params']['max_path_length'],
        obs_dim,
        action_dim,
        latent_dim,
        variant['algo_params']['permute_samples'],
        variant['algo_params']['sampling_mode']
    )

    prior_pz = PriorPz(num_classes, latent_dim)

    reconstruction_trainer = ReconstructionTrainer(
        encoder,
        decoder,
        prior_pz,
        replay_buffer,
        variant['algo_params']['batch_size_reconstruction'],
        variant['algo_params']['batch_size_validation'],
        num_classes,
        latent_dim,
        time_steps,
        variant['reconstruction_params']['lr_decoder'],
        variant['reconstruction_params']['lr_encoder'],
        variant['reconstruction_params']['alpha_kl_z'],
        variant['reconstruction_params']['beta_kl_y'],
        variant['reconstruction_params']['alpha_kl_z_query'],
        variant['reconstruction_params']['beta_kl_y_query'],
        variant['reconstruction_params']['use_state_diff'],
        variant['reconstruction_params']['component_constraint_learning'],
        variant['env_params']['state_reconstruction_clip'],
        variant['env_params']['use_state_decoder'],
        variant['algo_params']['use_data_normalization'],
        variant['reconstruction_params']['train_val_percent'],
        variant['reconstruction_params']['eval_interval'],
        variant['reconstruction_params']['early_stopping_threshold'],
        experiment_log_dir,
        variant['util_params']['temp_dir'],
        variant['reconstruction_params']['prior_mode'],
        variant['reconstruction_params']['prior_sigma'],
        variant['algo_params']['data_usage_reconstruction'],
        variant['reconstruction_params']['reconstruct_all_timesteps']
    )
    return encoder, decoder, reconstruction_trainer, replay_buffer, experiment_log_dir


def plot_encoder(
        encoder: Encoder,
        pseudodata_generator: PseudodataGenerator,
        replay_buffer: StackedReplayBuffer,
        time_steps: int,
        episode_length: int,
        targets,
        modify_target=False,
        save_dir=None,
        save_name=None
):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    z_min, z_max = None, None
    for target in targets:
        episode = pseudodata_generator.generate_episode(0, target, modify_target=modify_target, error_prob=1.0)
        encoder_input = torch.zeros((episode_length + time_steps, 4), device=ptu.device, dtype=torch.float)
        # Get input
        o = episode['observations']
        a = episode['actions']
        r = episode['rewards']
        next_o = episode['next_observations']
        # Normalize
        stats_dict = replay_buffer.get_stats()
        o = ptu.from_numpy((o - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
        a = ptu.from_numpy((a - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
        r = ptu.from_numpy((r - stats_dict["rewards"]["mean"]) / (stats_dict["rewards"]["std"] + 1e-9))
        next_o = ptu.from_numpy(
            (next_o - stats_dict["next_observations"]["mean"]) / (stats_dict["next_observations"]["std"] + 1e-9))
        encoder_input[time_steps:] = torch.cat([o, a, r, next_o], dim=1)
        encoder_input = encoder_input.unsqueeze(0)
        padding_mask = np.zeros((1, time_steps + episode_length), dtype=bool)
        padding_mask[:, :(time_steps - 1)] = True
        z = torch.zeros((episode_length, 1))

        for i in range(episode_length):
            z[i], _ = encoder.forward(encoder_input[:, i:(i + time_steps), :],
                                      padding_mask=padding_mask[:, i:(i + time_steps)])
        z_max = max(torch.max(z).item(), z_max) if z_max is not None else torch.max(z).item()
        z_min = min(torch.min(z).item(), z_min) if z_min is not None else torch.min(z).item()

        episode['task_indicators'] = ptu.get_numpy(z)

        plot_per_episode(episode, 'task_indicators', fig_ax=(fig, ax[0]))
        plot_per_episode(episode, 'rewards', fig_ax=(fig, ax[1]))
        plot_per_episode(episode, 'observations', y_const='specification', fig_ax=(fig, ax[2]))

    fig.tight_layout()
    fig.show()
    if (save_dir is not None) and (save_name is not None):
        fig.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')

    return z_min, z_max


def plot_decoder(
        replay_buffer: StackedReplayBuffer,
        decoder: DecoderMDP,
        z_min,
        z_max=None,
        save_dir=None,
        save_name=None,
):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Prepare input data
    min_state = replay_buffer.get_stats()['observations']['min'][0]
    max_state = replay_buffer.get_stats()['observations']['max'][0]
    action_mean = replay_buffer.get_stats()['actions']['mean'][0]

    nr_points = 101
    nr_z_values = 30
    states = torch.linspace(min_state, max_state, steps=nr_points, device=ptu.device)
    actions = action_mean * torch.ones(nr_points, device=ptu.device)

    stats_dict = replay_buffer.get_stats()
    states_norm = ptu.from_numpy(
        (ptu.get_numpy(states) - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
    actions_norm = ptu.from_numpy(
        (ptu.get_numpy(actions) - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
    padding_mask = torch.zeros(nr_points, device=ptu.device, dtype=torch.bool)

    # Init collection array
    reward_preds = np.zeros((nr_z_values, nr_points))

    if z_max is None:
        z_list = [z_min]
    else:
        z_list = np.linspace(z_min, z_max, num=nr_z_values)

    if z_max is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(z_list)))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=z_min, vmax=z_max))
        fig.colorbar(sm)

    for i, z in enumerate(z_list):
        zs = z * torch.ones(nr_points, device=ptu.device)

        _, reward_pred_norm = decoder.forward(states_norm.unsqueeze(1), actions_norm.unsqueeze(1),
                                              states_norm.unsqueeze(1), zs.unsqueeze(1),
                                              padding_mask=padding_mask.unsqueeze(1))
        reward_pred = ptu.get_numpy(reward_pred_norm) * (stats_dict["rewards"]["std"] + 1e-9) + \
                      stats_dict["rewards"]["mean"]
        ax.plot(ptu.get_numpy(states), reward_pred, color=colors[i])
        reward_preds[i] = reward_pred.squeeze()

    ax.set_xlabel('position')
    ax.set_ylabel('reward prediction')
    ax.set_title('Family of decoder functions \n(colored by latent values z)')
    fig.tight_layout()
    fig.show()
    if (save_dir is not None) and (save_name is not None):
        fig.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')

    # Additionally, make contour plot
    fig_con, ax_con = plt.subplots()
    cf = ax_con.contourf(ptu.get_numpy(states), z_list, reward_preds)
    fig_con.colorbar(cf)
    ax_con.set_xlabel('position')
    ax_con.set_ylabel('z')
    ax_con.set_title('Contour plot of reward predictions')

    fig_con.tight_layout()
    fig_con.show()
    if (save_dir is not None) and (save_name is not None):
        fig_con.savefig(os.path.join(save_dir, 'contour-' + save_name), bbox_inches='tight')


DEFAULT = "../configs/pseudodata/cheetah-stationary-target-qR.json"
MLP_FULLEP = "../configs/pseudodata/cheetah-stationary-target-qR-TimestepMLP-fullEp.json"
MEANMLP_KL = "../configs/pseudodata/cheetah-stationary-target-qR-MeanTimestepMLP-queryKL.json"


@click.command()
@click.argument('config', default=DEFAULT)
def main(config):
    # Load config
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    # Make custom adjustments to the config
    # Mainly, since we want to fully train the encoder at once, not over multiple epochs
    variant['util_params']['base_log_dir'] = os.path.join('..', 'output_analysis', 'pseudodata')
    variant['util_params']['temp_dir'] = os.path.join('..', '.temp_cemrl', 'pseudodata')

    variant['reconstruction_params']['use_next_state_for_reward_decoder'] = False
    variant['reconstruction_params']['early_stopping_threshold'] = 2500
    variant['reconstruction_params']['net_complex_enc_dec'] = 10 * variant['reconstruction_params'][
        'net_complex_enc_dec']

    variant['algo_params']['num_reconstruction_steps'] = 50000
    os.environ['DEBUG'] = '1'
    os.environ['PLOT'] = '1'

    # Initialize PseudodataGenerator, Encoder, Decoder, ReconstructionTrainer, ReplayBuffer
    print('Initializing networks')
    pseudo_generator = PseudodataGenerator(variant['env_params']['task_min_target'],
                                           variant['env_params']['task_max_target'],
                                           variant['algo_params']['max_path_length'])
    encoder, decoder, reconstruction_trainer, replay_buffer, experiment_log_dir = \
        init_networks(variant, pseudo_generator.obs_dim, pseudo_generator.action_dim)
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    for net in [encoder, decoder]:
        net.to(device=ptu.device)

    # Generate pseudodata and put it into stacked replay buffer
    print('Generating data')
    for i in range(pseudo_generator.num_tasks):
        episode = pseudo_generator.generate_episode(i)
        replay_buffer.add_episode(episode)

    replay_buffer.stats_dict = replay_buffer.get_stats()

    # call reconstructionTrainer on this stacked_replay_buffer, train until completion
    print('Reconstruction training')
    reconstruction_trainer.train(variant['algo_params']['num_reconstruction_steps'],
                                 plot_save_file=os.path.join(experiment_log_dir, 'reconstruction-training-curve.png'))
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # visualize results with existing functions (e.g. on "episode rollout")
    print('Plotting encoder')
    z_min, z_max = plot_encoder(encoder, pseudo_generator, replay_buffer, variant['algo_params']['time_steps'],
                                variant['algo_params']['max_path_length'],
                                [-25, -15, -5, 0, 5, 15, 25],  # [-25, -12.5, 0, 12.5, 25],
                                save_dir=experiment_log_dir, save_name='latent-encodings.png')
    z_min_2, z_max_2 = plot_encoder(encoder, pseudo_generator, replay_buffer, variant['algo_params']['time_steps'],
                                    variant['algo_params']['max_path_length'],
                                    [-25, -15, -5, 0, 5, 15, 25],  # [-25, -12.5, 0, 12.5, 25],
                                    modify_target=True,
                                    save_dir=experiment_log_dir, save_name='latent-encodings-wrong-trajectory.png')

    z_min_2 = min(z_min, z_min_2)
    z_max_2 = max(z_max, z_max_2)

    # visualize decoder capability showing the function learned for a particular z
    print('Plotting decoder')
    plot_decoder(replay_buffer, decoder, z_min, z_max,
                 save_dir=experiment_log_dir, save_name='decoder-functions.png')
    plot_decoder(replay_buffer, decoder, z_min_2, z_max_2,
                 save_dir=experiment_log_dir, save_name='decoder-functions-all.png')


if __name__ == '__main__':
    main()

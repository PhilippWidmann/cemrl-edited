import json
import os.path

import click
import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt

from cemrl_edited.encoder_decoder_networks import DecoderMDP, Encoder, PriorPz
from cemrl_edited.reconstruction_trainer import ReconstructionTrainer
from cemrl_edited.stacked_replay_buffer import StackedReplayBuffer
from analysis.plot_episode import plot_per_episode
from configs.default import default_config
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu
from philipp_runner import deep_update_dict
from thesis_plot_progress import SMALL_SIZE, MEDIUM_SIZE, FIGSIZE_HALF


plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-3)    # legend fontsize; corrected because it visually appears larger
plt.rc('legend', title_fontsize=SMALL_SIZE)    # legend fontsize

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

    def generate_episode(self, task_nr, target=None, modify_target=None, target_error=0, error_prob=1.0, mode='overshoot', dampen=1):
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
                if mode == 'overshoot':
                    if np.random.uniform() < 0.5:
                        mode = 'overshoot_plus'
                    else:
                        mode = 'overshoot_minus'

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
                elif mode == 'overshoot_plus':
                    intermediate_target = 4/3*self.tasks[-1]
                    final_target = 4/3*self.tasks[-1]
                elif mode == 'overshoot_minus':
                    intermediate_target = 4/3*self.tasks[0]
                    final_target = 4/3*self.tasks[0]
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
                dtype=object),
            'agent_infos': np.array(
                [{'exploration_trajectory': True} for _ in range(self.episode_length)],
                dtype=object)
        }
        return episode

    def _generate_state(self, state, action, target, itr_to_target=150):
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

    encoder = Encoder(
        obs_dim,
        action_dim,
        reward_dim,
        net_complex_enc_dec,
        variant['algo_params']['encoder_type'],
        variant['algo_params']['encoder_exclude_padding'],
        latent_dim,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        variant['reconstruction_params']['state_preprocessing_dim'],
        variant['reconstruction_params']['simplified_state_preprocessor'],
        time_steps,
        variant['algo_params']['encoder_merge_mode'],
        relevant_input_indices=variant['algo_params']['encoder_omit_input']
    )

    decoder = DecoderMDP(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
        variant['env_params']['use_state_decoder'],
        encoder.state_preprocessor,
        variant['reconstruction_params']['use_next_state_for_reward_decoder'],
    )

    combined_trajectories = variant['algo_params']['num_trajectories_per_task'] \
                            + variant['algo_params']['num_exploration_trajectories_per_task']
    replay_buffer = StackedReplayBuffer(
        variant['algo_params']['max_replay_buffer_size'],
        time_steps,
        variant['algo_params']['decoder_time_window'],
        combined_trajectories * variant['algo_params']['max_path_length'],
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
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(FIGSIZE_HALF[0], 2*FIGSIZE_HALF[1]))
    z_min, z_max = None, None
    for mode in ['overshoot_plus', 'overshoot_minus']:
        ax[0].set_prop_cycle(None)
        ax[1].set_prop_cycle(None)
        for target in targets:
            episode = pseudodata_generator.generate_episode(0, target, modify_target=modify_target, mode=mode, error_prob=1.0)
            encoder_input = torch.zeros((episode_length + time_steps, 4), device=ptu.device, dtype=torch.float)
            # Get input
            o = episode['observations']
            a = episode['actions']
            r = episode['rewards']
            next_o = episode['next_observations']
            # Normalize
            # stats_dict = replay_buffer.get_stats()
            #o = ptu.from_numpy((o - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
            #a = ptu.from_numpy((a - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
            #r = ptu.from_numpy((r - stats_dict["rewards"]["mean"]) / (stats_dict["rewards"]["std"] + 1e-9))
            #next_o = ptu.from_numpy(
            #    (next_o - stats_dict["next_observations"]["mean"]) / (stats_dict["next_observations"]["std"] + 1e-9))
            o = ptu.from_numpy(o)
            a = ptu.from_numpy(a)
            r = ptu.from_numpy(r)
            next_o = ptu.from_numpy(next_o)
            encoder_input[time_steps:] = torch.cat([o, a, r, next_o], dim=1)
            encoder_input = encoder_input.unsqueeze(0)
            encoder_input = encoder_input.split(1, -1)
            padding_mask = np.zeros((1, time_steps + episode_length), dtype=bool)
            padding_mask[:, :(time_steps - 1)] = True
            z = torch.zeros((episode_length, 1))

            for i in range(episode_length):
                current_encoder_input = [j[:, i:(i + time_steps), :] for j in encoder_input]
                z[i], _ = encoder.forward(current_encoder_input,
                                          padding_mask=padding_mask[:, i:(i + time_steps)])
            z_max = max(torch.max(z).item(), z_max) if z_max is not None else torch.max(z).item()
            z_min = min(torch.min(z).item(), z_min) if z_min is not None else torch.min(z).item()

            episode['task_indicators'] = ptu.get_numpy(z)

            marker = "o" if mode == "overshoot_plus" else "^"
            markevery = 40 if mode == "overshoot_plus" else (20, 40)
            plot_per_episode(episode, 'task_indicators', fig_ax=(fig, ax[0]),
                             marker=marker, markevery=markevery, markersize=8)
            plot_per_episode(episode, 'observations', const='time_vs_specification', fig_ax=(fig, ax[1]),
                             marker=marker, markevery=markevery, markersize=8)

    # Edit plot labels
    ax[0].set_title(None)
    ax[0].set_xlabel("Time step $t$")
    ax[0].set_ylabel("$z$")
    ax[1].set_title(None)
    ax[1].set_xlabel("Time step $t$")
    ax[1].set_ylabel("Position")

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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE_HALF)

    # Prepare input data
    min_state = -26#replay_buffer.get_stats()['observations']['min'][0]
    max_state = 26#replay_buffer.get_stats()['observations']['max'][0]
    action_mean = 0#replay_buffer.get_stats()['actions']['mean'][0]

    nr_points = 101
    nr_z_values = 15
    states = torch.linspace(min_state, max_state, steps=nr_points, device=ptu.device)
    actions = action_mean * torch.ones(nr_points, device=ptu.device)
    states_norm = states
    actions_norm = actions

    #stats_dict = replay_buffer.get_stats()
    #states_norm = ptu.from_numpy(
    #    (ptu.get_numpy(states) - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
    #actions_norm = ptu.from_numpy(
    #    (ptu.get_numpy(actions) - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
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
        cbar = fig.colorbar(sm)
        cbar.set_label('$z$', rotation=90)

    for i, z in enumerate(z_list):
        zs = z * torch.ones(nr_points, device=ptu.device)

        _, reward_pred_norm = decoder.forward(states_norm.unsqueeze(1), actions_norm.unsqueeze(1),
                                              states_norm.unsqueeze(1), zs.unsqueeze(1),
                                              padding_mask=padding_mask.unsqueeze(1))
        reward_pred = ptu.get_numpy(reward_pred_norm)# * (stats_dict["rewards"]["std"] + 1e-9) + \
                      #stats_dict["rewards"]["mean"]
        ax.plot(ptu.get_numpy(states), reward_pred, color=colors[i])
        reward_preds[i] = reward_pred.squeeze()

    ax.set_xlabel('Position')
    ax.set_ylabel('Reward prediction')
    #ax.set_title('Family of decoder functions \n(colored by latent values z)')
    fig.tight_layout()
    fig.show()
    if (save_dir is not None) and (save_name is not None):
        fig.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')

    # Unused: Additionally, make contour plot
    # fig_con, ax_con = plt.subplots()
    # cf = ax_con.contourf(ptu.get_numpy(states), z_list, reward_preds)
    # fig_con.colorbar(cf)
    # ax_con.set_xlabel('position')
    # ax_con.set_ylabel('z')
    # ax_con.set_title('Contour plot of reward predictions')
    # fig_con.tight_layout()
    # fig_con.show()
    # if (save_dir is not None) and (save_name is not None):
    #    fig_con.savefig(os.path.join(save_dir, 'contour-' + save_name), bbox_inches='tight')


def train_and_plot(config, path_to_weights=None, plot_save_dir=None, plot_save_prefix=''):
    # Load config
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    # Make custom adjustments to the config
    # Mainly, since we want to fully train the encoder at once, not over multiple epochs
    variant['util_params']['base_log_dir'] = os.path.join('..', 'output', 'pseudodata')
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

    if path_to_weights is not None:
        experiment_log_dir = path_to_weights
    if plot_save_dir is None:
        plot_save_dir = experiment_log_dir

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    for net in [encoder, decoder]:
        net.to(device=ptu.device)

    if path_to_weights is None:
        ### Train encoder and decoder if necessary
        # Generate pseudodata and put it into stacked replay buffer
        print('Generating data')
        for i in range(pseudo_generator.num_tasks):
            episode = pseudo_generator.generate_episode(i, mode='overshoot_plus')
            episode_2 = pseudo_generator.generate_episode(i, mode='overshoot_minus')
            if variant['algo_params']['num_exploration_trajectories_per_task'] == 2:
                episode_list = ([episode, episode_2], 400)
                replay_buffer.add_episode_group(episode_list)
            else:
                episode_list = ([episode], 200)
                replay_buffer.add_episode_group(episode_list)
                episode_list = ([episode_2], 200)
                replay_buffer.add_episode_group(episode_list)

        replay_buffer.stats_dict = replay_buffer.get_stats()

        # call reconstructionTrainer on this stacked_replay_buffer, train until completion
        print('Reconstruction training')
        reconstruction_trainer.train(variant['algo_params']['num_reconstruction_steps'],
                                     plot_save_file=os.path.join(experiment_log_dir, 'reconstruction-training-curve.pdf'))
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        torch.save(encoder, os.path.join(experiment_log_dir, 'encoder.pth'))
        torch.save(decoder, os.path.join(experiment_log_dir, 'decoder.pth'))
    else:
        ### Otherwise just load them from disk
        encoder.load_state_dict(torch.load(os.path.join(path_to_weights, 'encoder.pth'), map_location=ptu.device))
        decoder.load_state_dict(torch.load(os.path.join(path_to_weights, 'decoder.pth'), map_location=ptu.device))


    # visualize results with existing functions (e.g. on "episode rollout")
    print('Plotting encoder')
    #z_min, z_max = plot_encoder(encoder, pseudo_generator, replay_buffer, variant['algo_params']['time_steps'],
    #                            variant['algo_params']['max_path_length'],
    #                            [-25, -15, -5, 0, 5, 15, 25],  # [-25, -12.5, 0, 12.5, 25],
    #                            save_dir=experiment_log_dir, save_name='latent-encodings-target-reached.pdf')
    z_min_2, z_max_2 = plot_encoder(encoder, pseudo_generator, replay_buffer, variant['algo_params']['time_steps'],
                                    variant['algo_params']['max_path_length'],
                                    [-25, -15, -5, 0, 5, 15, 25],  # [-25, -12.5, 0, 12.5, 25],
                                    modify_target=True,
                                    save_dir=plot_save_dir, save_name=f'{plot_save_prefix}latent-encodings.pdf')

    #z_min_2 = min(z_min, z_min_2)
    #z_max_2 = max(z_max, z_max_2)

    # visualize decoder capability showing the function learned for a particular z
    print('Plotting decoder')
    #plot_decoder(replay_buffer, decoder, z_min, z_max,
    #             save_dir=experiment_log_dir, save_name='decoder-functions-target-reached.pdf')
    plot_decoder(replay_buffer, decoder, z_min_2, z_max_2,
                 save_dir=plot_save_dir, save_name=f'{plot_save_prefix}decoder-functions.pdf')


@click.command()
@click.argument('plot_thesis', default=True)
@click.argument('train_linked', default=False)
@click.argument('train_separate', default=False)
@click.argument('plot_save_dir', default='../../../Thesis/experiments/pseudodata/')
def main(plot_thesis, train_linked, train_separate, plot_save_dir):
    if plot_save_dir == '':
        plot_save_dir = None
    if plot_thesis:
        train_and_plot("../configs/thesis/pseudodata-linked.json", "../output/pseudodata/2022_07_15_02_47_07",
                       plot_save_prefix='linked-', plot_save_dir=plot_save_dir)
        train_and_plot("../configs/thesis/pseudodata-separate.json", "../output/pseudodata/2022_07_15_02_55_20",
                       plot_save_prefix='separate-', plot_save_dir=plot_save_dir)
    if train_linked:
        train_and_plot("../configs/thesis/pseudodata-linked.json")
    if train_separate:
        train_and_plot("../configs/thesis/pseudodata-separate.json")


if __name__ == '__main__':
    main()

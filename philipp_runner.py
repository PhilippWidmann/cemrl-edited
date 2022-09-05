# Continuous Environment Meta Reinforcement Learning (CEMRL)

import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import click
import json
import torch
import gym

from cerml.exploration_agent import construct_exploration_agent

gym.logger.set_level(40)

from cerml.policy_networks import SingleSAC, MultipleSAC
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

from cerml.encoder_decoder_networks import PriorPz, Encoder, DecoderMDP
from cerml.sac import PolicyTrainer
from cerml.stacked_replay_buffer import StackedReplayBuffer
from cerml.reconstruction_trainer import ReconstructionTrainer, NoOpReconstructionTrainer
from cerml.rollout_worker import RolloutCoordinator
from cerml.agent import CEMRLAgent, ScriptedPolicyAgent
from cerml.relabeler import Relabeler
from cerml.cemrl_algorithm import CEMRLAlgorithm
from meta_rand_envs.wrappers import ENVS


def setup_environment(variant):
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    torch.set_num_threads(1)
    if variant['algo_params']['use_fixed_seeding']:
        torch.manual_seed(variant['algo_params']['seed'])
        np.random.seed(variant['algo_params']['seed'])

    # create logging directory
    encoding_save_epochs = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450,
                            475, 500, 600, 750, 1000, 1250, 1500, 1750, 2000, 3000, 5000, 10000, 15000, 20000]
    experiment_log_dir = setup_logger(variant['config_name'], variant=variant, exp_id=None,
                                      base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='specific',
                                      snapshot_gap=variant['algo_params']['snapshot_gap'],
                                      snapshot_points=encoding_save_epochs)

    # create temp folder
    if not os.path.exists(variant['util_params']['temp_dir']):
        os.makedirs(variant['util_params']['temp_dir'])

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    PLOT = variant['util_params']['plot']
    os.environ['DEBUG'] = str(int(DEBUG))
    os.environ['PLOT'] = str(int(PLOT))

    # create multi-task environment and sample tasks
    env = ENVS[variant['env_name']](**variant['env_params'])
    if variant['env_params']['use_normalized_env']:
        env = NormalizedBoxEnv(env)
    if variant['train_or_showcase'] == 'showcase':
        env = CameraWrapper(env)

    return env, experiment_log_dir


def initialize_networks(variant, env, experiment_log_dir):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    tasks = list(range(len(env.tasks)))
    train_tasks = list(range(len(env.train_tasks)))
    test_tasks = tasks[-variant['env_params']['n_eval_tasks']:]

    # instantiate networks
    net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    num_classes = variant['reconstruction_params']['num_classes']

    # set parameters if not given
    if variant['algo_params']['decoder_time_window'] is None:
        variant['algo_params']['decoder_time_window'] = [-time_steps, 0]

    # encoder used: single transitions or trajectories
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

    prior_pz = PriorPz(num_classes, latent_dim)

    if variant['algo_params']['policy_mode'] == 'sac_single':
        policy_networks = SingleSAC(
            obs_dim,
            latent_dim,
            action_dim,
            variant['algo_params']['sac_layer_size']
        )
    elif variant['algo_params']['policy_mode'] == 'sac_multiple':
        policy_networks = MultipleSAC(
            obs_dim,
            latent_dim,
            action_dim,
            variant['algo_params']['sac_layer_size'],
            variant['algo_params']['num_policy_nets']
        )
    else:
        raise ValueError(f"{variant['algo_params']['policy_mode']} is not a valid policy_mode")

    networks = {
        'encoder': encoder,
        'prior_pz': prior_pz,
        'decoder': decoder,
        **policy_networks.get_networks()
    }

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

    # Agent
    agent_class = ScriptedPolicyAgent if variant['env_params']['scripted_policy'] else CEMRLAgent
    agent = agent_class(
        encoder,
        prior_pz,
        policy_networks
    )
    if variant['algo_params']['exploration_agent'] is not None:
        exploration_agent = construct_exploration_agent(
            variant['algo_params']['exploration_agent'],
            policy_networks,
            replay_buffer,
            variant['env_name'],
            variant['env_params'],
            experiment_log_dir if variant['path_to_weights'] is None else variant['path_to_weights'],
            variant['algo_params']['max_path_length'],
            variant['algo_params']['exploration_pretraining_steps'],
            encoder.state_preprocessor,
            variant['algo_params']['exploration_ensemble_agents'],
            variant['showcase_itr']
        )
    else:
        exploration_agent = agent

    # Rollout Coordinator
    rollout_coordinator = RolloutCoordinator(
        env,
        variant['env_name'],
        variant['env_params'],
        variant['train_or_showcase'],
        agent,
        exploration_agent,
        replay_buffer,
        time_steps,

        variant['algo_params']['max_path_length'],
        variant['algo_params']['permute_samples'],
        variant['util_params']['use_multiprocessing'],
        variant['algo_params']['use_data_normalization'],
        variant['algo_params']['use_sac_data_normalization'],
        variant['util_params']['num_workers'],
        variant['util_params']['gpu_id'],
        variant['env_params']['scripted_policy']
    )

    # ReconstructionTrainer
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
    if variant['algo_params']['encoder_type'] == 'NoEncoder':
        # debug case: completely omit any encoding and only do SAC training
        reconstruction_trainer = NoOpReconstructionTrainer()

    # PolicyTrainer
    policy_trainer = PolicyTrainer(
        policy_networks,
        replay_buffer,
        encoder,
        variant['algo_params']['batch_size_policy'],
        action_dim,
        variant['algo_params']['data_usage_sac'],
        variant['algo_params']['use_data_normalization'],
        variant['algo_params']['use_sac_data_normalization'],
        variant['algo_params']['sac_uses_exploration_data'],
        policy_lr=variant['algo_params']['lr_policy'],
        qf_lr=variant['algo_params']['lr_qf'],
        use_parametrized_alpha=variant['algo_params']['use_parametrized_alpha'],
        target_entropy_factor=variant['algo_params']['target_entropy_factor'],
        alpha=variant['algo_params']['sac_alpha']
    )

    relabeler = Relabeler(
        encoder,
        replay_buffer,
        variant['algo_params']['batch_size_relabel'],
        action_dim,
        obs_dim,
        variant['algo_params']['use_data_normalization'],
    )

    algorithm = CEMRLAlgorithm(
        replay_buffer,
        rollout_coordinator,
        reconstruction_trainer,
        policy_trainer,
        relabeler,
        agent,
        exploration_agent,
        networks,

        train_tasks,
        test_tasks,

        variant['algo_params']['num_train_epochs'],
        variant['algo_params']['exploration_pretraining_epochs'],
        variant['algo_params']['exploration_pretraining_epoch_steps'],
        variant['algo_params']['exploration_epoch_steps'],
        variant['algo_params']['num_reconstruction_steps'],
        variant['algo_params']['num_policy_steps'],
        variant['algo_params']['num_train_tasks_per_episode'],
        variant['algo_params']['num_initial_collection_cycles_per_task'],
        variant['algo_params']['num_trajectories_per_task'],
        variant['algo_params']['num_exploration_trajectories_per_task'],
        variant['algo_params']['num_eval_trajectories'],
        variant['algo_params']['showcase_every'],
        variant['algo_params']['snapshot_gap'],
        variant['algo_params']['num_showcase_deterministic'],
        variant['algo_params']['num_showcase_non_deterministic'],
        variant['algo_params']['use_relabeler'],
        variant['algo_params']['exploration_agent'] is not None,
        variant['algo_params']['exploration_by_probability'],
        variant['algo_params']['exploration_fixed_probability'],
        experiment_log_dir,
        latent_dim,
    )

    return algorithm, networks, rollout_coordinator, replay_buffer, train_tasks, test_tasks


def load_networks(variant, networks, cemrl_compatibility):
    itr = variant['showcase_itr']
    path = variant['path_to_weights']
    for name, net in networks.items():
        state_dict = torch.load(os.path.join(path, name + '_itr_' + str(itr) + '.pth'), map_location='cpu')
        if cemrl_compatibility:
            if name == 'encoder':
                state_dict = OrderedDict((key.replace('shared_encoder', 'shared_encoder.layers'), val)
                                         for key, val in state_dict.items())
        net.load_state_dict(state_dict)


def experiment(variant):
    env, experiment_log_dir = setup_environment(variant)
    algorithm, networks, *_ = initialize_networks(variant, env, experiment_log_dir)

    if variant['path_to_weights'] is not None:
        load_networks(variant, networks)

    if ptu.gpu_enabled():
        algorithm.to()

    # run the algorithm
    algorithm.train()


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


@click.command()
@click.argument('config', default="configs/thesis/mw-reach-line.json")
@click.option('--weights', default=None)
@click.option('--weights_itr', default=None)
@click.option('--gpu', default=None, type=int)
@click.option('--num_workers', default=None, type=int)
@click.option('--use_mp', default=None, type=bool)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, weights, weights_itr, gpu, use_mp, num_workers, docker, debug):

    variant = default_config

    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    # Overwrite machine defaults only when explicitly given
    if gpu is not None:
        variant['util_params']['gpu_id'] = gpu
    if num_workers is not None:
        variant['util_params']['num_workers'] = num_workers
    if use_mp is not None:
        variant['util_params']['use_multiprocessing'] = use_mp
    if weights is not None:
        variant['path_to_weights'] = weights
    if weights_itr is not None:
        variant['showcase_itr'] = weights_itr

    variant['config_name'] = Path(config).stem

    experiment(variant)


if __name__ == "__main__":
    main()

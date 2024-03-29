# Continuous Embedding Meta Reinforcement Learning (CEMRL)

import os, shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import click
import json

import rlkit.torch.pytorch_util as ptu

import pickle

from analysis.progress_logger import manage_logging
from analysis.plot_episode import plot_per_episode, get_plot_specification, plot_episode_encodings
import configs.legacy_default
import configs.analysis_config

from philipp_runner import setup_environment, initialize_networks, load_networks

#plt.rcParams.update({'font.size': 20})


def analysis(variant, cemrl_compatibility=False):
    if isinstance(variant['showcase_itr'], list):
        l = variant['showcase_itr'].copy()
        for itr in l:
            variant['showcase_itr'] = itr
            try:
                res = analysis(variant, cemrl_compatibility)
            except FileNotFoundError:
                warnings.warn(f'Stopped before iteration {itr} because the corresponding savefiles do not exist')
                exit(1)
        # Return last result from iteration list; no usecase currently
        return res

    all_figures = []
    # Prepare and load networks, just like for an actual run
    env, experiment_log_dir = setup_environment(variant)
    algorithm, networks, rollout_coordinator, replay_buffer, train_tasks, test_tasks = \
        initialize_networks(variant, env, experiment_log_dir)

    if variant['path_to_weights'] is not None:
        load_networks(variant, networks, cemrl_compatibility)

    if ptu.gpu_enabled():
        algorithm.to()

    # showcase learned policy loaded
    showcase_itr = variant['showcase_itr']
    path_to_folder = variant['path_to_weights']
    example_cases = variant['analysis_params']['example_cases']
    train_example_cases = variant['analysis_params']['train_example_cases']
    exploration_cases = list(range(variant['analysis_params']['num_exploration_cases']))#train_example_cases if variant['analysis_params']['include_exploration_examples'] else []

    save = variant['analysis_params']['save']
    show = variant['analysis_params']['show']
    figsize = variant['analysis_params']['figsize'] if 'figsize' in variant['analysis_params'].keys() else (6, 5)
    if save:
        save_dir = variant['save_dir'] if variant['save_dir'] is not None else os.path.join(variant['path_to_weights'], 'analysis/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        variant['save_prefix'] = variant['save_prefix'] + '_' if variant['save_prefix'] != '' else ''
    replay_buffer.stats_dict = pickle.load(open(os.path.join(path_to_folder, "replay_buffer_stats_dict_" + str(showcase_itr) + ".p"), "rb"))
    env.reset_task(np.random.randint(len(env.test_tasks)) + len(env.train_tasks))
    env.set_meta_mode('test')

    # log file and plot progress file
    if variant['analysis_params']['log_and_plot_progress']:
        manage_logging(path_to_folder, save=save)

    cases_dict = {'train': train_example_cases,
                  'test': example_cases,
                  'exploration': exploration_cases}
    results_dict = {'train': {}, 'test': {}, 'exploration': {}}
    for example_case in example_cases:
        results = rollout_coordinator.collect_data(test_tasks[example_case:example_case + 1], 'test',
                deterministic=True, max_trajs=1, animated=variant['analysis_params']['visualize_run'], save_frames=False, return_distributions=True)
        results_dict['test'][example_case] = results[0][0][0][0]
    for train_example_case in train_example_cases:
        results = rollout_coordinator.collect_data(train_tasks[train_example_case:train_example_case + 1], 'test',
                deterministic=True, max_trajs=1, animated=variant['analysis_params']['visualize_run'], save_frames=False, return_distributions=True)
        results_dict['train'][train_example_case] = results[0][0][0][0]
    for exploration_example_case in exploration_cases:
        results = rollout_coordinator.collect_data(train_tasks[exploration_example_case:exploration_example_case + 1], 'test',
                deterministic=True, max_trajs=0, max_trajs_exploration=1, animated=variant['analysis_params']['visualize_run'], save_frames=False, return_distributions=True,
                compute_exploration_task_indicators=True)
        results_dict['exploration'][exploration_example_case] = results[0][0][0][0]

    # plot encodings
    if variant['analysis_params']['plot_encoding']:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=False)
        plot_episode_encodings(results_dict['test'], fig_ax=(fig, ax[0][0]), color=variant['analysis_params']['color'])
        all_figures.append((fig, ax))

    # Do separate plots for train and test cases
    for type in results_dict.keys():
        cases = cases_dict[type]
        if not cases:
            continue
        for plot_spec in variant['analysis_params']['single_episode_plots']:
            plot_spec_dict = get_plot_specification(plot_spec)
            for case in cases:
                fig, axes = plt.subplots(nrows=len(plot_spec_dict), ncols=1, figsize=(figsize[0], figsize[1]*len(plot_spec_dict)), squeeze=False)
                for i, ax in enumerate(axes.flat):
                    p = plot_spec_dict[i]
                    fig, ax = plot_per_episode(results_dict[type][case], p['y'], p['scatter'], p['const'], p['fill'], p['x'], fig_ax=(fig, ax))

                fig.tight_layout()
                all_figures.append((fig, axes))
                if save:
                    save_name = variant['save_prefix'] + type + 'task_' + \
                                'itr-' + str(showcase_itr) + '_' + \
                                'case-' + str(case) + '_' + \
                                str(plot_spec) + '.png'
                    fig.tight_layout()
                    fig.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')
                if show:
                    fig.show()

        for plot_spec in variant['analysis_params']['multiple_episode_plots']:
            plot_spec_dict = get_plot_specification(plot_spec)
            fig, axes = plt.subplots(nrows=len(plot_spec_dict), ncols=1, figsize=(figsize[0], figsize[1]*len(plot_spec_dict)), squeeze=False)
            for case in cases:
                for i, ax in enumerate(axes.flat):
                    p = plot_spec_dict[i]
                    fig, ax = plot_per_episode(results_dict[type][case], p['y'], p['scatter'], p['const'], p['fill'], p['x'], fig_ax=(fig, ax))

            fig.tight_layout()
            all_figures.append((fig, axes))
            casename = str(cases[-1]) if type == 'exploration' else str(cases)
            if save:
                save_name = variant['save_prefix'] + type + 'task_' + \
                            'itr-' + str(showcase_itr) + '_' + \
                            'case-' + casename + '_' + \
                            str(plot_spec) + '.png'
                fig.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches='tight')
            if show:
                fig.show()

    return all_figures

# Code snippet for producing videos.
# Currently not functional, but should (might?) be able to be adapted/integrated, so  keeping here for later use
"""
    # visualize test cases
    for example_case in example_cases:
        # video taking
        if variant['analysis_params']['produce_video']:
            print("Producing video... do NOT kill program until completion!")
            results = rollout_coordinator.collect_data(test_tasks[example_case:example_case + 1], 'test', deterministic=True, max_trajs=1, animated=False, save_frames=True)
            #if max([f['success'] for f in results[0][0][0][0]['env_infos']]) > 0.5:
            #    print('Success')
            #else:
            #    print('Failure')
            path_video = results[0][0][0][0]
            video_frames = []
            video_frames += [t['frame'] for t in path_video['env_infos']]
            print("Saving video...")
            # save frames to file temporarily
            temp_dir = os.path.join(path_to_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            for i, frm in enumerate(video_frames):
                frm.save(os.path.join(temp_dir, '%06d.jpg' % i))
            video_filename = os.path.join(save_dir,
                                          variant['save_prefix'] + variant['env_name'] + '_testcase' + '_' + str(example_case) +
                                          '_' + 'itr-' + str(showcase_itr) + '_video.mp4')
            # run ffmpeg to make the video
            os.system('ffmpeg -r 25 -i {}/%06d.jpg -vb 20M -vcodec mpeg4 {}'.format(temp_dir, video_filename))
            # delete the frames
            shutil.rmtree(temp_dir)
    for train_example_case in variant['analysis_params']['train_example_cases']:
        if variant['analysis_params']['produce_video']:
            print("Producing video... do NOT kill program until completion!")
            results = rollout_coordinator.collect_data(train_tasks[train_example_case:train_example_case + 1], 'test', deterministic=True, max_trajs=1, animated=False, save_frames=True)
            #if max([f['success'] for f in results[0][0][0][0]['env_infos']]) > 0.5:
            #    print('Success')
            #else:
            #    print('Failure')
            path_video = results[0][0][0][0]
            video_frames = []
            video_frames += [t['frame'] for t in path_video['env_infos']]
            print("Saving video...")
            # save frames to file temporarily
            temp_dir = os.path.join(path_to_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            for i, frm in enumerate(video_frames):
                frm.save(os.path.join(temp_dir, '%06d.jpg' % i))
            video_filename = os.path.join(save_dir,
                                          variant['save_prefix'] + variant['env_name'] + '_traincase' + '_' + str(train_example_case) +
                                          '_' + 'itr-' + str(showcase_itr) + '_video.mp4')
            # run ffmpeg to make the video
            os.system('ffmpeg -r 25 -i {}/%06d.jpg -vb 20M -vcodec mpeg4 {}'.format(temp_dir, video_filename))
            # delete the frames
            shutil.rmtree(temp_dir)
        """


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            if k not in to.keys():
                to[k] = dict()
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def prepare_variant_file(analysis_goal_config):
    variant = deep_update_dict(analysis_goal_config, configs.legacy_default.legacy_default_config)

    path_to_folder = variant['path_to_weights']
    with open(os.path.join(os.path.join(path_to_folder, 'variant.json'))) as f:
        exp_params = json.load(f)
    variant["env_name"] = exp_params["env_name"]
    if "config_name" in exp_params.keys():
        variant["config_name"] = exp_params["config_name"]
    else:
        variant["config_name"] = exp_params["env_name"]
    variant["env_params"] = deep_update_dict(exp_params["env_params"], variant["env_params"])
    variant["algo_params"] = deep_update_dict(exp_params["algo_params"], variant["algo_params"])
    variant["reconstruction_params"] = deep_update_dict(exp_params["reconstruction_params"], variant["reconstruction_params"])

    # set other time steps than while training
    if "manipulate_time_steps" in variant['analysis_params'] and variant["analysis_params"]["manipulate_time_steps"]:
        variant["algo_params"]["time_steps"] = variant["analysis_params"]["time_steps"]

    # set other time steps than while training
    if "manipulate_change_trigger" in variant['analysis_params'] and variant["analysis_params"]["manipulate_change_trigger"]:
        variant["env_params"] = deep_update_dict(variant["analysis_params"]["change_params"], variant["env_params"])

    # set other episode length than while training
    if "manipulate_max_path_length" in variant['analysis_params'] and variant["analysis_params"]["manipulate_max_path_length"]:
        variant["algo_params"]["max_path_length"] = variant["analysis_params"]["max_path_length"]

    # set other task number than while training
    if "manipulate_test_task_number" in variant['analysis_params'] and variant["analysis_params"]["manipulate_test_task_number"]:
        variant["env_params"]["n_eval_tasks"] = variant["analysis_params"]["test_task_number"]

    return variant


@click.command()
@click.option('--weights', default=None)
@click.option('--weights_itr', default=None)
@click.option('--gpu', default=0)
@click.option('--num_workers', default=8)
@click.option('--use_mp', is_flag=True, default=False)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
@click.option('--cemrl_compatibility', is_flag=True, default=False)
def main(weights, weights_itr, gpu, use_mp, num_workers, docker, debug, cemrl_compatibility):
    variant = prepare_variant_file(configs.analysis_config.analysis_config)

    if weights is not None:
        variant['path_to_weights'] = weights
    if weights_itr is not None:
        variant['showcase_itr'] = weights_itr

    variant['util_params']['gpu_id'] = gpu
    variant['util_params']['use_multiprocessing'] = use_mp
    variant['util_params']['num_workers'] = num_workers

    if cemrl_compatibility:
        variant['algo_params']['encoder_type'] = 'TimestepMLP'
        variant['env_params']['use_state_decoder'] = True

    analysis(variant, cemrl_compatibility)


if __name__ == "__main__":
    main()

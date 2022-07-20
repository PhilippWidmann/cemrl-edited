import copy
import os
import warnings

import click
from analysis_runner import analysis, prepare_variant_file
from thesis_plot_progress import DEFAULTS

DEFAULT_CONFIG = dict(
    showcase_itr=500,
    util_params=dict(
        base_log_dir='.temp',  # name of output directory
        use_gpu=True,  # set True if GPU available and should be used
        use_multiprocessing=False,  # set True if data collection should be parallelized across CPUs
        num_workers=8,  # number of CPU workers for data collection
        gpu_id=0,  # number of GPU if machine with multiple GPUs
        debug=False,  # debugging triggers printing and writes logs to debug directory
        plot=False  # plot figures of progress for reconstruction and policy training
    ),
    analysis_params=dict(
        figsize=DEFAULTS['figsize'],
        save=False,
        show=False,
        log_and_plot_progress=False,
        visualize_run=False,
        plot_encoding=False,
        num_exploration_cases=0,
        example_cases=[],
        train_example_cases=[],
        single_episode_plots=[],
        multiple_episode_plots=[],
    ),
    plot_params=dict(
        title=None,
        x_label='Time step $t$',
        y_label='Position',
        xlim=(None, None),
        ylim=(None, None),
    )
)

CONFIGS = {
    "cheetah-goal-exploration-run-1": dict(
        path_to_weights='../output/cheetah-goal/2022_07_09_10_24_13 (Kopie)',
        save_names='cheetah-goal/exploration-run-1',
        multiple_episode_plots=[['time_vs_pos']],
        num_exploration_cases=10,
        ylim=(-40, 40),
        figsize=(3.84, 2.88),
    ),
    "cheetah-goal-exploration-run-2": dict(
        path_to_weights='../output/cheetah-goal/2022_07_08_10_44_48 (Kopie)',
        save_names='cheetah-goal/exploration-run-2',
        multiple_episode_plots=[['time_vs_pos']],
        num_exploration_cases=10,
        ylim=(-40, 40),
        figsize=(3.84, 2.88),
    ),
    "cheetah-goal-exploration-run-3": dict(
        path_to_weights='../output/cheetah-goal/2022_07_10_07_50_51 (Kopie)',
        save_names='cheetah-goal/exploration-run-3',
        multiple_episode_plots=[['time_vs_pos']],
        num_exploration_cases=10,
        ylim=(-40, 40),
        figsize=(3.84, 2.88),
    ),
    "cheetah-goal-ablation-default": dict(
        path_to_weights='../output/cheetah-goal/2022_07_09_10_24_13 (Kopie)',
        save_names=['cheetah-goal-ablation/default-encodings', 'cheetah-goal-ablation/default-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=(3.84, 2.88),
    ),
    "cheetah-goal-ablation-noLink": dict(
        path_to_weights='../output/cheetah-goal-separateEpisodes/2022_07_10_10_00_00',
        save_names=['cheetah-goal-ablation/noLink-encodings', 'cheetah-goal-ablation/noLink-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=(3.84, 2.88),
    ),
    "cheetah-goal-ablation-noExploration": dict(
        path_to_weights='../output/cheetah-goal-noExploration/2022_07_11_22_17_47',
        save_names=['cheetah-goal-ablation/noExploration-encodings', 'cheetah-goal-ablation/noExploration-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=(3.84, 2.88),
    ),
    "cheetah-goal-halfline-step": dict(
        path_to_weights='../output/cheetah-halfline-goal-step-2/2022_07_14_13_37_23',
        save_names=['cheetah-goal-halfline/step-encodings', 'cheetah-goal-halfline/step-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=(3.84, 2.88),
        showcase_itr=300,
    ),
    "cheetah-goal-halfline-past": dict(
        path_to_weights='../output/cheetah-halfline-goal-past-2/2022_07_14_01_53_26',
        save_names=['cheetah-goal-halfline/past-encodings', 'cheetah-goal-halfline/past-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=(3.84, 2.88),
        showcase_itr=300,
    ),
    "cheetah-goal-halfline-past-future": dict(
        path_to_weights='../output/cheetah-halfline-goal-past-and-future-2/2022_07_14_09_20_26',
        save_names=['cheetah-goal-halfline/past-future-encodings', 'cheetah-goal-halfline/past-future-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=(3.84, 2.88),
        showcase_itr=250,
    ),
    "cheetah-goal-halfline-full-episode": dict(
        path_to_weights='../output/cheetah-halfline-goal-fullEp-2/2022_07_14_11_28_22',
        save_names=['cheetah-goal-halfline/full-episode-encodings', 'cheetah-goal-halfline/full-episode-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=(3.84, 2.88),
        showcase_itr=300,
    ),
    "toy-goal-2D-exploration": dict(
        path_to_weights='../output/toy-goal-2D/2022_07_12_17_30_36',
        save_names=['toy-goal-plane/exploration-trajectories',],
        multiple_episode_plots=[['pos[0]_vs_pos[1]'],],
        x_label=['position $x_1$'],
        y_label=['position $x_2$'],
        num_exploration_cases=50,
        #example_cases=[0, 5, 10, 15, 20],
        figsize=(5.76, 5.2),
        showcase_itr=450,
        xlim=[-50, 50],
        ylim=[-50, 50]
    ),
    "toy-goal-2D-policy": dict(
        path_to_weights='../output/toy-goal-2D/2022_07_12_17_30_36 (Kopie)',
        save_names=['toy-goal-plane/policy-trajectories',],
        multiple_episode_plots=[['pos[0]_vs_pos[1]_scatter_specification[0]_vs_specification[1]'],],
        x_label=['position $x_1$', '$z_1$'],
        y_label=['position $x_2$', '$z_2$'],
        #num_exploration_cases=50,
        example_cases=[0, 5, 10, 15, 20, 25, 30, 35],
        figsize=(5.76, 5.2),
        showcase_itr=450,
        xlim=(-19, 19),
        ylim=(-19, 19),
    ),
    "toy-goal-2D-policy-encoding": dict(
        path_to_weights='../output/toy-goal-2D/2022_07_12_17_30_36 (Kopie)',
        save_names=['toy-goal-plane/policy-trajectories-encodings',],
        multiple_episode_plots=[['task_indicators[0]_vs_task_indicators[1]'],],
        x_label=['position $x_1$', '$z_1$'],
        y_label=['position $x_2$', '$z_2$'],
        #num_exploration_cases=50,
        example_cases=[0, 5, 10, 15, 20, 25, 30, 35],
        figsize=(5.76, 5.2),
        showcase_itr=450,
    ),
    "toy-goal-2D-policy-train": dict(
        path_to_weights='../output/toy-goal-2D/2022_07_12_17_30_36 (Kopie)',
        save_names=['toy-goal-plane/train-policy-trajectories',
                    'toy-goal-plane/train-policy-trajectories-encodings',],
        multiple_episode_plots=[['pos[0]_vs_pos[1]_scatter_specification[0]_vs_specification[1]'],
                                ['task_indicators[0]_vs_task_indicators[1]'],],
        x_label=['position $x_1$', '$z_1$'],
        y_label=['position $x_2$', '$z_2$'],
        #num_exploration_cases=50,
        train_example_cases=[0, 5, 10, 15, 20, 25, 30, 35],
        figsize=(5.76, 5.2),
        showcase_itr=450,
    )
}


def get_plot_param(param_name, index, config):
    param = DEFAULT_CONFIG['plot_params'][param_name] if param_name not in config.keys() else config[param_name]
    if isinstance(param, list):
        return param[index]
    else:
        if param_name == 'save_names' and index > 0:
            warnings.warn('Using the same save_name for multiple plots. Something will be overwritten.')
        return param


def copy_analysis_param(param, variant, config):
    if param in config.keys():
        variant['analysis_params'][param] = config[param]


@click.command()
@click.option('--save_dir', default="../../../Thesis/experiments/")
def main(save_dir):
    config_names = ("cheetah-goal-halfline-past-future",)
    if config_names is None:
        config_names = CONFIGS.keys()
    for config_name in config_names:
        config = CONFIGS[config_name]
        # Make default config copy and update it with path of chosen config
        variant = copy.deepcopy(DEFAULT_CONFIG)
        variant['path_to_weights'] = config['path_to_weights']
        variant = prepare_variant_file(variant)
        # Copy parameters from specific config
        if 'showcase_itr' in config.keys():
            variant['showcase_itr'] = config['showcase_itr']
        copy_analysis_param('num_exploration_cases', variant, config)
        copy_analysis_param('example_cases', variant, config)
        copy_analysis_param('train_example_cases', variant, config)
        copy_analysis_param('single_episode_plots', variant, config)
        copy_analysis_param('multiple_episode_plots', variant, config)
        copy_analysis_param('figsize', variant, config)

        # Make figure
        fig_ax_list = analysis(variant)
        for i, (fig, ax) in enumerate(fig_ax_list):
            ax = ax[0, 0]
            ax.set_title(get_plot_param('title', i, config))
            ax.set_xlabel(get_plot_param('x_label', i, config))
            ax.set_ylabel(get_plot_param('y_label', i, config))
            ax.set_xlim(get_plot_param('xlim', i, config))
            ax.set_ylim(get_plot_param('ylim', i, config))
            fig.tight_layout()
            file_path = os.path.join(save_dir, f'{get_plot_param("save_names", i, config)}.pdf')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fig.savefig(file_path, dpi=300)
            fig.show()

if __name__ == '__main__':
    main()
import copy
import os
import warnings
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from analysis_runner import analysis, prepare_variant_file
from thesis_plot_progress import DEFAULTS, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, COLOR_CYCLER, \
    FIGSIZE_FULL, FIGSIZE_HALF, FIGSIZE_THIRD, FIGSIZE_HALF_SQUARE

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['axes.prop_cycle'] = COLOR_CYCLER['default']

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
        xticks=None,
    )
)

CONFIGS = {
    "cheetah-goal-exploration-run-1": dict(
        path_to_weights='../output/cheetah-goal/2022_07_09_10_24_13',
        save_names='cheetah-goal/exploration-run-1',
        multiple_episode_plots=[['time_vs_pos']],
        num_exploration_cases=10,
        ylim=(-40, 40),
        figsize=FIGSIZE_THIRD,
    ),
    "cheetah-goal-exploration-run-2": dict(
        path_to_weights="../output/cheetah-goal/2022_07_30_20_41_18",
        save_names='cheetah-goal/exploration-run-2',
        multiple_episode_plots=[['time_vs_pos']],
        num_exploration_cases=10,
        ylim=(-40, 40),
        figsize=FIGSIZE_THIRD,
    ),
    "cheetah-goal-exploration-run-3": dict(
        path_to_weights='../output/cheetah-goal/2022_07_10_07_50_51',
        save_names='cheetah-goal/exploration-run-3',
        multiple_episode_plots=[['time_vs_pos']],
        num_exploration_cases=10,
        ylim=(-40, 40),
        figsize=FIGSIZE_THIRD,
    ),
    "cheetah-goal-exploration-run-bad": dict(
        path_to_weights='../output/cheetah-goal/2022_07_08_10_44_48',
        save_names='cheetah-goal/exploration-run-bad',
        multiple_episode_plots=[['time_vs_pos']],
        num_exploration_cases=10,
        ylim=(-40, 40),
        figsize=FIGSIZE_THIRD,
    ),
    "cheetah-goal-ablation-default": dict(
        path_to_weights='../output/cheetah-goal/2022_07_09_10_24_13',
        save_names=['cheetah-goal-ablation/default-encodings', 'cheetah-goal-ablation/default-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=FIGSIZE_THIRD,
    ),
    "cheetah-goal-ablation-noLink": dict(
        path_to_weights='../output/cheetah-goal-separateEpisodes/2022_07_10_10_00_00',
        save_names=['cheetah-goal-ablation/noLink-encodings', 'cheetah-goal-ablation/noLink-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=FIGSIZE_THIRD,
    ),
    "cheetah-goal-ablation-noExploration": dict(
        path_to_weights='../output/cheetah-goal-noExploration/2022_07_11_22_17_47',
        save_names=['cheetah-goal-ablation/noExploration-encodings', 'cheetah-goal-ablation/noExploration-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=FIGSIZE_THIRD,
    ),
    "cheetah-goal-halfline-step": dict(
        path_to_weights="../output/cheetah-halfline-goal-step/2022_07_22_14_18_48",
                        # "../output/cheetah-halfline-goal-step/2022_07_14_13_37_23",  # worst
                        # "../output/cheetah-halfline-goal-step/2022_07_22_14_18_48",  # best
                        # "../output/cheetah-halfline-goal-step/2022_07_23_03_38_01",  # mid
        save_names=['cheetah-goal-halfline/step-encodings', 'cheetah-goal-halfline/step-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=FIGSIZE_THIRD,
        showcase_itr=300,
    ),
    "cheetah-goal-halfline-past": dict(
        path_to_weights="../output/cheetah-halfline-goal-past/2022_07_14_01_53_26",
                        # "../output/cheetah-halfline-goal-past/2022_07_14_01_53_26",  # best
                        # "../output/cheetah-halfline-goal-past/2022_07_23_20_10_38",  # worst
                        # "../output/cheetah-halfline-goal-past/2022_07_24_10_40_36",  # mid
        save_names=['cheetah-goal-halfline/past-encodings', 'cheetah-goal-halfline/past-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=FIGSIZE_THIRD,
        showcase_itr=300,
    ),
    "cheetah-goal-halfline-past-future": dict(
        path_to_weights="../output/cheetah-halfline-goal-past-and-future/2022_07_22_14_19_26",
                        # "../output/cheetah-halfline-goal-past-and-future/2022_07_14_09_20_26",  # mid
                        # "../output/cheetah-halfline-goal-past-and-future/2022_07_22_14_19_26",  # best
                        # "../output/cheetah-halfline-goal-past-and-future/2022_07_23_05_09_26",  # worst
        save_names=['cheetah-goal-halfline/past-future-encodings', 'cheetah-goal-halfline/past-future-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=FIGSIZE_THIRD,
        showcase_itr=250,
    ),
    "cheetah-goal-halfline-full-episode": dict(
        path_to_weights="../output/cheetah-halfline-goal-fullEp/2022_07_23_16_56_12",
                        # "../output/cheetah-halfline-goal-fullEp/2022_07_14_11_28_22",  # mid
                        # "../output/cheetah-halfline-goal-fullEp/2022_07_23_16_56_12",  # best
                        # "../output/cheetah-halfline-goal-fullEp/2022_07_24_10_26_45",  # worst
        save_names=['cheetah-goal-halfline/full-episode-encodings', 'cheetah-goal-halfline/full-episode-pos'],
        multiple_episode_plots=[['time_vs_z_means_fill_time_vs_z_stds'], ['time_vs_pos_const_time_vs_specification']],
        y_label=['$z$', 'Position'],
        example_cases=[0, 5, 10, 15, 20],
        #figsize=FIGSIZE_THIRD,
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
        figsize=FIGSIZE_HALF_SQUARE,
        showcase_itr=450,
        xlim=[-50, 50],
        ylim=[-50, 50]
    ),
    "toy-goal-2D-policy": dict(
        path_to_weights='../output/toy-goal-2D/2022_07_12_17_30_36',
        save_names=['toy-goal-plane/policy-trajectories',],
        multiple_episode_plots=[['pos[0]_vs_pos[1]_scatter_specification[0]_vs_specification[1]'],],
        x_label=['position $x_1$', '$z_1$'],
        y_label=['position $x_2$', '$z_2$'],
        #num_exploration_cases=50,
        example_cases=[0, 5, 10, 15, 20, 25, 30, 35],
        figsize=FIGSIZE_HALF_SQUARE,
        showcase_itr=450,
        xlim=(-19, 19),
        ylim=(-19, 19),
    ),
    "toy-goal-2D-policy-encoding": dict(
        path_to_weights='../output/toy-goal-2D/2022_07_12_17_30_36',
        save_names=['toy-goal-plane/policy-trajectories-encodings',],
        multiple_episode_plots=[['task_indicators[0]_vs_task_indicators[1]'],],
        x_label=['position $x_1$', '$z_1$'],
        y_label=['position $x_2$', '$z_2$'],
        #num_exploration_cases=50,
        example_cases=[0, 5, 10, 15, 20, 25, 30, 35],
        figsize=FIGSIZE_HALF_SQUARE,
        showcase_itr=450,
    ),
    "toy-goal-2D-policy-train": dict(
        path_to_weights='../output/toy-goal-2D/2022_07_12_17_30_36',
        save_names=['toy-goal-plane/train-policy-trajectories',
                    'toy-goal-plane/train-policy-trajectories-encodings',],
        multiple_episode_plots=[['pos[0]_vs_pos[1]_scatter_specification[0]_vs_specification[1]'],
                                ['task_indicators[0]_vs_task_indicators[1]'],],
        x_label=['position $x_1$', '$z_1$'],
        y_label=['position $x_2$', '$z_2$'],
        #num_exploration_cases=50,
        train_example_cases=[0, 5, 10, 15, 20, 25, 30, 35],
        figsize=FIGSIZE_HALF_SQUARE,
        showcase_itr=450,
    ),
    "toy-goal-boxplot-ours": dict(
        path_to_weights='../output/toy-goal-line/2022_07_06_06_02_34',
        save_names='toy-goal-line/encoding-boxplot-ours',
        example_cases=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        plot_encoding=True,
        xlim=(-28, 28),
        xticks=np.arange(-25, 25.1, 12.5),
        figsize=FIGSIZE_HALF,
        showcase_itr=100,
        x_label=['Goal'],
        y_label=['$z$'],
        color=COLOR_CYCLER['default'].by_key()['color'][0]
    ),
    "toy-goal-boxplot-cemrl": dict(
        cemrl_compatibility=True,
        path_to_weights='../../cemrl/output/toy-goal-line/2022_07_15_03_06_51',
        save_names='toy-goal-line/encoding-boxplot-cemrl',
        example_cases=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        plot_encoding=True,
        xlim=(-28, 28),
        xticks=np.arange(-25, 25.1, 12.5),
        figsize=FIGSIZE_HALF,
        showcase_itr=100,
        x_label=['Goal'],
        y_label=['$z$'],
        color=COLOR_CYCLER['default'].by_key()['color'][1]
    ),
    "toy-goal-episode-encodings-ours": dict(
        path_to_weights='../output/toy-goal-line/2022_07_06_06_02_34',
        save_names=['toy-goal-line/positions-episode-ours'],
        multiple_episode_plots=[['time_vs_pos_const_time_vs_specification']],
        y_label=['Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=FIGSIZE_HALF,
        showcase_itr=100,
    ),
    "toy-goal-episode-encodings-cemrl": dict(
        cemrl_compatibility=True,
        path_to_weights='../../cemrl/output/toy-goal-line/2022_07_15_03_06_51',
        save_names=['toy-goal-line/positions-episode-cemrl'],
        multiple_episode_plots=[['time_vs_pos_const_time_vs_specification']],
        y_label=['Position'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=FIGSIZE_HALF,
        showcase_itr=100,
    ),
    "toy-goal-episode-rewards-cemrl": dict(
        cemrl_compatibility=True,
        path_to_weights='../../cemrl/output/toy-goal-line/2022_07_15_03_06_51',
        save_names=['thesis-defense/toy-goal-line-rewards-cemrl'],
        multiple_episode_plots=[['time_vs_rewards']],
        y_label=['Reward $r$'],
        example_cases=[0, 4, 8, 12, 16, 20, 24],
        figsize=FIGSIZE_HALF,
        showcase_itr=100,
    ),
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
    config_names = ("toy-goal-episode-rewards-cemrl", )
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
        copy_analysis_param('plot_encoding', variant, config)
        copy_analysis_param('figsize', variant, config)
        copy_analysis_param('color', variant, config)

        cemrl_compatibility = config['cemrl_compatibility'] if 'cemrl_compatibility' in config.keys() else False
        if cemrl_compatibility:
            variant['algo_params']['encoder_type'] = 'TimestepMLP'
            variant['env_params']['use_state_decoder'] = True

        # Make figure
        fig_ax_list = analysis(variant, cemrl_compatibility=cemrl_compatibility)
        for i, (fig, ax) in enumerate(fig_ax_list):
            ax = ax[0, 0]
            ax.set_title(get_plot_param('title', i, config))
            ax.set_xlabel(get_plot_param('x_label', i, config))
            ax.set_ylabel(get_plot_param('y_label', i, config))
            if get_plot_param('xticks', i, config) is not None:
                ax.set_xticks(get_plot_param('xticks', i, config))
                ax.set_xticklabels(get_plot_param('xticks', i, config))
            ax.set_xlim(get_plot_param('xlim', i, config))
            ax.set_ylim(get_plot_param('ylim', i, config))
            fig.tight_layout()
            file_path = os.path.join(save_dir, f'{get_plot_param("save_names", i, config)}.pdf')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fig.savefig(file_path, dpi=300)
            fig.show()

if __name__ == '__main__':
    main()
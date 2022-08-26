import os.path

import click
from cycler import cycler
import json
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
# Default sizes:
# Legend, axis labels and ticks: 10
# Axis title (i.e. text above figure): 12

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes; unused
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-3)    # legend fontsize; corrected because it visually appears larger
plt.rc('legend', title_fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title; unused

FIGSIZE_FULL = (12.4, 0.75*12.4)
FIGSIZE_HALF = (6.08, 0.75*6.08)
FIGSIZE_HALF_SQUARE = (6.08, 0.9*6.08)
FIGSIZE_THIRD = (3.96, 0.75*3.96)

DEFAULTS = {
    'figsize': FIGSIZE_HALF,  # (5.76, 4.32)=0.9* (6.4, 4.8); (3.84, 2.88)=0.6*(6.4, 4.8)
    'save_dir': '../test',
    'xlim': [None, None],
    'ylim': [None, None],
    'x_scale': 'log',
    'x_label': None,
    'y_label': None,
    'xticks': None,
    'title': None,
    'legend_config': None,
    'color_cycler': 'default',
    'line_cycler': 'default',
}
# Original colorblind palette: "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
COLOR_CYCLER = {
    #'default': cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    'default': cycler('color', ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#E69F00", "#56B4E9", "#CC79A7", "#000000", "#E69F00", "#56B4E9"]),
    'blue': cycler('color', ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']),
    #'variant': cycler('color', ['#1f77b4', '#d62728', '#e377c2', '#8c564b', '#9467bd', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e', '#2ca02c']),
    'variant': cycler('color', ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00", "#009E73", "#F0E442", "#000000", "#E69F00", "#56B4E9"]),
    'thesis_defense': cycler('color', ["#D55E00", "#009E73", "#F0E442", "#E69F00", "#56B4E9", "#CC79A7", "#000000", "#E69F00", "#56B4E9", "#0072B2"]),
}

LINE_CYCLER = {
    'default': cycler(linestyle=['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']),
    'diff_lines': cycler(linestyle=['-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5)), '-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5))])
}


CONFIGS = {
    'cheetah-goal/cheetah-goal': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'cheetah-goal',
        'ylim': [-2800, -500],
        'xlim': [300000, 7000000],
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/cheetah-goal/2022_07_09_10_24_13",
                         "../output/cheetah-goal/2022_07_30_20_41_18",
                         #"../output/cheetah-goal/2022_07_08_10_44_48",  # This is the bad run
                         "../output/cheetah-goal/2022_07_10_07_50_51"),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': 'CEMRL',
                'dirs': ("../../cemrl/output/cheetah-stationary-targetTwosided/2022_08_02_22_42_35",
                         "../../cemrl/output/cheetah-stationary-targetTwosided/2022_08_04_17_47_02",
                         "../../cemrl/output/cheetah-stationary-targetTwosided/2022_08_05_10_04_53",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
            },
            {
                'name': 'PEARL',
                'dirs': ("../../pearl/output/cheetah-stationary-targetTwosided/2022_07_11_21_50_01",
                         "../../pearl/output/cheetah-stationary-targetTwosided/2022_07_31_14_17_17",
                         "../../pearl/output/cheetah-stationary-targetTwosided/2022_08_01_00_23_19",),
                'x': 'Number of env steps total',
                'y': 'AverageReturn_all_test_tasks',
            },
        )
    },
    'cheetah-goal/cheetah-goal-success': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average success rate',
        'title': 'cheetah-goal',
        'xlim': [300000, 7000000],
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/cheetah-goal/2022_07_09_10_24_13",
                         "../output/cheetah-goal/2022_07_30_20_41_18",
                         #"../output/cheetah-goal/2022_07_08_10_44_48",  # This is the bad run
                         "../output/cheetah-goal/2022_07_10_07_50_51"),
                'x': 'n_env_steps_total',
                'y': 'test_eval_success_rate',
                'x_correction': True
            },
        )
    },
    'cheetah-goal/cheetah-goal-ours': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'cheetah-goal',
        'ylim': [-2800, -500],
        'xlim': [300000, 7000000],
        'color_cycler': 'blue',
        'line_cycler': 'diff_lines',
        'groups': (
            {
                'name': 'Ours, run 1',
                'dirs': ("../output/cheetah-goal/2022_07_09_10_24_13",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': 'Ours, run 2',
                'dirs': ("../output/cheetah-goal/2022_07_30_20_41_18",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            #{
            #    'name': 'Ours, run 2',
            #    'dirs': ("../output/cheetah-goal/2022_07_08_10_44_48",),  # This is the bad run
            #    'x': 'n_env_steps_total',
            #    'y': 'test_eval_avg_reward_deterministic',
            #    'x_correction': True
            #},
            {
                'name': 'Ours, run 3',
                'dirs': ("../output/cheetah-goal/2022_07_10_07_50_51",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
        )
    },
    'cheetah-goal/cheetah-goal-ours-good-bad': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'cheetah-goal',
        'ylim': [-2800, -500],
        'xlim': [300000, 7000000],
        'color_cycler': 'blue',
        'line_cycler': 'diff_lines',
        'figsize': FIGSIZE_THIRD,
        'groups': (
            {
                'name': '(a)',
                'dirs': ("../output/cheetah-goal/2022_07_09_10_24_13",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': '(b)',
                'dirs': ("../output/cheetah-goal/2022_07_08_10_44_48",),  # This is the bad run
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            }
        )
    },
    'toy-goal-line': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'toy-goal-line',
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/toy-goal-line/2022_07_06_06_02_34",
                         "../output/toy-goal-line/2022_07_06_09_31_28",
                         "../output/toy-goal-line/2022_07_06_13_11_43"),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': 'CEMRL',
                'dirs': ("../../cemrl/output/toy-goal-line/2022_07_08_14_09_00",
                         "../../cemrl/output/toy-goal-line/2022_07_15_03_06_51",
                         "../../cemrl/output/toy-goal-line/2022_08_01_18_10_57",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
            },
            {
                'name': 'PEARL',
                'dirs': ("../../pearl/output/toy-goal-line/2022_07_08_18_28_06",
                         "../../pearl/output/toy-goal-line/2022_07_15_03_06_10",
                         "../../pearl/output/toy-goal-line/2022_07_29_16_00_52",),
                'x': 'Number of env steps total',
                'y': 'AverageReturn_all_test_tasks',
            },
        )
    },
    'thesis-defense/toy-goal-line-without-ours': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'toy-goal-line',
        'color_cycler': 'thesis_defense',
        'groups': (
            {
                'name': 'CEMRL',
                'dirs': ("../../cemrl/output/toy-goal-line/2022_07_08_14_09_00",
                         "../../cemrl/output/toy-goal-line/2022_07_15_03_06_51",
                         "../../cemrl/output/toy-goal-line/2022_08_01_18_10_57",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
            },
            {
                'name': 'PEARL',
                'dirs': ("../../pearl/output/toy-goal-line/2022_07_08_18_28_06",
                         "../../pearl/output/toy-goal-line/2022_07_15_03_06_10",
                         "../../pearl/output/toy-goal-line/2022_07_29_16_00_52",),
                'x': 'Number of env steps total',
                'y': 'AverageReturn_all_test_tasks',
            },
        )
    },
    'toy-goal-line-success': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average success rate',
        'title': 'toy-goal-line',
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/toy-goal-line/2022_07_06_06_02_34",
                         "../output/toy-goal-line/2022_07_06_09_31_28",
                         "../output/toy-goal-line/2022_07_06_13_11_43"),
                'x': 'n_env_steps_total',
                'y': 'test_eval_success_rate',
                'x_correction': True
            },
        )
    },
    'mw-goal-line': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'metaworld-goal-line',
        'xlim': [40000, None],
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/mw-reach-line/2022_07_12_20_46_43",
                         "../output/mw-reach-line/2022_08_03_17_43_54",
                         "../output/mw-reach-line/2022_08_06_09_52_31",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': 'CEMRL',
                'dirs': ("../../cemrl/output/metaworld-ml1-reach-line-action-restricted-distReward/2022_07_13_23_37_46",
                         "../../cemrl/output/metaworld-ml1-reach-line-action-restricted-distReward/2022_08_03_21_17_46",
                         "../../cemrl/output/metaworld-ml1-reach-line-action-restricted-distReward/2022_08_04_12_50_30",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
            },
            {
                'name': 'PEARL',
                'dirs': ("../../pearl/output/metaworld-ml1-reach-line-action-restricted-distReward/2022_08_17_12_34_57",
                         "../../pearl/output/metaworld-ml1-reach-line-action-restricted-distReward/2022_08_17_12_33_18",
                         "../../pearl/output/metaworld-ml1-reach-line-action-restricted-distReward/2022_08_17_12_24_16",),
                'x': 'Number of env steps total',
                'y': 'AverageReturn_all_test_tasks',
            },
        )
    },
    'mw-goal-line-success': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average success rate',
        'title': 'metaworld-goal-line',
        'xlim': [60000, None],
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/mw-reach-line/2022_07_12_20_46_43",
                         "../output/mw-reach-line/2022_08_03_17_43_54",
                         "../output/mw-reach-line/2022_08_06_09_52_31",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_success_rate',
                'x_correction': True
            },
        )
    },
    'cheetah-goal-halfline/window-comparison-performance': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'cheetah-goal-halfline',
        'color_cycler': 'variant',
        'legend_config': 'halfline-ablation',
        'ylim': [-3300, None],
        'xlim': [10000, None],
        'groups': (
            {
                'name': '$c^{dec}_t = [0, H]$',
                'dirs': ("../output/cheetah-halfline-goal-fullEp/2022_07_14_11_28_22",
                         "../output/cheetah-halfline-goal-fullEp/2022_07_23_16_56_12",
                         "../output/cheetah-halfline-goal-fullEp/2022_07_24_10_26_45",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': '$c^{dec}_t = [t-T, t+T]$',
                'dirs': ("../output/cheetah-halfline-goal-past-and-future/2022_07_14_09_20_26",
                         "../output/cheetah-halfline-goal-past-and-future/2022_07_22_14_19_26",
                         "../output/cheetah-halfline-goal-past-and-future/2022_07_23_05_09_26",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': '$c^{dec}_t = [t-T, t]$',
                'dirs': ("../output/cheetah-halfline-goal-past/2022_07_14_01_53_26",
                         "../output/cheetah-halfline-goal-past/2022_07_23_20_10_38",
                         "../output/cheetah-halfline-goal-past/2022_07_24_10_40_36",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': '$c^{dec}_t = [t, t]$',
                'dirs': ("../output/cheetah-halfline-goal-step/2022_07_14_13_37_23",
                         "../output/cheetah-halfline-goal-step/2022_07_22_14_18_48",
                         "../output/cheetah-halfline-goal-step/2022_07_23_03_38_01",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
        )
    },
    'cheetah-goal/ablation-performance': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'cheetah-goal-line',
        'color_cycler': 'variant',
        'ylim': [-2800, -500],
        'xlim': [100000, None],
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/cheetah-goal/2022_07_09_10_24_13",
                         "../output/cheetah-goal/2022_07_30_20_41_18",
                         #"../output/cheetah-goal/2022_07_08_10_44_48",  # This is the bad run
                         "../output/cheetah-goal/2022_07_10_07_50_51"),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': 'Ours (no linked episodes)',
                'dirs': ("../output/cheetah-goal-separateEpisodes-with-exploration/2022_08_11_08_31_22",
                         "../output/cheetah-goal-separateEpisodes-with-exploration/2022_08_11_08_31_26",
                         "../output/cheetah-goal-separateEpisodes/2022_07_10_10_00_00",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': 'Ours (no exploration)',
                'dirs': ("../output/cheetah-goal-noExploration/2022_07_11_22_17_47",
                         "../output/cheetah-goal-noExploration/2022_07_26_14_41_18",
                         "../output/cheetah-goal-noExploration/2022_07_28_01_01_40",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
        )
    },
    'cheetah-goal/ablation-performance-success-rate': {
        'x_label': 'Training transition $n$',
        'y_label': 'Test success rate',
        'title': 'cheetah-goal-line',
        'color_cycler': 'variant',
        'xlim': [100000, None],
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/cheetah-goal/2022_07_09_10_24_13",
                         "../output/cheetah-goal/2022_07_30_20_41_18",
                         #"../output/cheetah-goal/2022_07_08_10_44_48",  # This is the bad run
                         "../output/cheetah-goal/2022_07_10_07_50_51"),
                'x': 'n_env_steps_total',
                'y': 'test_eval_success_rate',
                'x_correction': True
            },
            {
                'name': 'Ours (no linked episodes)',
                'dirs': ("../output/cheetah-goal-separateEpisodes-with-exploration/2022_08_11_08_31_22",
                         "../output/cheetah-goal-separateEpisodes-with-exploration/2022_08_11_08_31_26",
                         "../output/cheetah-goal-separateEpisodes/2022_07_10_10_00_00",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_success_rate',
                'x_correction': True
            },
            {
                'name': 'Ours (no exploration)',
                'dirs': ("../output/cheetah-goal-noExploration/2022_07_11_22_17_47",
                         "../output/cheetah-goal-noExploration/2022_07_26_14_41_18",
                         "../output/cheetah-goal-noExploration/2022_07_28_01_01_40",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_success_rate',
                'x_correction': True
            },
        )
    },
    'toy-goal-plane/performance': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average return $\hat{R}$',
        'title': 'toy-goal-2D',
        'figsize': FIGSIZE_HALF_SQUARE,
        'xlim': [80000, None],
        'ylim': [-4000, None],
        'groups': (
            {
                'name': 'Ours',
                'dirs': ("../output/toy-goal-2D/2022_07_12_17_30_36",
                         "../output/toy-goal-2D/2022_08_04_08_50_52",
                         "../output/toy-goal-2D/2022_08_07_01_53_13",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
                'x_correction': True
            },
            {
                'name': 'CEMRL',
                'dirs': ("../../cemrl/output/toy-goal-plane/2022_07_13_07_54_32",
                         "../../cemrl/output/toy-goal-plane/2022_08_01_23_10_48",
                         "../../cemrl/output/toy-goal-plane/2022_08_02_14_24_37",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_avg_reward_deterministic',
            },
            {
                'name': 'PEARL',
                'dirs': ("../../pearl/output/toy-goal-plane/2022_07_12_22_18_09",
                         "../../pearl/output/toy-goal-plane/2022_07_29_16_01_35",
                         "../../pearl/output/toy-goal-plane/2022_07_30_00_31_31",),
                'x': 'Number of env steps total',
                'y': 'AverageReturn_all_test_tasks',
            },
        )
    },
    'toy-goal-plane/train-vs-test': {
        'x_label': 'Training transition $n$',
        'y_label': 'Average success rate',
        'title': 'toy-goal-2D',
        'color_cycler': 'blue',
        'line_cycler': 'diff_lines',
        'groups': (
            {
                'name': 'Meta-testing set',
                'dirs': ("../output/toy-goal-2D/2022_07_12_17_30_36",
                         "../output/toy-goal-2D/2022_08_04_08_50_52",
                         "../output/toy-goal-2D/2022_08_07_01_53_13",),
                'x': 'n_env_steps_total',
                'y': 'test_eval_success_rate',
                'x_correction': True
            },
            {
                'name': 'Meta-training set',
                'dirs': ("../output/toy-goal-2D/2022_07_12_17_30_36",
                         "../output/toy-goal-2D/2022_08_04_08_50_52",
                         "../output/toy-goal-2D/2022_08_07_01_53_13",),
                'x': 'n_env_steps_total',
                'y': 'train_eval_success_rate',
                'x_correction': True
            },
        )
    },
}


def load_progress_log(log_dir, file='progress.csv'):
    return pd.read_csv(os.path.join(log_dir, file))


def get_param(param_name, config):
    param = DEFAULTS[param_name] if param_name not in config.keys() else config[param_name]
    if param_name == 'color_cycler':
        param = COLOR_CYCLER[param]
    if param_name == 'line_cycler':
        param = LINE_CYCLER[param]
    return param


def save_fig(fig, save_file, save_dir=None):
    if save_dir is not None:
        save_file = os.path.join(save_dir, save_file)
    fig.savefig(save_file, dpi=300)


def env_step_correction(x, dir):
    with open(os.path.join(dir, 'variant.json')) as f:
        variant = json.load(f)
        params = variant['algo_params']
    if params['exploration_agent'] is None:
        return x
    else:
        pretraining_steps = params['exploration_ensemble_agents'] * \
                            (params['exploration_pretraining_epoch_steps'] * params['exploration_pretraining_epochs']
                             + params['exploration_pretraining_steps'])
        return x + pretraining_steps


def plot_progress_curve(group, fig, ax):
    x = pd.DataFrame()
    y = pd.DataFrame()
    num_groups = len(group['dirs'])
    for i, dir in enumerate(group['dirs']):
        data = load_progress_log(dir)
        x[i] = data[group['x']]
        if 'x_correction' in group.keys():
            x[i] = env_step_correction(x[i], dir)
        y[i] = data[group['y']]
        if i > 0 and not (x[i] == x[0]).all():
            raise ValueError(f'Different experiments in group {group["name"]} have different x-values.')
    if num_groups > 1:
        x = x[0]
        y_std = y.std(axis=1)
        y = y.mean(axis=1)
        ax.fill_between(x, y-y_std, y+y_std, alpha=0.3, linewidth=0)

    ax.plot(x, y)
    return fig, ax


def print_legend(fig, ax, legend_names, legend_config):
    if legend_config is None:
        ax.legend(legend_names)
        fig.tight_layout()
    elif legend_config == 'halfline-ablation':
        fig.legend(legend_names, handlelength=1, loc='lower center',
                  bbox_to_anchor=(0.5, 0), ncol=2)
        fig.tight_layout(rect=[0, 0.22, 1, 1])
        #ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    else:
        raise ValueError(f'Unknown legend config {legend_config}')



@click.command()
@click.option('--save_dir', default="../../../Thesis/experiments/")
def main(save_dir):
    config_names = ('thesis-defense/toy-goal-line-without-ours',)
    if config_names is None:
        config_names = CONFIGS.keys()
    for config_name in config_names:
        config = CONFIGS[config_name]
        figsize = get_param('figsize', config)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        legend_names = []
        prop_cycle = (get_param('color_cycler', config) + get_param('line_cycler', config))
        ax.set_prop_cycle(prop_cycle)
        for group in config['groups']:
            fig, ax = plot_progress_curve(group, fig, ax)
            legend_names += [group['name']]
        ax.set_title(get_param('title', config))
        ax.set_xlabel(get_param('x_label', config))
        ax.set_ylabel(get_param('y_label', config))
        ax.set_xscale(get_param('x_scale', config))
        ax.set_xlim(get_param('xlim', config))
        ax.set_ylim(get_param('ylim', config))
        print_legend(fig, ax, legend_names, get_param('legend_config', config))
        file_path = os.path.join(save_dir, f'{config_name}.pdf')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(file_path, dpi=300)
        fig.show()


if __name__ == '__main__':
    main()

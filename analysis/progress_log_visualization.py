import os.path

import click
import pandas as pd
import matplotlib.pyplot as plt


def load_progress_log(dir):
    return pd.read_csv(os.path.join(dir, 'progress.csv'))


def save_fig(fig, save_file, save_dir=None):
    if save_dir is not None:
        save_file = os.path.join(save_dir, save_file)
    fig.savefig(save_file, dpi=300)


def plot_progress_curve(data, y, y_shaded=None, x='n_env_steps_total', x_scale='log',
                        show=False, save_file=None, fig_ax=None, **kwargs):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()

    if x == 'n_env_steps_total':
        x_label = "Training transition $n$"
    else:
        x_label = x
    x_data = data[x]

    if y in ['train_avg_reward', 'train_eval_avg_reward_deterministic']:
        y_axis = 'Train reward'
        y_label = 'Average reward'
        y_data = data['train_eval_avg_reward_deterministic']
    elif y in ['test_avg_reward', 'test_eval_avg_reward_deterministic']:
        y_axis = 'Test reward'
        y_label = 'Average reward'
        y_data = data['test_eval_avg_reward_deterministic']
    else:
        y_label = y
        y_axis = y
        y_data = data[y]

    # Do the main plot
    line = ax.plot(x_data, y_data, label=y_label)

    if y_shaded == 'minmax' and y in ['train_avg_reward', 'train_eval_avg_reward_deterministic']:
        fill_lower = data['train_eval_min_reward_deterministic']
        fill_upper = data['train_eval_max_reward_deterministic']
        shaded_label = 'Min and max evaluation span'
        shaded = ax.fill_between(x_data, fill_lower, fill_upper, alpha=0.5, linewidth=0, label=shaded_label)
    if y_shaded == 'minmax' and y in ['test_avg_reward', 'test_eval_avg_reward_deterministic']:
        fill_lower = data['test_eval_min_reward_deterministic']
        fill_upper = data['test_eval_max_reward_deterministic']
        shaded = ax.fill_between(x_data, fill_lower, fill_upper, alpha=0.5, linewidth=0)
        shaded_label = 'Min and max evaluation span'
        #ax.legend([line, shaded], [y_label, shaded_label])

    ax.set_title(y_axis)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_axis)
    ax.set_xscale(x_scale)
    ax.grid()
    if 'ybound' in kwargs.keys():
        ax.set_ybound(kwargs['ybound'])
    if 'xbound' in kwargs.keys():
        ax.set_xbound(kwargs['xbound'])

    if save_file is not None:
        save_fig(fig, save_file)
    if show:
        fig.show()
    return fig, ax


@click.command()
@click.argument('dir', default="output/cheetah-stationary-dir/2021_11_28_23_06_08")
@click.option('--y', default='Reconstruction_val_reward_loss')#'test_avg_reward')
@click.option('--y_shaded', default=None)#'minmax_test_reward')
@click.option('--x_scale', default='log')
@click.option('--save', default=False)
@click.option('--save_dir', default=None)
@click.option('--save_prefix', default='')
def main(dir, y, y_shaded, x_scale, save, save_dir, save_prefix):
    save_dir = '../output_analysis/2022_01_14/'
    cheetah = '../output/cheetah-stationary-dir/2022_01_12_04_00_32'
    cheetah_obs = '../output/cheetah-stationary-dir-observable/2022_01_14_05_51_39'
    cheetah_obs_2 = '../output/cheetah-stationary-dir-observable-2/2022_01_14_05_50_59'

    ml1 = '../output/metaworld-ml1-reach/2022_01_12_03_58_33'
    ml1_observable = '../output/metaworld-ml1-reach-observable/2021_12_23_12_53_08'

    y = 'test_avg_reward'
    y_shaded = 'minmax_test_reward'
    x_scale='log'
    fig, ax = plot_progress_curve(cheetah, y, y_shaded, save=True, save_dir=save_dir, save_file='cheetah-reward.png')
    fig, ax = plot_progress_curve(cheetah_obs, y, y_shaded, save=True, save_dir=save_dir, save_file='cheetah-obs-reward.png')
    fig, ax = plot_progress_curve(cheetah_obs_2, y, y_shaded, save=True, save_dir=save_dir, save_file='cheetah-obs-2-reward.png')

    fig, ax = plot_progress_curve(cheetah, y, y_shaded, x_scale=x_scale, save=False)
    fig, ax = plot_progress_curve(cheetah_obs, y, y_shaded, x_scale=x_scale, fig_ax=(fig,ax), save=False)
    fig, ax = plot_progress_curve(cheetah_obs_2, y, y_shaded, x_scale=x_scale, fig_ax=(fig,ax), save=True, save_dir=save_dir, save_file='all-cheetah-reward.png')
    ax.legend(['base', 'observable_task', 'observable_task_reward'])
    fig.savefig(os.path.join(save_dir, 'all-cheetah-reward.png'), dpi=300)

    fig, ax = plot_progress_curve(cheetah, 'MI(z, true_task_spec)', show=False)
    fig, ax = plot_progress_curve(cheetah, 'MI(z, reward)', fig_ax=(fig, ax), show=False)
    ax.set_ylim(0,1)
    ax.set_xscale('linear')
    ax.set_ylabel('mutual information [bits]')
    ax.legend(['MI(z, true_task)', 'MI(z, reward)'])
    fig.savefig(os.path.join(save_dir, 'cheeah-mi-z.png'), dpi=300)
    fig.show()

    fig, ax = plot_progress_curve(cheetah, 'MI(y, true_task_spec)', show=False)
    fig, ax = plot_progress_curve(cheetah, 'MI(y, reward)', fig_ax=(fig, ax), show=False)
    ax.set_ylim(0, 1)
    ax.set_xscale('linear')
    ax.set_ylabel('mutual information [bits]')
    ax.legend(['MI(y, true_task)', 'MI(y, reward)'])
    fig.savefig(os.path.join(save_dir, 'cheeah-mi-y.png'), dpi=300)
    fig.show()

    fig, ax = plot_progress_curve(cheetah, 'MI(r, true_task_spec)', x_scale='linear')
    ax.set_ylim(0, 1)
    ax.set_xscale('linear')
    ax.set_ylabel('mutual information [bits]')
    ax.legend(['MI(reward, true_task)'])
    fig.savefig(os.path.join(save_dir, 'cheetah-mi-r.png'), dpi=300)
    fig.show()

    # Metaworld
    fig, ax = plot_progress_curve(ml1, 'train_avg_reward', 'minmax_train_reward', x_scale=x_scale, save=False, show=False)
    fig, ax = plot_progress_curve(ml1_observable, 'train_avg_reward', 'minmax_train_reward', x_scale=x_scale, fig_ax=(fig,ax), save=True,
                                  save_dir=save_dir, save_file='all-ml-reward-train.png')
    ax.legend(['base', 'observable_task'])
    fig.savefig(os.path.join(save_dir, 'all-ml-reward-train.png'), dpi=300)

    fig, ax = plot_progress_curve(ml1, 'train_eval_success_rate', x_scale=x_scale, save=False, show=False)
    fig, ax = plot_progress_curve(ml1_observable, 'train_eval_success_rate', x_scale=x_scale, fig_ax=(fig, ax), save=True,
                                  save_dir=save_dir, save_file='all-ml-success-train.png')
    ax.legend(['base', 'observable_task'])
    fig.savefig(os.path.join(save_dir, 'all-ml-success-train.png'), dpi=300)

    fig, ax = plot_progress_curve(ml1, y, y_shaded, x_scale=x_scale, save=False, show=False)
    fig, ax = plot_progress_curve(ml1_observable, y, y_shaded, x_scale=x_scale, fig_ax=(fig,ax), save=True,
                                  save_dir=save_dir, save_file='all-ml-reward.png')
    ax.legend(['base', 'observable_task'])
    fig.savefig(os.path.join(save_dir, 'all-ml-reward.png'), dpi=300)

    fig, ax = plot_progress_curve(ml1, 'test_eval_success_rate', x_scale=x_scale, save=False, show=False)
    fig, ax = plot_progress_curve(ml1_observable, 'test_eval_success_rate', x_scale=x_scale, fig_ax=(fig, ax), save=True,
                                  save_dir=save_dir, save_file='all-ml-success.png')
    ax.legend(['base', 'observable_task'])
    fig.savefig(os.path.join(save_dir, 'all-ml-success.png'), dpi=300)

    fig, ax = plot_progress_curve(ml1, 'MI(z, true_task_spec)', show=False)
    fig, ax = plot_progress_curve(ml1, 'MI(z, reward)', fig_ax=(fig, ax), show=False)
    ax.set_ylim(0, None)
    ax.set_xscale('linear')
    ax.set_ylabel('mutual information [bits]')
    ax.legend(['MI(z, true_task)', 'MI(z, reward)'])
    fig.savefig(os.path.join(save_dir, 'ml-mi-z.png'), dpi=300)
    fig.show()

    fig, ax = plot_progress_curve(ml1, 'MI(r, true_task_spec)', x_scale='linear')
    ax.set_ylim(0, None)
    ax.set_xscale('linear')
    ax.set_ylabel('mutual information [bits]')
    ax.legend(['MI(reward, true_task)'])
    fig.savefig(os.path.join(save_dir, 'ml-mi-r.png'), dpi=300)
    fig.show()

if __name__ == '__main__':
    main()

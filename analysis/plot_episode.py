import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_quantity(results, quantity):
    if quantity == 'time':
        data = list(range(len(results['env_infos'])))
    elif quantity in ['observations', 'next_observations', 'actions', 'rewards', 'task_indicators', 'base_task_indicators',
             'next_task_indicators', 'next_base_task_indicators', 'terminals']:
        data = results[quantity]
    elif quantity in ['base_task', 'specification']:
        data = [a[0][quantity] for a in results['true_tasks']]
    elif quantity in results['env_infos'][0].keys():
        data = [a[quantity] for a in results['env_infos']]
    elif quantity in ['z_means', 'z_stds']:
        # The squeeze is a bit hacky and will break for actual Gaussian mixtures (not just single Gaussians)
        data = np.array([a['latent_distribution'][quantity] for a in results['agent_infos']]).squeeze()
        if data.ndim > 1:
            y_probs = np.array([a['latent_distribution']['y_probs'] for a in results['agent_infos']])
            y_selection = y_probs.argmax(axis=-1)
            data = [data[i, y_selection[i]] for i in range(len(y_selection))]
    else:
        raise ValueError(f'Desired quantity={quantity} is unknown or not available for this environment')
    return np.array(data).squeeze()


def get_plot_specification(specifications):
    if isinstance(specifications, str):
        specifications = [specifications]

    plots = []
    for plot_spec in specifications:
        y_const, y_fill = None, None
        if '_fill_' in plot_spec:
            plot_spec, y_fill = plot_spec.split('_fill_')
        if '_const_' in plot_spec:
            plot_spec, y_const = plot_spec.split('_const_')
        x, y = plot_spec.split('_vs_')
        plots.append({'x': x, 'y': y, 'y_const': y_const, 'y_fill': y_fill})
    return plots


def plot_per_episode(results, y, y_const=None, y_fill=None, x='time', fig_ax=None):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(1, 1)

    data_x = get_quantity(results, x)
    data_y = get_quantity(results, y)

    p = ax.plot(data_x, data_y)
    if y_fill is not None:
        # Skip the 0-th timestep, since the "informationless" encoding can have very large variance
        data_y_fill = get_quantity(results, y_fill)
        ax.fill_between(data_x[1:], (data_y - data_y_fill)[1:], (data_y + data_y_fill)[1:], color=p[-1].get_color(), alpha=0.3)
    if y_const is not None:
        data_y_const = get_quantity(results, y_const)
        ax.plot(data_x, data_y_const, color=p[-1].get_color(), linestyle='--')

    ax.set_xlabel(x.replace('_', ' '))
    ax.set_ylabel(y.replace('_', ' '))
    ax.set_title(y.replace('_', ' '))

    return fig, ax

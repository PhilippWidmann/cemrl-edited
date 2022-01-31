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
    else:
        raise ValueError(f'Desired quantity={quantity} is unknown or not available for this environment')
    return data


def get_plot_specification(specifications):
    if isinstance(specifications, str):
        specifications = [specifications]

    plots = []
    for plot_spec in specifications:
        if '_const_' in plot_spec:
            x_y_name, y_const = plot_spec.split('_const_')
        else:
            x_y_name, y_const = plot_spec, None
        x, y = x_y_name.split('_vs_')
        plots.append({'x': x, 'y': y, 'y_const': y_const})
    return plots


def plot_per_episode(results, y, y_const=None, x='time', fig_ax=None):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(1, 1)

    data_x = get_quantity(results, x)
    data_y = get_quantity(results, y)

    p = ax.plot(data_x, data_y)
    if y_const is not None:
        data_y_const = get_quantity(results, y_const)
        ax.plot(data_x, data_y_const, color=p[-1].get_color(), linestyle='--')

    ax.set_xlabel(x.replace('_', ' '))
    ax.set_ylabel(y.replace('_', ' '))
    ax.set_title(y.replace('_', ' '))

    return fig, ax

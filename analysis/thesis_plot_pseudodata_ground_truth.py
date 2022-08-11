import os

import click
import numpy as np
import matplotlib.pyplot as plt
from thesis_plot_progress import FIGSIZE_HALF, SMALL_SIZE, MEDIUM_SIZE

plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-3)    # legend fontsize; corrected because it visually appears larger
plt.rc('legend', title_fontsize=SMALL_SIZE)    # legend fontsize

def plot(states, targets, rewards):
    # Unused: Contour plot
    # fig_con, ax_con = plt.subplots(figsize=FIGSIZE_HALF)
    # contours = np.concatenate((np.linspace(-50, 0, 10, endpoint=False), [-0.5, 0]))
    # cf = ax_con.contourf(states, targets, rewards, levels=contours)
    # cbar_con = fig_con.colorbar(cf)
    # cbar_con.set_label('Reward', rotation=90)
    # ax_con.set_xlabel('Position')
    # ax_con.set_ylabel('Targets')
    # ax_con.set_title('Contour plot of reward function')
    # fig_con.tight_layout()
    # fig_con.show()

    # Multiple line plot
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(targets), vmax=max(targets)))
    cbar = fig.colorbar(sm)
    cbar.set_label('Goal', rotation=90)
    ax.set_xlabel('Position')
    ax.set_ylabel('Reward')
    #ax.set_title('Reward function for different targets')

    for i, z in enumerate(targets):
        if z % 4 == 0:
            ax.plot(states, rewards[i].squeeze(), color=colors[i])
    fig.tight_layout()
    return fig


@click.command()
@click.option('--save_dir', default="../../../Thesis/experiments/pseudodata/")
def main(save_dir):
    # actual reward function
    states = np.linspace(-25, 25, 101)
    targets = np.linspace(-25, 25, 101)
    rewards = np.array([[-abs(s - t) for s in states] for t in targets])
    fig = plot(states, targets, rewards)

    file_path = os.path.join(save_dir, f'true-decoder-functions.pdf')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fig.savefig(file_path, dpi=300)
    fig.show()


if __name__ == '__main__':
    main()

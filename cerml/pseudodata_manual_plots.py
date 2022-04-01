import numpy as np
import matplotlib.pyplot as plt


def plot(states, targets, rewards):
    # Contour plot
    fig_con, ax_con = plt.subplots()
    contours = np.concatenate((np.linspace(-50, 0, 10, endpoint=False), [-0.5, 0]))
    cf = ax_con.contourf(states, targets, rewards, levels=contours)
    cbar_con = fig_con.colorbar(cf)
    cbar_con.set_label('reward', rotation=90)
    ax_con.set_xlabel('position')
    ax_con.set_ylabel('targets')
    ax_con.set_title('Contour plot of reward function')

    fig_con.tight_layout()
    fig_con.show()

    # Multiple line plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(targets), vmax=max(targets)))
    fig, ax = plt.subplots()
    cbar = fig.colorbar(sm)
    cbar.set_label('target', rotation=90)
    ax.set_xlabel('position')
    ax.set_ylabel('reward')
    ax.set_title('Reward function for different targets')

    for i, z in enumerate(targets):
        if z % 4 == 0:
            ax.plot(states, rewards[i].squeeze(), color=colors[i])
    fig.tight_layout()
    fig.show()


def main():
    # actual reward function
    states = np.linspace(-25, 25, 101)
    targets = np.linspace(-25, 25, 101)
    rewards = np.array([[-abs(s - t) for s in states] for t in targets])
    plot(states, targets, rewards)

    # learned decoder function
    states = np.linspace(-25, 25, 101)
    targets = np.linspace(-25, 25, 101)
    #targets_reordered = np.concatenate((targets[:int(len(targets)/2)],
    #                                   np.flip(targets[int(len(targets)/2):])))
    def dec(s, t):
        if t >= 0:
            return -abs(abs(s) - t)
        else:
            return -abs(abs(s) - t)
    rewards = np.array([[dec(s, t) for s in states] for t in targets])
    plot(states, targets, rewards)


if __name__ == '__main__':
    main()

import numpy as np
import torch.nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify


def construct_exploration_agent(exploration_type,
                                policy,
                                replay_buffer):
    if exploration_type == 'counterfactual_task':
        return CounterfactualTaskExplorationAgent(policy,
                                                  replay_buffer)
    else:
        return ToyGoalExplorationAgent(exploration_type,
                                       zigzag_max=25,
                                       step_size=1)


class CounterfactualTaskExplorationAgent(nn.Module):
    def __init__(self,
                 policy,
                 replay_buffer
                 ):
        super().__init__()
        self.policy = policy
        self.replay_buffer = replay_buffer

    def get_action(self, encoder_input, state, input_padding=None, deterministic=False, z_debug=None, env=None,
                   return_distributions=False, agent_info=None):
        state = ptu.from_numpy(state).view(1, -1)
        if agent_info == {}:
            z, y = self.replay_buffer.sample_task_indicator()
        else:
            z, y = agent_info['counterfactual_task']

        a, a_info = self.policy.get_action(state, z, y, deterministic=deterministic)
        a_info['counterfactual_task'] = (z, y)
        a_info['exploration_trajectory'] = True
        return a, a_info, \
               np_ify(z.clone().detach())[0, :], \
               np_ify(y.clone().detach())[0]


class ToyGoalExplorationAgent(nn.Module):
    def __init__(self,
                 exploration_type,
                 zigzag_max=25,
                 step_size=1
                 ):
        super().__init__()
        if exploration_type not in ['zigzag', 'line']:
            raise ValueError(f'Unknow exploration_type {exploration_type}')
        self.exploration_type = exploration_type
        self.zigzag_max = zigzag_max
        self.step_size = step_size

    def get_action(self, encoder_input, state, input_padding=None, deterministic=False, z_debug=None, env=None,
                   return_distributions=False, agent_info=None):
        # Note: This is handcrafted specifically for the toy environment where the state is the one-dimensional position
        previous_states = ptu.get_numpy(encoder_input[0, :, 0])
        direction = np.sign(state[0] - previous_states[-1])
        if direction == 0:
            direction = np.random.choice([-1, 1])

        if self.exploration_type == 'zigzag':
            # Go in a zigzag line between furthest targets
            if abs(state[0]) < self.zigzag_max:
                action = direction * self.step_size
            else:
                action = - direction * self.step_size
        elif self.exploration_type == 'line':
            # Go to one side and stay at the furthest target
            if abs(state[0]) < self.zigzag_max:
                action = direction * self.step_size
            else:
                action = 0
        else:
            raise NotImplementedError(f'Unknown exploration type {self.exploration_type}')

        action_info = {'exploration_trajectory': True}
        return np.array([action]), action_info, np.array([0.0]), np.array(0)

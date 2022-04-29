import os.path
import warnings
import numpy as np
import torch
import torch.nn as nn

# For compatibility with the URLB submodule
import dm_env
from dm_env._environment import StepType
from dm_env.specs import Array, BoundedArray

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify

import sys
sys.path.append("submodules/url_benchmark")
from submodules.url_benchmark.pretrain import pretrain_model
from submodules.url_benchmark.dmc import ExtendedTimeStep

from meta_rand_envs.wrappers import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv


def construct_exploration_agent(exploration_type,
                                policy,
                                replay_buffer,
                                env_name,
                                env_args,
                                experiment_log_dir,
                                max_timesteps):
    if exploration_type == 'counterfactual_task':
        return CounterfactualTaskExplorationAgent(policy,
                                                  replay_buffer)
    elif 'urlb' in exploration_type:
        # Expect the specific agent to be specified in the format urlb_rnd
        agent_type = exploration_type.split('_')
        if len(agent_type) == 2:
            agent_type = agent_type[1]
        else:
            warnings.warn('Specific URLB agent unspecified or not understood. Defaulting to "rnd"')
            agent_type = 'rnd'
        return URLBAgent(env_name, env_args, agent_type, experiment_log_dir, max_timesteps)
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


class URLBAgent(nn.Module):
    def __init__(self,
                 env_name,
                 env_args,
                 agent_type,
                 experiment_log_dir,
                 max_timesteps
                 ):
        super().__init__()
        self.agent_type = agent_type

        self.workdir = os.path.join(experiment_log_dir, 'exploration')
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        env_args['exploration_reward'] = True
        env = ENVS[env_name](**env_args)
        if env_args['use_normalized_env']:
            env = NormalizedBoxEnv(env)
        self.train_env = URLBAgent.URLBEnvWrapper(env, max_timesteps)

        env = ENVS[env_name](**env_args)
        if env_args['use_normalized_env']:
            env = NormalizedBoxEnv(env)
        self.eval_env = URLBAgent.URLBEnvWrapper(env, max_timesteps)

        cfg_override = [f'agent={self.agent_type}',
                        f'save_video=false',
                        'num_train_frames=10000']
        self.workspace = pretrain_model(self.train_env, self.eval_env, cfg_override, self.workdir)

    def get_action(self, encoder_input, state, input_padding=None, deterministic=False, z_debug=None, env=None,
                   return_distributions=False, agent_info=None):
        if 'meta' in agent_info.keys():
            meta = agent_info['meta']
        else:
            meta = self.workspace.agent.init_meta()
        with torch.no_grad():
            action = self.workspace.agent.act(state,
                                              meta,
                                              self.workspace.global_step,
                                              eval_mode=False)
        # self.workspace._global_step += 1
        # meta info changes only periodically for all agents (e.g. choosing a new random skill in DIAYN)
        # since we init meta at the start of each episode, do not have to update it here
        agent_info = {'meta': meta, 'exploration_trajectory': True}
        return action, agent_info, np.array([0.0]), np.array(0)

    class URLBEnvWrapper(dm_env.Environment):
        def __init__(self, env, max_timesteps):
            self.env = env
            self.timestep = 0
            self.max_timesteps = max_timesteps
            self._observation_spec = BoundedArray(shape=tuple(env.observation_space.shape),
                                                  dtype=env.observation_space.dtype,
                                                  minimum=env.observation_space.low,
                                                  maximum=env.observation_space.high,
                                                  name='observation')
            self._action_spec = BoundedArray(shape=tuple(env.action_space.shape),
                                             dtype=env.action_space.dtype,
                                             minimum=env.action_space.low,
                                             maximum=env.action_space.high,
                                             name='action')

        def reset(self) -> ExtendedTimeStep:
            self.timestep = 0
            ob = self.env.reset()
            action = np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype)
            return ExtendedTimeStep(StepType.FIRST, 0, 0.99, ob, action) # Todo: 0.99 should be the discount factor

        def step(self, action) -> ExtendedTimeStep:
            self.timestep += 1
            assert self.timestep <= self.max_timesteps
            ob, reward, done, info = self.env.step(action)
            step_type = StepType.LAST if self.timestep == self.max_timesteps else StepType.MID
            return ExtendedTimeStep(step_type, reward, 0.99, ob, action)

        def observation_spec(self):
            return self._observation_spec

        def action_spec(self):
            return self._action_spec

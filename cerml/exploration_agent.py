import copy
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
from submodules.url_benchmark.pretrain import generate_model
from submodules.url_benchmark.dmc import ExtendedTimeStep

from meta_rand_envs.wrappers import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv


def construct_exploration_agent(exploration_type,
                                policy,
                                replay_buffer,
                                env_name,
                                env_args,
                                experiment_log_dir,
                                max_timesteps,
                                exploration_pretraining_steps,
                                exploration_epoch_training_steps,
                                state_preprocessor,
                                num_exploration_ensemble_agents,
                                showcase_itr):
    if exploration_type == 'counterfactual_task':
        return CounterfactualTaskExplorationAgent(policy,
                                                  replay_buffer)
    elif 'ensemble_urlb' in exploration_type:
        exploration_type = exploration_type.replace('ensemble_urlb', 'urlb')
        return EnsembleURLBAgent(num_exploration_ensemble_agents, env_name, env_args, exploration_type, experiment_log_dir, max_timesteps, exploration_pretraining_steps, exploration_epoch_training_steps, showcase_itr, state_preprocessor)
    elif 'urlb' in exploration_type:
        return URLBAgent(env_name, env_args, exploration_type, experiment_log_dir, max_timesteps, exploration_pretraining_steps, exploration_epoch_training_steps, showcase_itr, state_preprocessor)
    else:
        return ToyGoalExplorationAgent(exploration_type,
                                       zigzag_max=25,
                                       step_size=1)


class ExplorationAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def get_action(self, *args, **kwargs):
        raise NotImplementedError('The generic ExplorationAgent has no methods. Instantiate a subclass.')

    def train_agent(self):
        pass

    def save_agent(self, epoch):
        pass


class CounterfactualTaskExplorationAgent(ExplorationAgent):
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
        return a, a_info, np.array([None]), None


class ToyGoalExplorationAgent(ExplorationAgent):
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
        if direction == 0 and abs(state[0]) <= 10E-6:
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
        return np.array([action]), action_info, np.array([None]), None


class URLBAgent(ExplorationAgent):
    def __init__(self,
                 env_name,
                 env_args,
                 exploration_type,
                 experiment_log_dir,
                 max_timesteps,
                 pretraining_steps,
                 epoch_training_steps,
                 showcase_itr,
                 state_preprocessor,
                 ensemble_id=None,
                 ):
        super().__init__()
        # Expect the specific agent to be specified in the format urlb_rnd
        agent_type = exploration_type.split('_', 1)
        if len(agent_type) == 2:
            agent_type = agent_type[1]
        else:
            raise ValueError('Specific URLB agent unspecified or not understood.')
        self.agent_type = agent_type
        self.pretraining_steps = pretraining_steps
        self.epoch_training_steps = epoch_training_steps
        self.showcase_itr = showcase_itr
        self.state_preprocessor = state_preprocessor
        self.agent_type = agent_type

        self.workdir = os.path.join(experiment_log_dir, 'exploration')
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        env_args = copy.deepcopy(env_args)
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
                        f'num_train_frames={pretraining_steps+1}',
                        f'snapshots={[pretraining_steps]}',
                        f'snapshot_dir="."']
        print('Initializing exploration agent')
        self.workspace, self.trained = generate_model(self.train_env, self.eval_env, cfg_override, self.workdir, snapshot_itr=self.showcase_itr,
                                                      snapshot_prefix=f'agent_{ensemble_id}_' if ensemble_id is not None else '')
        if agent_type == 'smm_autodim':
            self.workspace.agent.delayed_init(state_preprocessor)
        if self.trained:
            print('Loaded exploration agent from file')
        else:
            print("Pretraining exploration agent")
            self.train_agent(additional_frames=0)
            self.trained = True

    def train_agent(self, additional_frames=None, agent_id=''):
        additional_frames = self.epoch_training_steps if additional_frames is None else additional_frames
        print(f"Training exploration agent {agent_id}")
        if self.agent_type == 'smm_autodim':
            self.state_preprocessor.to(self.workspace.device)
        self.workspace.train(additional_frames=additional_frames, save_snapshots=False)
        if self.agent_type == 'smm_autodim':
            self.state_preprocessor.to(ptu.device)

    def save_agent(self, epoch):
        self.workspace.save_snapshot(epoch)

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
        return action, agent_info, np.array([None]), None#np.array([0.0]), np.array(0)

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


class EnsembleURLBAgent(ExplorationAgent):
    def __init__(self,
                 num_ensemble_agents,
                 *args,
                 **kwargs):
        super().__init__()
        self.agent_ids = list(range(num_ensemble_agents))
        self.agents = torch.nn.ModuleList()
        for i in self.agent_ids:
            self.agents.append(URLBAgent(*args, ensemble_id=i, **kwargs))

    def get_action(self, *args, agent_info=None, **kwargs):
        if 'ensemble_agent' in agent_info.keys():
            id = agent_info['ensemble_agent']
        else:
            id = np.random.choice(self.agent_ids)
        action, agent_info, task, base_task = self.agents[id].get_action(*args, agent_info=agent_info, **kwargs)
        agent_info['ensemble_agent'] = id
        return action, agent_info, task, base_task

    def train_agent(self):
        for i in range(len(self.agents)):
            self.agents[i].train_agent(self.agent_ids[i])

    def save_agent(self, epoch):
        for a in self.agents:
            a.save_agent(epoch)

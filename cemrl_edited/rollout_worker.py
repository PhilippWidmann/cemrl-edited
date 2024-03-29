import gym
gym.logger.set_level(40)

import numpy as np
import torch
import ray
import os

from collections import OrderedDict

from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
import rlkit.torch.pytorch_util as ptu

from meta_rand_envs.wrappers import ENVS


class RolloutCoordinator:
    def __init__(self,
                 env,
                 env_name,
                 env_args,
                 train_or_showcase,
                 agent,
                 exploration_agent,
                 replay_buffer,
                 time_steps,
                 max_path_length,
                 permute_samples,

                 use_multiprocessing,
                 use_data_normalization,
                 use_sac_data_normalization,
                 num_workers,
                 gpu_id,
                 scripted_policy
                 ):
        self.env = env
        self.env_name = env_name
        self.env_args = env_args
        self.train_or_showcase = train_or_showcase
        self.agent = agent
        self.exploration_agent = exploration_agent
        self.replay_buffer = replay_buffer
        self.time_steps = time_steps
        self.max_path_length = max_path_length
        self.permute_samples = permute_samples

        self.use_multiprocessing = use_multiprocessing
        self.use_data_normalization = use_data_normalization
        self.use_sac_data_normalization = use_sac_data_normalization
        self.num_workers = num_workers
        self.gpu_id = gpu_id
        self.scripted_policy = scripted_policy

        self.num_env_steps = 0

        if self.use_multiprocessing:
            ray.init(
                #log_to_driver=False
                # memory=1000 * 1024 * 1024,
                # object_store_memory=2500 * 1024 * 1024,
                # driver_object_store_memory=1000 * 1024 * 1024
            )

    def collect_data(self, tasks, train_test, compute_exploration_task_indicators=False, deterministic=False, max_samples=np.inf,
                     max_trajs=np.inf, max_trajs_exploration=0, animated=False, save_frames=False, return_distributions=False):
        """
        Distribute tasks over workers
        :return: Trajectories as a list of list of list (workers, tasks, (path_dict, num_env_steps))
        """
        tasks_per_worker = [[] for _ in range(self.num_workers)]
        counter = 0
        for task in tasks:
            if counter % self.num_workers == 0:
                counter = 0
            tasks_per_worker[counter].append(task)
            counter += 1

        if self.use_multiprocessing:
            # put on cpu before starting ray
            self.agent.to('cpu')
            self.agent.policy.to('cpu')#, 'policy')
            self.agent.encoder.to('cpu')
            workers = [RemoteRolloutWorker.remote(None, self.env_name, self.env_args, self.train_or_showcase,
                                                  self.agent, self.time_steps, self.max_path_length, self.permute_samples, self.gpu_id, self.scripted_policy,
                                                  self.use_multiprocessing, self.use_data_normalization, self.use_sac_data_normalization, self.replay_buffer.stats_dict,
                                                  task_list, self.env.tasks, self.env.train_tasks, self.env.test_tasks, self.exploration_agent, compute_exploration_task_indicators) for task_list in tasks_per_worker]
            results = ray.get([worker.obtain_samples_from_list.remote(train_test,
                deterministic=deterministic, max_samples=max_samples, max_trajs=max_trajs, max_trajs_exploration=max_trajs_exploration, animated=animated,
                save_frames=save_frames, return_distributions=return_distributions) for worker in workers])
        else:
            workers = [RolloutWorker(self.env, self.env_name, self.env_args, self.train_or_showcase,
                                     self.agent, self.time_steps, self.max_path_length, self.permute_samples, self.gpu_id, self.scripted_policy,
                                     self.use_multiprocessing, self.use_data_normalization, self.use_sac_data_normalization, self.replay_buffer.stats_dict,
                                     task_list, self.env.tasks, self.env.train_tasks, self.env.test_tasks, self.exploration_agent, compute_exploration_task_indicators) for task_list in tasks_per_worker]
            results = [[worker.obtain_samples(task, train_test,
                deterministic=deterministic, max_samples=max_samples, max_trajs=max_trajs, max_trajs_exploration=max_trajs_exploration, animated=animated,
                save_frames=save_frames, return_distributions=return_distributions
            ) for task in task_list] for worker, task_list in zip(workers, tasks_per_worker)]

        self.agent.to(ptu.device)
        self.agent.policy.to(ptu.device)#, 'policy')
        self.agent.encoder.to(ptu.device)
        return results

    def collect_replay_data(self, tasks, max_samples=np.inf, max_trajs=np.inf, max_trajs_exploration=0):
        """
        Run episodes on the environment and store the trajectories in the replay buffer
        :param tasks: List of tasks to run
        :param max_samples: Samples to collect per task
        :return: Number of environment steps executed
        """
        num_env_steps = 0
        results = self.collect_data(tasks, 'train',
                                    deterministic=False, max_samples=max_samples, max_trajs=max_trajs, max_trajs_exploration=max_trajs_exploration, animated=False)
        for worker in results:
            for task in worker:
                num_env_steps += self.replay_buffer.add_episode_group(task)
        return num_env_steps

    def evaluate(self, train_test, tasks, num_eval_trajectories, deterministic=True, animated=False, save_frames=False, log=True):
        results = self.collect_data(tasks, 'train_test', deterministic=deterministic, max_trajs=num_eval_trajectories,
                                    animated=animated, save_frames=save_frames)
        eval_statistics = OrderedDict()
        if log:
            deterministic_string = '_deterministic' if deterministic else '_non_deterministic'
            per_path_rewards = [np.sum(path["rewards"]) for worker in results for task in worker for path in task[0]]
            per_path_rewards = np.array(per_path_rewards)
            eval_average_reward = per_path_rewards.mean()
            eval_std_reward = per_path_rewards.std()
            eval_max_reward = per_path_rewards.max()
            eval_min_reward = per_path_rewards.min()
            eval_statistics[train_test + '_eval_avg_reward' + deterministic_string] = eval_average_reward
            eval_statistics[train_test + '_eval_std_reward' + deterministic_string] = eval_std_reward
            eval_statistics[train_test + '_eval_max_reward' + deterministic_string] = eval_max_reward
            eval_statistics[train_test + '_eval_min_reward' + deterministic_string] = eval_min_reward
            # success rates for meta world
            if "success" in results[0][0][0][0]["env_infos"][0]:
                success_values = np.array([[timestep["success"] for timestep in path["env_infos"]] for worker in results for task in worker for path in task[0]])
                episode_length = success_values.shape[1]
                success_types = np.full(success_values.shape[0], 'base')
                if 'success_type' in results[0][0][0][0]["env_infos"][0]:
                    success_types = np.array([path["env_infos"][0]["success_type"] for worker in results for task in worker for path in task[0]])
                    if not np.isin(success_types, ['base', 'end']).all():
                        raise ValueError('Unknown success_type provided by environment')

                success_trajectories = np.zeros(success_values.shape[0])
                # base (default): Successful if any transition is successful; For tasks that cannot be consistently fulfilled, e.g. jumping to height
                success_trajectories[success_types == 'base'] = np.sum(success_values[success_types == 'base'], axis=1) > 0
                # end: Successful if of the last 20% of transition, at least half are successful. For e.g. goal reaching tasks
                success_trajectories[success_types == 'end'] = np.sum(success_values[success_types == 'end', int(0.8*episode_length):], axis=1) > 0.1*episode_length
                success_rate = np.sum(success_trajectories.astype(int)) / success_values.shape[0]
                eval_statistics[train_test + '_eval_success_rate'] = success_rate
            if int(os.environ['DEBUG']) == 1:
                print(train_test + ":")
                print("Mean reward: " + str(eval_average_reward))
                print("Std reward: " + str(eval_std_reward))
                print("Max reward: " + str(eval_max_reward))
                print("Min reward: " + str(eval_min_reward))
            return eval_average_reward, eval_std_reward, eval_max_reward, eval_min_reward, eval_statistics
        else:
            return


class RolloutWorker:
    def __init__(self,
                 env,
                 env_name,
                 env_args,
                 train_or_showcase,
                 agent,
                 time_steps,
                 max_path_length,
                 permute_samples,
                 gpu_id,
                 scripted_policy,
                 use_multiprocessing,
                 use_data_normalization,
                 use_sac_data_normalization,
                 replay_buffer_stats_dict,
                 task_list,
                 env_tasks,
                 env_train_tasks,
                 env_test_tasks,
                 exploration_agent=None,
                 compute_exploration_task_indicators=False
                 ):
        if use_multiprocessing:
            environment = ENVS[env_name](**env_args)
            if env_args['use_normalized_env']:
                environment = NormalizedBoxEnv(environment)
            if train_or_showcase == 'showcase':
                environment = CameraWrapper(environment)
            self.env = environment
        else:
            self.env = env
        self.agent = agent
        self.exploration_agent = exploration_agent
        self.compute_exploration_task_indicators = compute_exploration_task_indicators
        self.time_steps = time_steps
        self.max_path_length = max_path_length
        self.permute_samples = permute_samples
        self.gpu_id = gpu_id
        self.scripted_policy = scripted_policy
        self.use_data_normalization = use_data_normalization
        self.use_sac_data_normalization = use_sac_data_normalization

        self.replay_buffer_stats_dict = replay_buffer_stats_dict
        self.task_list = task_list

        self.env.tasks = env_tasks
        self.env.train_tasks = env_train_tasks
        self.env.train_tasks = env_test_tasks

        self.action_space = self.env.action_space.low.size
        self.obs_space = self.env.observation_space.low.size
        self.context = None
        self.padding_mask = None

    def obtain_samples_from_list(self, train_test, deterministic=False, max_samples=np.inf, max_trajs=np.inf, max_trajs_exploration=0, animated=False, save_frames=False, return_distributions=False):
        results = []
        for task in self.task_list:
            result = self.obtain_samples(task, train_test, deterministic=deterministic, max_samples=max_samples, max_trajs=max_trajs, max_trajs_exploration=max_trajs_exploration, animated=animated, save_frames=save_frames, return_distributions=return_distributions)
            results.append(result)

        return results

    def obtain_samples(self, task, train_test, deterministic=False, max_samples=np.inf, max_trajs=np.inf, max_trajs_exploration=0, animated=False, save_frames=False, return_distributions=False):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        :return: list of trajectory dicts
            number of executed environment steps
        """

        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        paths = []
        n_steps_total = 0
        n_steps_total_exploration = 0
        n_trajs = 0
        n_trajs_exploration = 0
        while n_trajs_exploration < max_trajs_exploration:
            self.env.reset_task(task)
            self.env.set_meta_mode(train_test)
            path = self.rollout(deterministic=deterministic, max_path_length=self.max_path_length, animated=animated, save_frames=save_frames, return_distributions=return_distributions, use_exploration_agent=True)
            paths.append(path)
            n_steps_total_exploration += len(path['observations'])
            n_trajs_exploration += 1
        while n_steps_total < max_samples and n_trajs < max_trajs:
            self.env.reset_task(task)
            self.env.set_meta_mode(train_test)
            path = self.rollout(deterministic=deterministic, max_path_length=self.max_path_length if max_samples - n_steps_total > self.max_path_length else max_samples - n_steps_total, animated=animated, save_frames=save_frames, return_distributions=return_distributions)
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
        return paths, n_steps_total + n_steps_total_exploration

    def rollout(self, deterministic=False, max_path_length=np.inf, animated=False, save_frames=False, return_distributions=False, use_exploration_agent=False):
        if use_exploration_agent:
            if self.exploration_agent is None:
                raise RuntimeError('Trying to collect exploration trajectories, but no exploration agent was given.')
            current_agent = self.exploration_agent
        else:
            current_agent = self.agent

        observations = []
        task_indicators = []
        base_task_indicators = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        self.context = [torch.zeros((self.time_steps, self.obs_space)),
                        torch.zeros((self.time_steps, self.action_space)),
                        torch.zeros((self.time_steps, 1)),
                        torch.zeros((self.time_steps, self.obs_space))]
        self.padding_mask = np.ones((1, self.time_steps), dtype=bool)
        action_space = int(np.prod(self.env.action_space.shape))

        if self.scripted_policy:
            self.env.metaworld_env._partially_observable = False

        o = self.env.reset()
        next_o = None
        path_length = 0
        agent_info = {}

        #debug
        #true_task = torch.tensor([[1.0]])

        if animated:
            self.env.render()
        while path_length < max_path_length:
            o_input, agent_input, padding_mask = self.build_encoder_input(o, self.context, self.padding_mask)
            out = current_agent.get_action(agent_input, o_input, input_padding=padding_mask, deterministic=deterministic,
                                           z_debug=None, env=self.env, return_distributions=return_distributions,
                                           agent_info=agent_info)
            a = out[0]
            agent_info = out[1]
            task_indicator = out[2]
            base_task_indicator = out[3]
            if base_task_indicator is None and self.compute_exploration_task_indicators:
                # Exploration agents typically do not provide task indicators.
                # Get these by doing an additional run with the policy agent, but disregard the action
                # Use this only for visualization purposes; it is not necessary when filling the replay buffer,
                # since task indicators are recomputed during training/optimization
                out_supplementary = self.agent.get_action(agent_input, o_input, input_padding=padding_mask, deterministic=deterministic,
                                                          z_debug=None, env=self.env, return_distributions=return_distributions,
                                                          agent_info=agent_info)
                task_indicator = out_supplementary[2]
                base_task_indicator = out_supplementary[3]
                if 'latent_distribution' in out_supplementary[1].keys():
                    agent_info['latent_distribution'] = out_supplementary[1]['latent_distribution']
            next_o, r, d, env_info = self.env.step(a)
            self.update_context(o, a, np.array([r], dtype=np.float32), next_o)
            if self.scripted_policy:
                observations.append(np.hstack((o[0:6], np.zeros(6))))
            else:
                observations.append(o)
            task_indicators.append(task_indicator)
            base_task_indicators.append(base_task_indicator)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            #debug
            #true_task = torch.tensor([[1.0]]) if env_info['true_task'] == 1 else torch.tensor([[-1.0]])
            path_length += 1
            o = next_o
            if animated:
                self.env.render()
            if save_frames:
                from PIL import Image
                image = Image.fromarray(np.flipud(self.env.get_image(width=1600, height=1600))) # make even higher for better quality
                env_info['frame'] = image
            env_infos.append(env_info)
            if d:
                break

        next_o_input, agent_input, paddings_mask = self.build_encoder_input(next_o, self.context, self.padding_mask)
        _, _, next_task_indicator, next_base_task_indicator, *_ = \
            current_agent.get_action(agent_input, next_o_input, input_padding=padding_mask, deterministic=deterministic, env=self.env, agent_info=agent_info)
        if next_base_task_indicator is None and self.compute_exploration_task_indicators:
            _, _, next_task_indicator, next_base_task_indicator, *_ = self.agent.get_action(agent_input, next_o_input, input_padding=padding_mask,
                                                      deterministic=deterministic,
                                                      z_debug=None, env=self.env,
                                                      return_distributions=return_distributions,
                                                      agent_info=agent_info)
        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        task_indicators = np.array(task_indicators)
        base_task_indicators = np.array(base_task_indicators)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            task_indicators = np.expand_dim(task_indicators, 1)
            base_task_indicators = np.expand_dim(base_task_indicators, 1)
            next_o = np.array([next_o])
            next_task_indicator = np.array([next_task_indicator])
            next_base_task_indicator = np.array([next_base_task_indicator])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        next_task_indicators = np.vstack(
            (
                task_indicators[1:, :],
                np.expand_dims(next_task_indicator, 0)
            )
        )
        next_base_task_indicators = np.concatenate(
            (
                base_task_indicators[1:],
                np.expand_dims(next_base_task_indicator, 0)
            )
        )

        true_tasks = [env_infos[i]['true_task'] for i in range(len(env_infos))]

        return dict(
            observations=observations,
            task_indicators=task_indicators,
            base_task_indicators=base_task_indicators,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            next_task_indicators=next_task_indicators,
            next_base_task_indicators=next_base_task_indicators,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            true_tasks=np.array(true_tasks).reshape(-1, 1),
        )

    def update_context(self, o, a, r, next_o):
        if self.use_data_normalization and self.replay_buffer_stats_dict is not None:
            stats_dict = self.replay_buffer_stats_dict
            o = torch.from_numpy((o - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
            a = torch.from_numpy((a - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
            r = torch.from_numpy((r - stats_dict["rewards"]["mean"]) / (stats_dict["rewards"]["std"] + 1e-9))
            next_o = torch.from_numpy((next_o - stats_dict["next_observations"]["mean"]) / (stats_dict["next_observations"]["std"] + 1e-9))
        else:
            o = torch.from_numpy(o)
            a = torch.from_numpy(a)
            r = torch.from_numpy(r)
            next_o = torch.from_numpy(next_o)

        data = [o, a, r, next_o]
        data = [d.view(1, -1).float() for d in data]
        for i in range(len(self.context)):
            self.context[i] = torch.cat([self.context[i], data[i]], dim=0)[-self.time_steps:]
        self.padding_mask = np.concatenate([self.padding_mask[:, 1:], np.zeros((1, 1), dtype=bool)], axis=-1)

    def build_encoder_input(self, obs, context, padding_mask):
        encoder_input = [c.detach().clone() for c in context]
        padding_mask = np.copy(padding_mask)

        if np.sum(~padding_mask) == 0:
            # For the very first step of an episode, pretend that one padding point is actual data.
            # Slight inaccuracy, but avoids implementation problems of having no data at all.
            padding_mask[..., -1] = False

        if self.permute_samples:
            perm = torch.LongTensor(torch.randperm(encoder_input.shape[0]))
            encoder_input = [e[perm] for e in encoder_input]
            padding_mask = padding_mask[..., ptu.get_numpy(perm)]
        [e.unsqueeze_(0) for e in encoder_input]

        if self.use_sac_data_normalization and self.replay_buffer_stats_dict is not None:
            o_input = (obs - self.replay_buffer_stats_dict["observations"]["mean"]) \
                      / (self.replay_buffer_stats_dict["observations"]["std"] + 1e-9)
        else:
            o_input = obs

        encoder_input = [e.to(ptu.device) for e in encoder_input]
        return o_input, encoder_input, padding_mask


@ray.remote
class RemoteRolloutWorker(RolloutWorker):
    def __init__(self, env, env_name, env_args, train_or_showcase, agent, time_steps, max_path_length, permute_samples, gpu_id,  scripted_policy, use_multiprocessing, use_data_normalization, use_sac_data_normalization, replay_buffer_stats_dict, task_list, tasks, train_tasks, test_tasks, exploration_agent=None, compute_exploration_task_indicators=False):
        super().__init__(env, env_name, env_args, train_or_showcase, agent, time_steps, max_path_length, permute_samples, gpu_id,  scripted_policy, use_multiprocessing, use_data_normalization, use_sac_data_normalization, replay_buffer_stats_dict, task_list, tasks, train_tasks, test_tasks, exploration_agent, compute_exploration_task_indicators)

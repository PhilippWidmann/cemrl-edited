from collections import OrderedDict

import metaworld
import random
import glfw
import numpy as np
from meta_rand_envs.base import MetaEnvironment
from meta_rand_envs.metaworld_reach import ML1Reach
import meta_rand_envs.metaworld_benchmarks as mw_bench

# based on repo master from https://github.com/rlworkgroup/metaworld on commit: 2020/10/29 @ 11:17PM, title " Update pick-place-v2 scripted-policy success (#251)", id: 5bcc76e1d455b8de34a044475c9ea3979ca53e2d


class ObservableML1(metaworld.Benchmark):

    ENV_NAMES = metaworld._ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if not env_name in metaworld._env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = metaworld._env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = metaworld._env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = metaworld._make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        dict(partially_observable=False),  # Cheating here: Make the goal position observable, essentialy removing the need for a latent representation
                                        seed=seed)
        self._test_tasks = metaworld._make_tasks(
            self._test_classes, {env_name: args_kwargs},
            dict(partially_observable=False),
            seed=(seed + 1 if seed is not None else seed))


class MetaWorldEnv(MetaEnvironment):
    def __init__(self, *args, **kwargs):
        self.metaworld_env = None
        ml10or45 = kwargs['ml10or45']
        self.scripted = kwargs['scripted_policy']
        if ml10or45 == 10:
            if self.scripted:
                self.ml_env = mw_bench.ML10()
            else:
                self.ml_env = metaworld.ML10()
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'] / 10)
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'] / 5)
        elif ml10or45 == 45:
            self.ml_env = metaworld.ML45()
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'] / 45)
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'] / 5)
        elif ml10or45 == 1:
            self.ml_env = metaworld.ML1(kwargs['base_task'])
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'])
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'])
        elif ml10or45 == 3:
            self.ml_env = mw_bench.ML3()
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'] / 3)
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'])
        elif ml10or45 == '1_observable':
            self.ml_env = ObservableML1(kwargs['base_task'])
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'])
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'])
        elif ml10or45 == 'reach-special':
            if 'partially_observable' not in kwargs.keys():
                kwargs['partially_observable'] = True
            self.ml_env = ML1Reach(kwargs['base_task'], partially_observable=kwargs['partially_observable'])
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'])
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'])
        else:
            raise NotImplementedError

        self.name2number = {}
        counter = 0
        for name, env_cls in self.ml_env.train_classes.items():
            self.name2number[name] = counter
            counter += 1
        for name, env_cls in self.ml_env.test_classes.items():
            self.name2number[name] = counter
            counter += 1

        self.sample_tasks(num_train_tasks_per_base_task, num_test_tasks_per_base_task)

    def sample_tasks(self, num_train_tasks_per_base_task, num_test_tasks_per_base_task):
        self.train_tasks = []
        for name, env_cls in self.ml_env.train_classes.items():
            random.seed(0)
            tasks = random.sample([task for task in self.ml_env.train_tasks if task.env_name == name], num_train_tasks_per_base_task)
            self.train_tasks += tasks

        self.test_tasks = []
        for name, env_cls in self.ml_env.test_classes.items():
            random.seed(1)
            tasks = random.sample([task for task in self.ml_env.test_tasks if task.env_name == name], num_test_tasks_per_base_task)
            self.test_tasks += tasks

        self.tasks = self.train_tasks + self.test_tasks
        if self.scripted:
            self.train_tasks = self.train_tasks + self.test_tasks
        self.reset_task(0)

    def reset_task(self, idx):
        # close window to avoid mulitple windows open at once
        if hasattr(self, 'viewer'):
            self.close()

        task = self.tasks[idx]
        if task.env_name in self.ml_env.train_classes:
            self.metaworld_env = self.ml_env.train_classes[task.env_name]()
        elif task.env_name in self.ml_env.test_classes:
            self.metaworld_env = self.ml_env.test_classes[task.env_name]()

        self.metaworld_env.viewer_setup = self.viewer_setup
        self.metaworld_env.set_task(task)
        self.metaworld_env.reset()
        self.active_env_name = task.env_name
        self.reset()

    def step(self, action):
        ob, reward, done, info = self.metaworld_env.step(action)
        info['true_task'] = dict(base_task=self.name2number[self.active_env_name],
                                 specification=self.metaworld_env._target_pos.sum(),
                                 target=self.metaworld_env._target_pos,
                                 target_x=self.metaworld_env._target_pos[0])
        info['pos'] = self.metaworld_env.tcp_center
        info['pos_x'] = self.metaworld_env.tcp_center[0]
        return ob.astype(np.float32), reward, done, info

    def reset(self):
        unformated = self.metaworld_env.reset()
        return unformated.astype(np.float32)

    def viewer_setup(self):
        self.metaworld_env.viewer.cam.azimuth = -20
        self.metaworld_env.viewer.cam.elevation = -20

    def __getattr__(self, attrname):
        return getattr(self.metaworld_env, attrname)

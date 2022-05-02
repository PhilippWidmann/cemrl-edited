import colorsys
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
from meta_rand_envs.base import NonStationaryGoalTargetEnv


class HalfCheetahNonStationaryTargetEnv(NonStationaryGoalTargetEnv, MujocoEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.termination_possible = kwargs.get('termination_possible', False)
        # For debugging the exploration agent only
        self.exploration_reward = kwargs.get('exploration_reward', False)
        NonStationaryGoalTargetEnv.__init__(self, *args, **kwargs)
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        # should actually go into NonStationaryGoalTargetEnv, breaks abstraction
        self._init_geom_rgba = self.model.geom_rgba.copy()

        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.tasks = self.train_tasks + self.test_tasks

    def _compute_reward(self, action, xposafter):
        if not self.exploration_reward:
            vector_to_target = self.active_task - xposafter
            reward_run = -1.0 * abs(vector_to_target)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl + reward_run
            return reward, reward_ctrl, reward_run
        else:
            return xposafter, 0, xposafter

    def step(self, action):
        self.check_env_change()

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()

        vector_to_target = self.active_task - xposafter
        reward, reward_ctrl, reward_run = self._compute_reward(action, xposafter)
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=0, specification=self.active_task),
                                      vector_to_target=vector_to_target, pos=xposafter)

    # from pearl
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20

    def reset_task(self, idx):
        self.task = self.tasks[idx]
        self.active_task = self.task['target']
        self.reset_change_points()
        self.recolor()
        self.steps = 0
        self.reset()


class ObservableGoalHalfCheetahNonStationaryTargetEnv(HalfCheetahNonStationaryTargetEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([self.active_task])
        ]).astype(np.float32).flatten()


class HalfCheetahNonStationaryTargetNormalizedRewardEnv(HalfCheetahNonStationaryTargetEnv):
    def _compute_reward(self, action, xposafter):
        vector_to_target = self.active_task - xposafter
        reward_run = -1.0 * abs(vector_to_target)
        reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        reward = reward_ctrl + reward_run / np.abs(self.active_task)
        return reward, reward_ctrl, reward_run


class HalfCheetahNonStationaryTargetQuadraticRewardEnv(HalfCheetahNonStationaryTargetEnv):
    def _compute_reward(self, action, xposafter):
        vector_to_target = self.active_task - xposafter
        reward_run = -1.0 * vector_to_target**2
        reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        reward = reward_ctrl + reward_run
        return reward, reward_ctrl, reward_run


class HalfCheetahNonStationaryTargetForwardEnv(HalfCheetahNonStationaryTargetQuadraticRewardEnv):
    def change_active_task(self, *args, **kwargs):
        # Idea: If the task changes, the target will be further in front,
        # meaning we don't have to handle going in two directions
        if self.meta_mode == 'train':
            self.active_task = np.random.choice(self.train_tasks)['target'] + self.active_task
        elif self.meta_mode == 'test':
            self.active_task = np.random.choice(self.test_tasks)['target'] + self.active_task
        self.recolor()


class ObservableAbsGoalHalfCheetahNonStationaryTargetQuadraticRewardEnv(HalfCheetahNonStationaryTargetQuadraticRewardEnv):
    def _get_obs(self):
        starting_dist_to_target = abs(self.active_task)
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([starting_dist_to_target])
        ]).astype(np.float32).flatten()


class HalfCheetahNonStationaryTargetQuadraticRewardVariableStartEnv(HalfCheetahNonStationaryTargetQuadraticRewardEnv):
    def __init__(self, *args, **kwargs):
        self.possible_start_positions = kwargs['start_positions']
        self.active_start_position = None
        super().__init__(*args, **kwargs)

    def reset_model(self):
        qpos = self.init_qpos
        qpos[0] = self.active_start_position
        qpos = qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        targets = np.random.uniform(self.task_min_target, self.task_max_target, size=(num_tasks,))
        starts = np.random.choice(self.possible_start_positions, size=num_tasks, replace=True)
        tasks = [{'target': targets[i], 'start': starts[i]} for i in range(num_tasks)]
        return tasks

    def reset_task(self, idx):
        self.active_start_position = self.tasks[idx]['start']
        super().reset_task(idx)


class ObservableHalfCheetahNonStationaryTargetQuadraticRewardVariableStartEnv(HalfCheetahNonStationaryTargetQuadraticRewardVariableStartEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([self.active_task]),
            np.array([self.active_start_position])
        ]).astype(np.float32).flatten()


class ObservableGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv(HalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([self.active_task])
        ]).astype(np.float32).flatten()


class ObservableVecToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv(HalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def _get_obs(self):
        xpos = self.sim.data.qpos[0]
        vector_to_target = self.active_task - xpos
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([vector_to_target])
        ]).astype(np.float32).flatten()


class ObservableDistToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv(HalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def _get_obs(self):
        xpos = self.sim.data.qpos[0]
        dist_to_target = abs(self.active_task - xpos)
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([dist_to_target])
        ]).astype(np.float32).flatten()


class ObservableRewardHalfCheetahNonStationaryTargetNormalizedRewardEnv(HalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def _get_obs(self, reward=0):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([reward])
        ]).astype(np.float32).flatten()

    def step(self, action):
        self.check_env_change()

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        vector_to_target = self.active_task - xposafter
        reward, reward_ctrl, reward_run = self._compute_reward(action, xposafter)

        ob = self._get_obs(reward_run)
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=0, specification=self.active_task),
                                      vector_to_target=vector_to_target, pos=xposafter)


class ObservableDirToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv(HalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def _get_obs(self):
        xposafter = self.sim.data.qpos[0]
        vector_to_target = self.active_task - xposafter
        dir_to_target = np.sign(vector_to_target)
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
            np.array([dir_to_target])
        ]).astype(np.float32).flatten()

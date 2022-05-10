from collections import OrderedDict

import metaworld
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import SawyerReachEnvV2
from gym.spaces import Box
import numpy as np


class ML1Reach(metaworld.Benchmark):
    def __init__(self, env_name, seed=0, partially_observable=True):
        super().__init__()
        if not env_name in REACH_ENV_DICT.keys():
            raise ValueError(f"{env_name} is not a valid special-reach-environment")
        cls = REACH_ENV_DICT[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = dict(args=[], kwargs={'task_id': None,})

        self._train_tasks = metaworld._make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        dict(partially_observable=partially_observable),
                                        seed=seed)
        self._test_tasks = metaworld._make_tasks(
            self._test_classes, {env_name: args_kwargs},
            dict(partially_observable=partially_observable),
            seed=(seed + 1 if seed is not None else seed))


# Base class for a reach environment with differently sampled goal locations
class SawyerReachEnvV2AlternateGoal(SawyerReachEnvV2):
    def __init__(self, goal_low, goal_high):
        super().__init__()
        # code is taken from super().__init__, but need to overwrite variables
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    # Ensure that the goal is not too close to the starting point by resampling
    def reset_model(self):
        res = super().reset_model()
        if self.random_init:
            i = 0
            while np.linalg.norm(self._target_pos - self.tcp_center) < 0.15:
                res = super().reset_model()
                i += 1
                if i > 100:
                    raise RuntimeError('Cannot seem to sample a valid target point. '
                                       'Something in the logic/code must be wrong.')
        return res

    def step(self, action):
        ob, reward, done, info = super().step(action)
        # Todo: Find a better way (e.g. choose a new random train task for each episode)
        # This is a hack, since the object position can leak the target if the algorithm memorizes it for all train tasks
        ob[4:11] = 0
        ob[22:29] = 0
        return ob, reward, done, info


# Environment where the goal only lies on a line left/right from the initial position
class SawyerReachEnvV2Line(SawyerReachEnvV2AlternateGoal):
    def __init__(self):
        goal_low = (-0.5, 0.6, 0.15)
        goal_high = (0.5, 0.6, 0.15)
        super(SawyerReachEnvV2Line, self).__init__(goal_low, goal_high)


# Environment where the goal only lies on a line left/right from the initial position
# Additionally, the action space is restricted to only moving left/right
class SawyerReachEnvV2LineRestricted(SawyerReachEnvV2Line):
    def __init__(self):
        super(SawyerReachEnvV2LineRestricted, self).__init__()
        self.action_space = Box(
            np.array([-1]),
            np.array([+1]),
        )

    # Only left-right movement -> Pad other action dimensions with 0
    def step(self, action):
        if len(action) != 1:
            raise RuntimeError('Wrong action dimensionality')
        full_action = np.array([action[0], 0, 0, 0])
        return super().step(full_action)


class SawyerReachEnvV2LineRestrictedDistReward(SawyerReachEnvV2LineRestricted):
    # Overwrite the complicated reward function and use negative distance instead
    def compute_reward(self, actions, obs):
        _, dist, in_place = super().compute_reward(actions, obs)
        return -dist, dist, in_place


# Environment where the goal only lies on a halfline to the right from the initial position
class SawyerReachEnvV2Halfline(SawyerReachEnvV2AlternateGoal):
    def __init__(self):
        goal_low = (0.15, 0.6, 0.15)
        goal_high = (0.5, 0.6, 0.15)
        super(SawyerReachEnvV2Halfline, self).__init__(goal_low, goal_high)


# Environment where the goal only lies on a halfline to the right from the initial position
# Additionally, the action space is restricted to only moving left/right
class SawyerReachEnvV2HalflineRestricted(SawyerReachEnvV2Halfline):
    def __init__(self):
        super(SawyerReachEnvV2HalflineRestricted, self).__init__()
        self.action_space = Box(
            np.array([-1]),
            np.array([+1]),
        )

    # Only left-right movement -> Pad other action dimensions with 0
    def step(self, action):
        if len(action) != 1:
            raise RuntimeError('Wrong action dimensionality')
        full_action = np.array([action[0], 0, 0, 0])
        return super().step(full_action)


# Environment where the goal only lies on a halfline to the right from the initial position
class SawyerReachEnvV2HalflineDistReward(SawyerReachEnvV2AlternateGoal):
    def __init__(self):
        goal_low = (0.15, 0.6, 0.15)
        goal_high = (0.5, 0.6, 0.15)
        super(SawyerReachEnvV2HalflineDistReward, self).__init__(goal_low, goal_high)

    # Overwrite the complicated reward function and use negative distance instead
    def compute_reward(self, actions, obs):
        _, dist, in_place = super(SawyerReachEnvV2HalflineDistReward, self).compute_reward(actions, obs)
        return -dist, dist, in_place


# Environment where the goal only lies on a halfline to the right from the initial position
# Additionally, the action space is restricted to only moving left/right
class SawyerReachEnvV2HalflineRestrictedDistReward(SawyerReachEnvV2HalflineDistReward):
    def __init__(self):
        super(SawyerReachEnvV2HalflineRestrictedDistReward, self).__init__()
        self.action_space = Box(
            np.array([-1]),
            np.array([+1]),
        )

    # Only left-right movement -> Pad other action dimensions with 0
    def step(self, action):
        if len(action) != 1:
            raise RuntimeError('Wrong action dimensionality')
        full_action = np.array([action[0], 0, 0, 0])
        return super().step(full_action)


# Environment where the goal only lies on a horizontal plane around the initial point
class SawyerReachEnvV2Plane(SawyerReachEnvV2AlternateGoal):
    def __init__(self):
        goal_low = (-0.45, 0.4, 0.15)
        goal_high = (0.45, 0.8, 0.15)
        super(SawyerReachEnvV2Plane, self).__init__(goal_low, goal_high)


# Environment where the goal only lies on a line left/right from the initial position
# Additionally, the action space is restricted to only moving left/right
class SawyerReachEnvV2PlaneRestricted(SawyerReachEnvV2Plane):
    def __init__(self):
        super(SawyerReachEnvV2PlaneRestricted, self).__init__()
        self.action_space = Box(
            np.array([-1, -1]),
            np.array([+1, +1]),
        )

    # Only left-right movement -> Pad other action dimensions with 0
    def step(self, action):
        if len(action) != 2:
            raise RuntimeError('Wrong action dimensionality')
        full_action = np.array([action[0], action[1], 0, 0])
        return super().step(full_action)


REACH_ENV_DICT = {
    "reach-v2-line": SawyerReachEnvV2Line,
    "reach-v2-line-action-restricted": SawyerReachEnvV2LineRestricted,
    "reach-v2-line-action-restricted-distReward": SawyerReachEnvV2LineRestrictedDistReward,
    "reach-v2-plane": SawyerReachEnvV2Plane,
    "reach-v2-plane-action-restricted": SawyerReachEnvV2PlaneRestricted,
    "reach-v2-halfline": SawyerReachEnvV2Halfline,
    "reach-v2-halfline-action-restricted": SawyerReachEnvV2HalflineRestricted,
    "reach-v2-halfline-distReward": SawyerReachEnvV2HalflineDistReward,
    "reach-v2-halfline-action-restricted-distReward": SawyerReachEnvV2HalflineRestrictedDistReward,
}

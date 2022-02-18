import numpy as np
from meta_rand_envs.half_cheetah_non_stationary_velocity import HalfCheetahNonStationaryVelocityEnv


class HalfCheetahNonStationaryVelocityEnvNoPos(HalfCheetahNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()


class HalfCheetahNonStationaryVelocityEnvNoPosButVel(HalfCheetahNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_obs(self, vel=0):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            [vel],
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()

    def step(self, action):
        self.check_env_change()

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        forward_vel = (xposafter - xposbefore) / self.dt
        reward_run = -1.0 * abs(forward_vel - self.active_task)
        reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        reward = reward_run + reward_ctrl

        ob = self._get_obs(forward_vel)
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=0, specification=self.active_task), velocity=forward_vel)
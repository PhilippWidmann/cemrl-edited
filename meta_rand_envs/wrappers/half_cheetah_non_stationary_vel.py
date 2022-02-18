from meta_rand_envs.half_cheetah_non_stationary_velocity import HalfCheetahNonStationaryVelocityEnv, \
    ObservableGoalVelHalfCheetahNonStationaryVelocityEnv, ObservableRewardHalfCheetahNonStationaryVelocityEnv
from . import register_env
from meta_rand_envs.experimental_half_cheetah_non_stationary_velocity import HalfCheetahNonStationaryVelocityEnvNoPos, \
    HalfCheetahNonStationaryVelocityEnvNoPosButVel


@register_env('cheetah-stationary-vel')
@register_env('cheetah-stationary-vel-oneTimestep')
@register_env('cheetah-non-stationary-vel')
class HalfCheetahNonStationaryVelWrappedEnv(HalfCheetahNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryVelocityEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-vel-noPos')
class HalfCheetahNonStationaryVelNoPosWrappedEnv(HalfCheetahNonStationaryVelocityEnvNoPos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_env('cheetah-stationary-vel-noPosButVel')
class HalfCheetahNonStationaryVelNoPosButVelWrappedEnv(HalfCheetahNonStationaryVelocityEnvNoPosButVel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_env('cheetah-stationary-vel-observable')
@register_env('cheetah-non-stationary-vel-observable')
class ObservableGoalVelHalfCheetahNonStationaryVelWrappedEnv(ObservableGoalVelHalfCheetahNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        ObservableGoalVelHalfCheetahNonStationaryVelocityEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-vel-observable-reward')
class ObservableRewardHalfCheetahNonStationaryVelWrappedEnvObservable(ObservableRewardHalfCheetahNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        ObservableRewardHalfCheetahNonStationaryVelocityEnv.__init__(self, *args, **kwargs)


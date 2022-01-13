from meta_rand_envs.half_cheetah_non_stationary_velocity import HalfCheetahNonStationaryVelocityEnv, \
    HalfCheetahNonStationaryVelocityEnvObservable
from . import register_env

@register_env('cheetah-stationary-vel')
@register_env('cheetah-non-stationary-vel')
class HalfCheetahNonStationaryVelWrappedEnv(HalfCheetahNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryVelocityEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-vel-observable')
@register_env('cheetah-non-stationary-vel-observable')
class HalfCheetahNonStationaryVelWrappedEnvObservable(HalfCheetahNonStationaryVelocityEnvObservable):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryVelocityEnvObservable.__init__(self, *args, **kwargs)


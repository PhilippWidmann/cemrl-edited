from meta_rand_envs.half_cheetah_non_stationary_target import HalfCheetahNonStationaryTargetEnv
from . import register_env


@register_env('cheetah-stationary-target')
@register_env('cheetah-non-stationary-target')
class HalfCheetahNonStationaryTargetWrappedEnv(HalfCheetahNonStationaryTargetEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryTargetEnv.__init__(self, *args, **kwargs)

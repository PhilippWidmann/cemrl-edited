from meta_rand_envs.half_cheetah_non_stationary_direction import HalfCheetahNonStationaryDirectionEnv, \
    HalfCheetahNonStationaryDirectionEnvObservable, HalfCheetahNonStationaryDirectionEnvObservable2
from . import register_env

@register_env('cheetah-stationary-dir')
@register_env('cheetah-non-stationary-dir')
@register_env('cheetah-continuous-learning-dir')
class HalfCheetahNonStationaryDirWrappedEnv(HalfCheetahNonStationaryDirectionEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryDirectionEnv.__init__(self, *args, **kwargs)

@register_env('cheetah-stationary-dir-observable')
@register_env('cheetah-non-stationary-dir-observable')
class HalfCheetahNonStationaryDirObservableWrappedEnv(HalfCheetahNonStationaryDirectionEnvObservable):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryDirectionEnvObservable.__init__(self, *args, **kwargs)

@register_env('cheetah-stationary-dir-observable-2')
@register_env('cheetah-non-stationary-dir-observable-2')
class HalfCheetahNonStationaryDirObservableWrappedEnv2(HalfCheetahNonStationaryDirectionEnvObservable2):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryDirectionEnvObservable2.__init__(self, *args, **kwargs)
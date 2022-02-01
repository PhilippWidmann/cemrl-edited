from meta_rand_envs.half_cheetah_non_stationary_target import HalfCheetahNonStationaryTargetEnv, \
    HalfCheetahNonStationaryTargetNormalizedRewardEnv, ObservableGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableVecToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableDistToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableRewardHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableDirToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv
from . import register_env


@register_env('cheetah-stationary-target')
@register_env('cheetah-stationary-target-allT')
@register_env('cheetah-non-stationary-target')
class HalfCheetahNonStationaryTargetWrappedEnv(HalfCheetahNonStationaryTargetEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryTargetEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-normalizedReward')
@register_env('cheetah-stationary-target-normalizedReward-allT')
@register_env('cheetah-non-stationary-target-normalizedReward')
class HalfCheetahNonStationaryTargetNormalizedRewardWrappedEnv(HalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryTargetNormalizedRewardEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-normalizedReward-observable-goal')
class ObservableGoalHalfCheetahNonStationaryTargetNormalizedRewardWrappedEnv(ObservableGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def __init__(self, *args, **kwargs):
        ObservableGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-normalizedReward-observable-vecToGoal')
class ObservableVecToGoalHalfCheetahNonStationaryTargetNormalizedRewardWrappedEnv(ObservableVecToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def __init__(self, *args, **kwargs):
        ObservableVecToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-normalizedReward-observable-distToGoal')
class ObservableDistToGoalHalfCheetahNonStationaryTargetNormalizedRewardWrappedEnv(ObservableDistToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def __init__(self, *args, **kwargs):
        ObservableDistToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-normalizedReward-observable-reward')
class ObservableRewardHalfCheetahNonStationaryTargetNormalizedRewardWrappedEnv(ObservableRewardHalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def __init__(self, *args, **kwargs):
        ObservableRewardHalfCheetahNonStationaryTargetNormalizedRewardEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-normalizedReward-observable-dirToGoal')
class ObservableDirToGoalHalfCheetahNonStationaryTargetNormalizedRewardWrappedEnv(ObservableDirToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def __init__(self, *args, **kwargs):
        ObservableDirToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv.__init__(self, *args, **kwargs)

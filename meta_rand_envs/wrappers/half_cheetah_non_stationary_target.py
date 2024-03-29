from meta_rand_envs.half_cheetah_non_stationary_target import HalfCheetahNonStationaryTargetEnv, \
    HalfCheetahNonStationaryTargetNormalizedRewardEnv, ObservableGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableVecToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableDistToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableRewardHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableDirToGoalHalfCheetahNonStationaryTargetNormalizedRewardEnv, \
    ObservableGoalHalfCheetahNonStationaryTargetEnv, HalfCheetahNonStationaryTargetQuadraticRewardEnv, \
    HalfCheetahNonStationaryTargetQuadraticRewardVariableStartEnv, \
    ObservableHalfCheetahNonStationaryTargetQuadraticRewardVariableStartEnv, HalfCheetahNonStationaryTargetForwardEnv, \
    ObservableAbsGoalHalfCheetahNonStationaryTargetQuadraticRewardEnv
from . import register_env


@register_env('cheetah-stationary-target')
@register_env('cheetah-stationary-target-allT-noActions')
@register_env('cheetah-stationary-target-allT-specialInput')
@register_env('cheetah-stationary-target-allT')
@register_env('cheetah-stationary-targetTwosided')
@register_env('cheetah-stationary-targetTwosided-allT-GRU')
@register_env('cheetah-stationary-targetTwosided-allT-FCN')
@register_env('cheetah-stationary-targetTwosided-allT-Conv')
@register_env('cheetah-non-stationary-target')
class HalfCheetahNonStationaryTargetWrappedEnv(HalfCheetahNonStationaryTargetEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryTargetEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-non-stationary-target-forward')
class HalfCheetahNonStationaryTargetForwardWrappedEnv(HalfCheetahNonStationaryTargetForwardEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-quadraticReward')
@register_env('cheetah-stationary-target-quadraticReward-experimenting')
@register_env('cheetah-stationary-targetTwosided-quadraticReward')
@register_env('cheetah-stationary-target-quadraticReward-GRU')
@register_env('cheetah-stationary-target-quadraticReward-noActions')
@register_env('cheetah-stationary-target-quadraticReward-specialInput')
class HalfCheetahNonStationaryTargetQuadraticRewardWrappedEnv(HalfCheetahNonStationaryTargetQuadraticRewardEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryTargetEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-quadraticReward-observableAbsGoal')
class ObservableAbsGoalHalfCheetahNonStationaryTargetQuadraticRewardWrappedEnv(ObservableAbsGoalHalfCheetahNonStationaryTargetQuadraticRewardEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-normalizedReward')
@register_env('cheetah-stationary-target-normalizedReward-allT')
@register_env('cheetah-stationary-target-normalizedReward-allT-GRU')
@register_env('cheetah-stationary-target-normalizedReward-allT-Conv')
@register_env('cheetah-stationary-target-normalizedReward-allT-FCN')
@register_env('cheetah-stationary-targetTwosided-normalizedReward-allT')
@register_env('cheetah-stationary-targetTwosided-normalizedReward-allT-GRU')
@register_env('cheetah-stationary-targetTwosided-normalizedReward-allT-Conv')
@register_env('cheetah-stationary-targetTwosided-normalizedReward-allT-FCN')
@register_env('cheetah-non-stationary-target-normalizedReward')
class HalfCheetahNonStationaryTargetNormalizedRewardWrappedEnv(HalfCheetahNonStationaryTargetNormalizedRewardEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryTargetNormalizedRewardEnv.__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-targetVarStart-quadraticReward')
@register_env('cheetah-stationary-targetVarStart-quadraticReward-specialInput')
class HalfCheetahNonStationaryTargetVariableStartWrappedEnv(HalfCheetahNonStationaryTargetQuadraticRewardVariableStartEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


@register_env('observable-cheetah-stationary-targetVarStart-quadraticReward')
class ObservableHalfCheetahNonStationaryTargetVariableStartWrappedEnv(ObservableHalfCheetahNonStationaryTargetQuadraticRewardVariableStartEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


@register_env('cheetah-stationary-target-observable-goal')
class ObservableGoalHalfCheetahNonStationaryTargetWrappedEnv(ObservableGoalHalfCheetahNonStationaryTargetEnv):
    def __init__(self, *args, **kwargs):
        ObservableGoalHalfCheetahNonStationaryTargetEnv.__init__(self, *args, **kwargs)


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

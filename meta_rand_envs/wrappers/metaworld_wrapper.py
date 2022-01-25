from meta_rand_envs.metaworld import MetaWorldEnv
from . import register_env


@register_env('metaworld')
@register_env('metaworld-ml1-pick-and-place')
@register_env('metaworld-ml1-reach')
@register_env('metaworld-ml1-reach-observable')
@register_env('metaworld-ml1-reach-line')
@register_env('metaworld-ml1-reach-line-action-restricted')
@register_env('metaworld-ml1-reach-plane')
@register_env('metaworld-ml1-reach-plane-action-restricted')
@register_env('metaworld-ml1-reach-line-observable')
@register_env('metaworld-ml1-reach-line-action-restricted-observable')
@register_env('metaworld-ml1-reach-plane-observable')
@register_env('metaworld-ml1-reach-plane-action-restricted-observable')
@register_env('metaworld-ml1-reach-halfline')
@register_env('metaworld-ml1-reach-halfline-action-restricted')
@register_env('metaworld-ml1-reach-halfline-distReward')
@register_env('metaworld-ml1-reach-halfline-action-restricted-distReward')
@register_env('metaworld-ml1-push')
@register_env('metaworld-ml3')
@register_env('metaworld-ml10')
@register_env('metaworld-ml10-1')
@register_env('metaworld-ml10-2')
@register_env('metaworld-ml10-3')
@register_env('metaworld-ml10-scripted')
@register_env('metaworld-ml10-constrained')
@register_env('metaworld-ml10-constrained-1')
@register_env('metaworld-ml10-constrained-2')
@register_env('metaworld-ml45')
class MetaWorldWrappedEnv(MetaWorldEnv):
    def __init__(self, *args, **kwargs):
        MetaWorldEnv.__init__(self, *args, **kwargs)

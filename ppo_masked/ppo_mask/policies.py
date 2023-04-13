from ppo_masked.common.maskable.policies import (
    MaskableActorCriticCnnPolicy, MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy)

MlpPolicy = MaskableActorCriticPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
MultiInputPolicy = MaskableMultiInputActorCriticPolicy

import torch as th
from modules.models.policy import CustomCNN


def gen_policy_kwargs(nn_t = [512, 512, 512]) -> dict:
    #customCNN = CustomCNN()

    policy_kwargs: dict = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        activation_fn=th.nn.LeakyReLU, 
        net_arch=dict(pi=nn_t, vf=nn_t)
    )

    return policy_kwargs
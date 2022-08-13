import numpy as np
import os.path as osp
import torch
import zipfile
import pickle
import json
import typing
from typing import Callable, Dict, Optional, Text, Tuple, Union

from stable_baselines3 import PPO
from mocapact.sb3 import features_extractor

if typing.TYPE_CHECKING:
    import stable_baselines3

def get_exponential_fn(start: float, decay: float, lr_min: float):
    def func(progress_remaining: float) -> float:
        lr = start * np.exp(-(1-progress_remaining)*decay)
        return np.maximum(lr, lr_min)
    return func

def get_piecewise_constant_fn(start: float, decay: float):
    decay_squared = decay**2
    def func(progress_remaining: float) -> float:
        return (
            start                    if progress_remaining >= 2/3
            else start/decay         if progress_remaining >= 1/3
            else start/decay_squared
        )
    return func

def load_policy(
    model_path: Text,
    observable_keys: Union[Tuple[Text], Dict[Text, Text]],
    rl_alg_type: Callable[..., 'stable_baselines3.BaseAlgorithm'] = PPO,
    device: Union[torch.device, str] = 'auto',
    seed: Optional[int] = None,
) -> 'stable_baselines3.BaseAlgorithm':
    """
    Loads a Stable-Baselines policy, which consists of a zip file of parameters
    and a pickled VecNormalize environment for the observation statistics.
    """
    # Normalization statistics
    with open(osp.join(model_path, 'vecnormalize.pkl'), 'rb') as f:
        norm_env = pickle.load(f)
        obs_stats = norm_env.obs_rms

    # Set up model
    with zipfile.ZipFile(osp.join(model_path, 'best_model.zip')) as archive:
        json_string = archive.read("data").decode()
        json_dict = json.loads(json_string)
        policy_kwargs = {k: v for k, v in json_dict['policy_kwargs'].items() if not k.startswith(':')}
        if 'Tanh' in policy_kwargs['activation_fn']:
            policy_kwargs['activation_fn'] = torch.nn.Tanh
        elif 'ReLU' in policy_kwargs['activation_fn']:
            policy_kwargs['activation_fn'] = torch.nn.ReLU
        else:
            policy_kwargs['activation_fn'] = torch.nn.ELU
    policy_kwargs['features_extractor_class'] = features_extractor.CmuHumanoidFeaturesExtractor
    policy_kwargs['features_extractor_kwargs'] = dict(
        observable_keys=observable_keys,
        obs_rms=obs_stats
    )

    model = rl_alg_type.load(
        osp.join(model_path, 'best_model.zip'),
        device=device,
        custom_objects=dict(policy_kwargs=policy_kwargs, learning_rate=0., clip_range=0., seed=seed)
    )

    return model

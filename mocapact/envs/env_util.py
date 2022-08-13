"""
Adaptation of Stable-Baseline's `make_vec_env`. Allows for wrapping
with a custom VecMonitor, in our case a VecMonitor that also monitors
normalized episode reward and length.
"""
import os
import numpy as np
from typing import Any, Callable, Dict, Optional, Type, Union
import gym

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecEnv


def make_vec_env(
    env_id: Type[gym.Env],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    vec_monitor_cls: Callable[..., VecMonitor] = VecMonitor,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.
    :param env_id: the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the environment constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            # Used to suppress the DeprecationWarning
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            if seed is not None:
                if 'environment_kwargs' not in env_kwargs:
                    env_kwargs['environment_kwargs'] = {}
                rng = np.random.RandomState(seed=seed+rank)
                env_kwargs['environment_kwargs']['random_state'] = rng
            env = env_id(**env_kwargs)
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

    # Create the monitor folder if needed
    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)
    # Wrap the env in a Monitor wrapper
    # to have additional training information
    env = vec_monitor_cls(env, filename=monitor_dir, **monitor_kwargs)

    if seed is not None:
        set_random_seed(seed)

    return env

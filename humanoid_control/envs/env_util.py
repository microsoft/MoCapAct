import os
import numpy as np
from typing import Any, Callable, Dict, Optional, Type, Union
import gym

from dm_control.locomotion.tasks.reference_pose import types

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecEnv, VecNormalize, VecVideoRecorder

from humanoid_control import observables
from humanoid_control.sb3 import wrappers
from humanoid_control.envs import tracking

def make_env(
    seed=0,
    clip_ids=[],
    start_steps=[0],
    end_steps=[0],
    min_steps=10,
    training=True,
    act_noise=0.,
    always_init_at_clip_start=False,
    record_video=False,
    video_folder=None,
    n_workers=4,
    termination_error_threshold=float('inf'),
    gamma=0.95,
    normalize_obs=True,
    normalize_rew=True
):
    dataset = types.ClipCollection(
        ids=clip_ids,
        start_steps=start_steps,
        end_steps=end_steps
    )
    task_kwargs = dict(
        reward_type='comic',
        min_steps=min_steps - 1,
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=termination_error_threshold
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=(0,),
        act_noise=act_noise,
        task_kwargs=task_kwargs
    )
    env = make_vec_env(
        env_id=tracking.MocapTrackingGymEnv,
        n_envs=n_workers,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_monitor_cls=wrappers.MocapTrackingVecMonitor
    )
    if record_video and video_folder:
        env = VecVideoRecorder(env, video_folder,
                               record_video_trigger=lambda x: x >= 0,
                               video_length=float('inf'))
    env = VecNormalize(env, training=training, gamma=gamma,
                       norm_obs=normalize_obs,
                       norm_reward=normalize_rew,
                       norm_obs_keys=observables.MULTI_CLIP_OBSERVABLES_SANS_ID)
    return env

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
    vec_monitor_cls=VecMonitor,
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

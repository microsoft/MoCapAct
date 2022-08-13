"""
Wraps the dm_control environment and task into a Gym env. The task assumes
the presence of a CMU position-controlled humanoid.

Adapted from:
https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
"""

import os.path as osp
import numpy as np
import tree
import mujoco

from typing import Any, Callable, Dict, Optional, Text, Tuple
from dm_env import TimeStep
from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.walkers import initializers
from dm_control.suite.wrappers import action_noise
from gym import core
from gym import spaces

class StandInitializer(initializers.WalkerInitializer):
    def __init__(self):
        ref_path = cmu_mocap_data.get_path_for_cmu(version='2020')
        mocap_loader = loader.HDF5TrajectoryLoader(ref_path)
        trajectory = mocap_loader.get_trajectory('CMU_040_12')
        clip_reference_features = trajectory.as_dict()
        clip_reference_features = tracking._strip_reference_prefix(clip_reference_features, 'walker/')
        self._stand_features = tree.map_structure(lambda x: x[0], clip_reference_features)

    def initialize_pose(self, physics, walker, random_state):
        del random_state
        utils.set_walker_from_features(physics, walker, self._stand_features)
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)


class DmControlWrapper(core.Env):
    """
    Wraps the dm_control environment and task into a Gym env. The task assumes
    the presence of a CMU position-controlled humanoid.

    Adapted from:
    https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
    """

    metadata = {"render.modes": ["rgb_array"], "videos.frames_per_second": 30}

    def __init__(
        self,
        task_type: Callable[..., composer.Task],
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        act_noise: float = 0.,
        arena_size: Tuple[float, float] = (8., 8.),

        # for rendering
        width: int = 640,
        height: int = 480,
        camera_id: int = 3
    ):
        """
        task_kwargs: passed to the task constructor
        environment_kwargs: passed to composer.Environment constructor
        """
        task_kwargs = task_kwargs or dict()
        environment_kwargs = environment_kwargs or dict()

        # create task
        self._env = self._create_env(
            task_type,
            task_kwargs,
            environment_kwargs,
            act_noise=act_noise,
            arena_size=arena_size
        )
        self._original_rng_state = self._env.random_state.get_state()

        # Set observation and actions spaces
        self._observation_space = self._create_observation_space()
        action_spec = self._env.action_spec()
        dtype = np.float32
        self._action_space = spaces.Box(
            low=action_spec.minimum.astype(dtype),
            high=action_spec.maximum.astype(dtype),
            shape=action_spec.shape,
            dtype=dtype
        )

        # set seed
        self.seed()

        self._height = height
        self._width = width
        self._camera_id = camera_id

    @staticmethod
    def make_env_constructor(task_type: Callable[..., composer.Task]):
        return lambda *args, **kwargs: DmControlWrapper(task_type, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def dm_env(self) -> composer.Environment:
        return self._env

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space
    
    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def np_random(self):
        return self._env.random_state

    def seed(self, seed: Optional[int] = None):
        if seed:
            srng = np.random.RandomState(seed=seed)
            self._env.random_state.set_state(srng.get_state())
        else:
            self._env.random_state.set_state(self._original_rng_state)
        return self._env.random_state.get_state()[1]

    def _create_env(
        self,
        task_type,
        task_kwargs,
        environment_kwargs,
        act_noise=0.,
        arena_size=(8., 8.)
    ) -> composer.Environment:
        walker = self._get_walker()
        arena = self._get_arena(arena_size)
        task = task_type(
            walker,
            arena,
            **task_kwargs
        )
        env = composer.Environment(
            task=task,
            **environment_kwargs
        )
        task.random = env.random_state # for action noise
        if act_noise > 0.:
            env = action_noise.Wrapper(env, scale=act_noise/2)

        return env

    def _get_walker(self):
        directory = osp.dirname(osp.abspath(__file__))
        initializer = StandInitializer()
        return cmu_humanoid.CMUHumanoidPositionControlledV2020(initializer=initializer)

    def _get_arena(self, arena_size):
        return floors.Floor(arena_size)

    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                if np.prod(v.shape) > 0:
                    obs_spaces[k] = spaces.Box(
                        -np.infty,
                        np.infty,
                        shape=(np.prod(v.shape),),
                        dtype=np.float32
                    )
            elif v.dtype == np.uint8:
                tmp = v.generate_value()
                obs_spaces[k] = spaces.Box(
                    v.minimum.item(),
                    v.maximum.item(),
                    shape=tmp.shape,
                    dtype=np.uint8
                )
        return spaces.Dict(obs_spaces)

    def get_observation(self, time_step: TimeStep) -> Dict[str, np.ndarray]:
        dm_obs = time_step.observation
        obs = dict()
        for k in self.observation_space.spaces:
            if self.observation_space[k].dtype == np.uint8: # image
                obs[k] = dm_obs[k].squeeze()
            else:
                obs[k] = dm_obs[k].ravel().astype(self.observation_space[k].dtype)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0.
        done = time_step.last()
        obs = self.get_observation(time_step)
        info = dict(
            internal_state=self._env.physics.get_state().copy(),
            discount=time_step.discount
        )
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        time_step = self._env.reset()
        return self.get_observation(time_step)

    def render(
        self,
        mode: Text = 'rgb_array',
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_id: Optional[int] = None
    ) -> np.ndarray:
        assert mode == 'rgb_array', "This wrapper only supports rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

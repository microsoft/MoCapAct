import numpy as np
from gym import core, spaces

from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.suite.wrappers import action_noise

from humanoid_control.tasks import tracking


class MocapTrackingGymEnv(core.Env):
    """
    Wraps the MultiClipMocapTracking into a Gym env.

    Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py

    task_kwargs: passed to MultiClipMocapTracking constructor
    environment_kwargs: passed to composer's Environment constructor
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        dataset=None,
        ref_steps=(0,),
        act_noise=0.,
        task_kwargs=None,
        environment_kwargs=None,

        # for rendering
        width=640,
        height=480,
        camera_id=3
    ):
        dataset = dataset or types.ClipCollection(ids=['CMU_016_22'])
        task_kwargs = task_kwargs or dict()
        environment_kwargs = environment_kwargs or dict()

        # create task
        self._env = self._create_env(ref_steps, dataset, act_noise, task_kwargs, environment_kwargs)
        self._rng_state = self._env.random_state.get_state()

        # true and normalized action spaces
        self._action_space = spaces.Box(
            low=np.float32(-1.),
            high=np.float32(1.),
            shape=self._env.action_spec().shape,
            dtype=np.float32
        )

        # create observation space
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.int64:  # clip ID
                obs_spaces[k] = spaces.Discrete(len(dataset.ids))
            elif np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(-np.infty, np.infty, shape=(np.prod(v.shape),), dtype=np.float32)
        self._observation_space = spaces.Dict(obs_spaces)

        # set seed
        self.seed()
        self.np_random = self._env.random_state

        self._height = height
        self._width = width
        self._camera_id = camera_id

        self._current_joint_pos = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = time_step.observation
        return {k: obs[k].ravel() for k in self.observation_space.spaces}

    def _create_env(self, ref_steps, dataset, act_noise, task_kwargs, environment_kwargs):
        walker_type = cmu_humanoid.CMUHumanoidPositionControlledV2020
        arena = floors.Floor()
        task = tracking.MultiClipMocapTracking(
            walker_type,
            arena,
            cmu_mocap_data.get_path_for_cmu(version='2020'),
            ref_steps,
            dataset,
            **task_kwargs
        )
        env = composer.Environment(
            task=task,
            **environment_kwargs
        )
        task.random = env.random_state  # for action noise
        if act_noise > 0:
            env = action_noise.Wrapper(env, scale=act_noise / 2)

        return env

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed=None):
        self._env.random_state.set_state(self._rng_state)
        return self._env.random_state.get_state()[1]

    def step(self, action):
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0.
        done = time_step.last()
        obs = self._get_obs(time_step)
        info = dict(internal_state=self._env.physics.get_state().copy(),
                    time_in_clip=time_step.observation['walker/time_in_clip'].item(),
                    start_time_in_clip=self._start_time_in_clip,
                    last_time_in_clip=self._last_time_in_clip,
                    discount=time_step.discount)
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        self._start_time_in_clip = time_step.observation['walker/time_in_clip'].item()
        self._last_time_in_clip = self._env.task._last_step / (len(self._env.task._clip_reference_features['joints'])-1)
        return self._get_obs(time_step)

    # pylint: disable=arguments-differ
    def render(self, mode='rgb_array', height=None, width=None, camera_id=None):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

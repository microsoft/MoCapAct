"""Task for a walker to reach standing height."""

import numpy as np
import tree
from gym import core
from gym import spaces
import mujoco

from dm_control import composer
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks.reference_pose.tracking import _strip_reference_prefix
from dm_control.locomotion.tasks.reference_pose.tracking import DEFAULT_PHYSICS_TIMESTEP
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.walkers import initializers
from dm_control.mujoco.wrapper import mjbindings

STANDING_HEIGHT = 1.8

class StandUpInitalizer(initializers.WalkerInitializer):
    def __init__(self):
        ref_path = cmu_mocap_data.get_path_for_cmu('2020')
        self._get_up_keys = ['CMU_139_16', 'CMU_139_17', 'CMU_139_18',
                             'CMU_140_01', 'CMU_140_02', 'CMU_140_08', 'CMU_140_09']
        #self._get_up_keys = ['CMU_139_16']
        self._stand_mocap_key = 'CMU_040_12'
        self._stand_mocap_key = 'CMU_016_22' # TODO: remove
        self._loader = loader.HDF5TrajectoryLoader(ref_path)

        trajectory = self._loader.get_trajectory(self._stand_mocap_key)
        clip_reference_features = trajectory.as_dict()
        clip_reference_features = _strip_reference_prefix(clip_reference_features, 'walker/')
        self._stand_features = tree.map_structure(lambda x: x[0], clip_reference_features)

    def initialize_pose(self, physics, walker, random_state):
        if random_state.rand() < 0: #0.5: # lying on ground
            is_stand = False
            mocap_key = random_state.choice(self._get_up_keys)
            start = random_state.choice(range(30))
            trajectory = self._loader.get_trajectory(mocap_key)
            clip_reference_features = trajectory.as_dict()
            clip_reference_features = _strip_reference_prefix(clip_reference_features, 'walker/')
            timestep_features = tree.map_structure(lambda x: x[start], clip_reference_features)
        else: # standing off ground
            is_stand = True
            timestep_features = self._stand_features
        utils.set_walker_from_features(physics, walker, timestep_features)
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
        if is_stand:
            height_perturb = random_state.uniform(0.1, 0.25)
            height_perturb = 0. # TODO: remove
            walker.shift_pose(physics, position=[0, 0, height_perturb])

class StandUp(composer.Task):
    def __init__(
        self,
        walker,
        arena,
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=0.03
    ):
        self._arena = arena
        self._walker = walker
        self._walker.create_root_joints(self._arena.attach(self._walker))

        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        enabled_observables.append(self._walker.observables.sensors_touch)
        for obs in enabled_observables:
            obs.enabled = True

        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)

    @property
    def root_entity(self):
        return self._arena

    def get_reward(self, physics):
        head_height = self._walker.observables.head_height(physics)
        diff = head_height - STANDING_HEIGHT
        return np.exp(-diff**2)

    def initialize_episode(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

    def before_step(self, physics, action, random_state):
        self._walker.apply_action(physics, action, random_state)

class StandUpGymEnv(core.Env):
    def __init__(
        self,
        seed=0,
        extra_obs_space=None,
        # for rendering
        width=640,
        height=480,
        camera_id=3
    ):
        # create task
        self._seed = seed
        self._env = self._create_env()
        self._rng_state = self.np_random.get_state()

        self._action_space = spaces.Box(
            low=np.float32(-1.),
            high=np.float32(1.),
            shape=self._env.action_spec().shape,
            dtype=np.float32
        )

        self._max_episode_steps = 270
        self._elapsed_steps = None

        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.int64: # clip ID
                continue
            elif np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(-np.infty, np.infty, shape=(np.prod(v.shape),), dtype=np.float32)
        if extra_obs_space:
            for k, space in extra_obs_space.items():
                if k not in obs_spaces:
                    obs_spaces[k] = space
        self._observation_space = spaces.Dict(obs_spaces)

        self.seed()

        self._width = width
        self._height = height
        self._camera_id = camera_id

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = {}
        for k, space in self.observation_space.spaces.items():
            if k in time_step.observation:
                obs[k] = time_step.observation[k].ravel()
            else:
                obs[k] = np.full(space.shape, np.nan)
        return obs

    def _create_env(self):
       initializer = StandUpInitalizer()
       walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(initializer=initializer)
       arena = floors.Floor()
       task = StandUp(walker=walker, arena=arena)
       rng = np.random.RandomState(seed=self._seed)
       env = composer.Environment(task=task, random_state=rng)
       return env

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def np_random(self):
        return self._env.random_state

    def seed(self, seed=None):
        self._action_space.seed(self._seed)
        self._env.random_state.set_state(self._rng_state)
        return self._env.random_state.get_state()[1]

    def step(self, action):
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        self._elapsed_steps += 1
        reward = time_step.reward or 0.
        done = time_step.last() or self._elapsed_steps >= self._max_episode_steps
        obs = self._get_obs(time_step)
        info = dict(discount=time_step.discount)
        return obs, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        time_step = self._env.reset()
        return self._get_obs(time_step)

    def render(self, mode='rgb_array', height=None, width=None, camera_id=None):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

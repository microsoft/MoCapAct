import numpy as np
from gym import spaces
from typing import Any, Dict, Optional, Tuple

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.walkers import cmu_humanoid

from humanoid_control.envs import dm_control_wrapper
from humanoid_control.tasks import tracking

class MocapTrackingGymEnv(dm_control_wrapper.DmControlWrapper):
    """
    Wraps the MultiClipMocapTracking into a Gym env.
    """

    def __init__(
        self,
        dataset: Optional[types.ClipCollection] = None,
        ref_steps: Tuple[int] = (0,),
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        act_noise: float = 0.,

        # for rendering
        width: int = 640,
        height: int = 480,
        camera_id: int = 3
    ):
        self._dataset = dataset or types.ClipCollection(ids=['CMU_016_22'])
        task_kwargs = task_kwargs or dict()
        task_kwargs['ref_path'] = cmu_mocap_data.get_path_for_cmu(version='2020')
        task_kwargs['dataset'] = self._dataset
        task_kwargs['ref_steps'] = ref_steps
        environment_kwargs = environment_kwargs or dict()
        super().__init__(
            tracking.MultiClipMocapTracking,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            act_noise=act_noise,
            width=width,
            height=height,
            camera_id=camera_id
        )

    def _get_walker(self):
        return cmu_humanoid.CMUHumanoidPositionControlledV2020

    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.int64: # clip ID
                obs_spaces[k] = spaces.Discrete(len(self._dataset.ids))
            elif np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(
                    -np.infty,
                    np.infty,
                    shape=(np.prod(v.shape),),
                    dtype=np.float32
                )
        return spaces.Dict(obs_spaces)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        info['time_in_clip'] = obs['walker/time_in_clip'].item()
        info['start_time_in_clip'] = self._start_time_in_clip
        info['last_time_in_clip'] = self._last_time_in_clip
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self.get_observation(time_step)
        self._start_time_in_clip = obs['walker/time_in_clip'].item()
        self._last_time_in_clip = self._env.task._last_step / (len(self._env.task._clip_reference_features['joints'])-1)
        return obs

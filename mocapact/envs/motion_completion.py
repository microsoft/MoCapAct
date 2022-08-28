import numpy as np
from pathlib import Path
from gym import spaces
from typing import Any, Dict, Optional, Tuple, Union
from dm_control.locomotion.tasks.reference_pose import types

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.walkers import cmu_humanoid

from mocapact.envs import dm_control_wrapper
from mocapact.tasks import motion_completion

class MotionCompletionGymEnv(dm_control_wrapper.DmControlWrapper):
    def __init__(
        self,
        dataset: types.ClipCollection,
        ref_steps: Tuple[int] = (0,),
        mocap_path: Optional[Union[str, Path]] = None,
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        include_clip_id: bool = False,

        # for rendering
        width: int = 640,
        height: int = 480,
        camera_id: int = 3
    ):
        self._dataset = dataset
        task_kwargs = task_kwargs or dict()
        task_kwargs['ref_path'] = mocap_path if mocap_path else cmu_mocap_data.get_path_for_cmu(version='2020')
        task_kwargs['dataset'] = self._dataset
        task_kwargs['ref_steps'] = ref_steps
        self._include_clip_id = include_clip_id
        super().__init__(
            motion_completion.MotionCompletion,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            act_noise=0.,
            arena_size=(100., 100.),
            width=width,
            height=height,
            camera_id=camera_id
        )

    def _get_walker(self):
        return cmu_humanoid.CMUHumanoidPositionControlledV2020
        
    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(
                    -np.infty,
                    np.infty,
                    shape=(np.prod(v.shape),),
                    dtype=np.float32
                )
            elif k == 'walker/clip_id' and self._include_clip_id:
                obs_spaces[k] = spaces.Discrete(len(self._dataset.ids))
        return spaces.Dict(obs_spaces)


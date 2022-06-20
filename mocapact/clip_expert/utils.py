import numpy as np
from typing import Optional, Text
from dm_control.locomotion.tasks.reference_pose import types

def make_env_kwargs(
    clip_id: Text,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    min_steps: int = 10,
    ghost_offset: float = 1.,
    always_init_at_clip_start: bool = False,
    termination_error_threshold: float = 0.3,
    act_noise: float = 0.1
):
    """
    Gives the environment kwargs used to construct the
    mocap tracking gym environment for the clip expert.
    """
    dataset = types.ClipCollection(
        ids=[clip_id],
        start_steps=[start_step],
        end_steps=[end_step]
    )
    task_kwargs = dict(
        reward_type='comic',
        min_steps=min_steps-1,
        ghost_offset=np.array([ghost_offset, 0., 0.]),
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=termination_error_threshold
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=(0,),
        act_noise=act_noise,
        task_kwargs=task_kwargs
    )

    return env_kwargs

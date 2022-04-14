from typing import Any, Callable, Optional, Sequence, Text, Union
import numpy as np

import typing
if typing.TYPE_CHECKING:
    from dm_control.locomotion.walkers import legacy_base
    from dm_control import mjcf

from dm_control import composer
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils

class MultiClipMocapTracking(tracking.MultiClipMocapTracking):
    """
    Same as inherited class, except with option to override the
    maximum number of steps in an episode. If the number of steps
    is longer than a clip, the ghost is kept frozen at the end of
    the clip.
    """

    def __init__(
        self,
        walker: Callable[..., 'legacy_base.Walker'],
        arena: composer.Arena,
        ref_path: Text,
        ref_steps: Sequence[int],
        dataset: Union[Text, Sequence[Any]],
        termination_error_threshold: float = 0.3,
        prop_termination_error_threshold: float = 0.1,
        min_steps: int = 10,
        max_steps_override: Optional[Union[int, float]] = None,
        reward_type: Text = 'termination_reward',
        physics_timestep: float = tracking.DEFAULT_PHYSICS_TIMESTEP,
        always_init_at_clip_start: bool = False,
        proto_modifier: Optional[Any] = None,
        prop_factory: Optional[Any] = None,
        disable_props: bool = True,
        ghost_offset: Optional[Sequence[Union[int, float]]] = None,
        body_error_multiplier: Union[int, float] = 1,
        actuator_force_coeff: float = 0.015,
        enabled_reference_observables: Optional[Sequence[Text]] = None,
    ):
        super().__init__(
            walker,
            arena,
            ref_path,
            ref_steps,
            dataset,
            termination_error_threshold,
            prop_termination_error_threshold,
            min_steps,
            reward_type,
            physics_timestep,
            always_init_at_clip_start,
            proto_modifier,
            prop_factory,
            disable_props,
            ghost_offset,
            body_error_multiplier,
            actuator_force_coeff,
            enabled_reference_observables
        )
        
        self._max_steps_override = max_steps_override
        self._time_step_override = None

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)
        self._time_step_override = 0

    def after_step(self, physics: 'mjcf.Physics', random_state):
        tracking.ReferencePosesTask.after_step(self, physics, random_state)
        self._time_step += 1
        self._time_step_override += 1

        # Update the walker's data for this timestep.
        self._walker_features = utils.get_features(
            physics,
            self._walker,
            props=self._props
        )
        # features for default error
        self._walker_joints = np.array(physics.bind(self._walker.mocap_joints).qpos) # pytype: disable=attribute-error

        self._current_reference_features = {
            k: v[self._time_step].copy()
            for k, v in self._clip_reference_features.items()
        }

        # Error.
        self._compute_termination_error()

        # Terminate based on the error.
        self._end_mocap = self._time_step == self._last_step
        if self._max_steps_override is not None:
            if self._end_mocap and self._time_step_override < self._max_steps_override:
                self._time_step -= 1
                self._end_mocap = False
            elif not self._end_mocap and self._time_step_override >= self._max_steps_override:
                self._end_mocap = True
        
        self._reference_observations.update(self.get_all_reference_observations(physics))

        self._update_ghost(physics)

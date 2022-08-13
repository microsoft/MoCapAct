"""
The motion completion task.
"""
from typing import Any, Callable, Optional, Sequence, Text, Union
import numpy as np

import typing
if typing.TYPE_CHECKING:
    from dm_control.locomotion.walkers import legacy_base
    from dm_control import mjcf

from dm_control import composer
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils

HUMANOID_PROMPT_COLOR = (170/255, 74/255, 68/255, 1.)
HUMANOID_COMPLETION_COLOR = (0.7, 0.5, 0.3, 1.)

class MotionCompletion(tracking.MultiClipMocapTracking):
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
        min_steps: int = 10,
        max_steps: Optional[int] = None,
        steps_before_color_change: Optional[int] = 32,
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
            termination_error_threshold=termination_error_threshold,
            prop_termination_error_threshold=float('inf'),
            min_steps=min_steps,
            reward_type='comic',
            physics_timestep=physics_timestep,
            always_init_at_clip_start=always_init_at_clip_start,
            proto_modifier=proto_modifier,
            prop_factory=prop_factory,
            disable_props=disable_props,
            ghost_offset=ghost_offset,
            body_error_multiplier=body_error_multiplier,
            actuator_force_coeff=actuator_force_coeff,
            enabled_reference_observables=enabled_reference_observables
        )
        
        self._max_steps = max_steps
        self._steps_before_color_change = steps_before_color_change
        self._time_step_override = None
        self._start_step = None

    def _is_disallowed_contact(self, contact):
        set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def initialize_episode(self, physics: 'mjcf.Physics', random_state: np.random.RandomState):
        super().initialize_episode(physics, random_state)
        colors = physics.named.model.mat_rgba['walker/self']
        colors[:] = HUMANOID_PROMPT_COLOR

        self._should_truncate = False
        walker_foot_geoms = set(self._walker.ground_contact_geoms)
        walker_nonfoot_geoms = [
            geom for geom in self._walker.mjcf_model.find_all('geom')
            if geom not in walker_foot_geoms
        ]
        self._walker_nonfoot_geomids = set(physics.bind(walker_nonfoot_geoms).element_id)
        self._ground_geomids = set(physics.bind(self._arena.ground_geoms).element_id)

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)
        self._start_step = self._time_step
        self._time_step_override = 0

    def get_reward(self, physics: 'mjcf.Physics') -> float:
        # Find if there's a disallowed contact.
        disallowed_contact = False
        for contact in physics.data.contact:
            if self._is_disallowed_contact(contact):
                disallowed_contact = True
                break

        self._should_truncate = disallowed_contact and (self._termination_error > self._termination_error_threshold)

        return 0.
        

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

        self._end_mocap = self._time_step == self._last_step
        if self._max_steps is not None:
            if self._end_mocap and self._time_step_override < self._max_steps:
                self._time_step -= 1
                self._end_mocap = False
            elif not self._end_mocap and self._time_step_override >= self._max_steps:
                self._end_mocap = True
        
        self._reference_observations.update(self.get_all_reference_observations(physics))

        self._update_ghost(physics)

        if self._time_step == self._last_step-1 or self._time_step - self._start_step == self._steps_before_color_change:
            colors = physics.named.model.mat_rgba['walker/self']
            colors[:] = HUMANOID_COMPLETION_COLOR

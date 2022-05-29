import numpy as np

from dm_control.locomotion.tasks import go_to_target
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.utils import rewards

class GoToTarget(go_to_target.GoToTarget):
    def __init__(
        self,
        walker,
        arena,
        dense_reward=False,
        moving_target=False,
        target_relative=False,
        target_relative_dist=1.5,
        steps_before_moving_target=10,
        distance_tolerance=go_to_target.DEFAULT_DISTANCE_TOLERANCE_TO_TARGET,
        target_spawn_position=None,
        walker_spawn_position=None,
        walker_spawn_rotation=None,
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=0.03
    ):
        super().__init__(walker, arena, moving_target, target_relative, target_relative_dist,
                         steps_before_moving_target, distance_tolerance, target_spawn_position,
                         walker_spawn_position, walker_spawn_rotation, physics_timestep,
                         control_timestep)
        self._dense_reward = dense_reward

    def get_reward(self, physics):
        if not self._dense_reward:
            return super().get_reward(physics)
        distance = np.linalg.norm(
            physics.bind(self._target).pos[:2] -
            physics.bind(self._walker.root_body).xpos[:2]
        )
        reward = rewards.tolerance(
            distance,
            bounds=(0., self._distance_tolerance),
            margin=self._distance_tolerance,
            sigmoid='linear',
            value_at_margin=0.
        )
        if distance < self._distance_tolerance and self._moving_target:
            self._reward_step_counter += 1
        return reward

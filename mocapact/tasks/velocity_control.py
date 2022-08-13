"""
The velocity control task.
"""
import collections
import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable as dm_observable
from dm_control.locomotion.tasks.reference_pose import tracking

class VelocityControl(composer.Task):
    """
    A task that requires the walker to track a randomly changing velocity.
    """

    def __init__(
        self,
        walker,
        arena,
        max_speed=4.5,
        reward_margin=0.75,
        direction_exponent=1.,
        steps_before_changing_velocity=166,
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=0.03
    ):
        self._walker = walker
        self._arena = arena
        self._walker.create_root_joints(self._arena.attach(self._walker))
        self._max_speed = max_speed
        self._reward_margin = reward_margin
        self._direction_exponent = direction_exponent
        self._steps_before_changing_velocity = steps_before_changing_velocity
        self._move_speed = 0.
        self._move_angle = 0.
        self._move_speed_counter = 0.

        self._task_observables = collections.OrderedDict()
        def task_state(physics):
            del physics
            sin, cos = np.sin(self._move_angle), np.cos(self._move_angle)
            phase = self._move_speed_counter / self._steps_before_changing_velocity
            return np.array([self._move_speed, sin, cos, phase])
        self._task_observables['target_obs'] = dm_observable.Generic(task_state)

        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        enabled_observables.append(self._walker.observables.sensors_touch)
        enabled_observables.append(self._walker.observables.torso_xvel)
        enabled_observables.append(self._walker.observables.torso_yvel)
        enabled_observables += list(self._task_observables.values())
        for obs in enabled_observables:
            obs.enabled = True

        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)
        
    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def _is_disallowed_contact(self, contact):
        set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def _sample_move_speed(self, random_state):
        self._move_speed = random_state.uniform(high=self._max_speed)
        self._move_angle = random_state.uniform(high=2*np.pi)
        self._move_speed_counter = 0

    def should_terminate_episode(self, physics):
        del physics
        return self._failure_termination

    def get_discount(self, physics):
        del physics
        if self._failure_termination:
            return 0.
        else:
            return 1.

    def initialize_episode(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)
        self._sample_move_speed(random_state)

        self._failure_termination = False
        walker_foot_geoms = set(self._walker.ground_contact_geoms)
        walker_nonfoot_geoms = [
            geom for geom in self._walker.mjcf_model.find_all('geom')
            if geom not in walker_foot_geoms
        ]
        self._walker_nonfoot_geomids = set(physics.bind(walker_nonfoot_geoms).element_id)
        self._ground_geomids = set(physics.bind(self._arena.ground_geoms).element_id)

    def get_reward(self, physics):
        xvel = self._walker.observables.torso_xvel(physics)
        yvel = self._walker.observables.torso_yvel(physics)
        speed = np.linalg.norm([xvel, yvel])
        speed_error = self._move_speed - speed
        speed_reward = np.exp(-(speed_error / self._reward_margin)**2)
        if np.isclose(xvel, 0.) and np.isclose(yvel, 0.):
            angle_reward = 1.
        else:
            direction = np.array([xvel, yvel])
            direction /= np.linalg.norm(direction)
            direction_tgt = np.array([np.cos(self._move_angle), np.sin(self._move_angle)])
            dot = direction_tgt.dot(direction)
            angle_reward = ((dot + 1) / 2)**self._direction_exponent

        reward = speed_reward * angle_reward
        return reward

    def before_step(self, physics, action, random_state):
        self._walker.apply_action(physics, action, random_state)
    
    def after_step(self, physics, random_state):
        self._failure_termination = False
        for contact in physics.data.contact:
            if self._is_disallowed_contact(contact):
                self._failure_termination = True
                break

        self._move_speed_counter += 1
        if self._move_speed_counter >= self._steps_before_changing_velocity:
            self._sample_move_speed(random_state)

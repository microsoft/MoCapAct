"""
Aliases of observables used for clip experts, distillation, and RL transfer.
"""
# All observables associated with the CMU humanoid and clip tracking task
CMU_HUMANOID_OBSERVABLES = (
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/end_effectors_pos',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/world_zaxis'
)

MULTI_CLIP_OBSERVABLES_SANS_ID = (
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/end_effectors_pos',
    'walker/gyro_control',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/joints_vel_control',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/time_in_clip',
    'walker/velocimeter_control',
    'walker/world_zaxis',
    'walker/reference_rel_joints',
    'walker/reference_rel_bodies_pos_global',
    'walker/reference_rel_bodies_quats',
    'walker/reference_rel_bodies_pos_local',
    'walker/reference_ego_bodies_quats',
    'walker/reference_rel_root_quat',
    'walker/reference_rel_root_pos_local',
    'walker/reference_appendages_pos',
)

MULTI_CLIP_OBSERVABLES = MULTI_CLIP_OBSERVABLES_SANS_ID + ('walker/clip_id',)

# Observables used in controlling the humanoid
BASE_OBSERVABLES = (
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/sensors_velocimeter',
    'walker/sensors_gyro',
    'walker/end_effectors_pos',
    'walker/world_zaxis',
    'walker/actuator_activation',
    'walker/sensors_touch',
    'walker/sensors_torque',
)

TIME_INDEX_OBSERVABLES = BASE_OBSERVABLES + ('walker/time_in_clip',)
HIGH_LEVEL_OBSERVABLES_SANS_REFERENCE = BASE_OBSERVABLES + ('walker/body_height',)
HIGH_LEVEL_OBSERVABLES = HIGH_LEVEL_OBSERVABLES_SANS_REFERENCE + ('walker/reference_rel_bodies_pos_local', 'walker/reference_rel_bodies_quats')

# Observables for hierarchical observations
HIERARCHICAL_OBSERVABLES = dict(
    ref_encoder=HIGH_LEVEL_OBSERVABLES,
    decoder=BASE_OBSERVABLES
)

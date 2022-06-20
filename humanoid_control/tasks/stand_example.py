import numpy as np
from dm_control import composer
from dm_control import viewer
from dm_control.locomotion.arenas import floors
from humanoid_control.tasks import stand
from dm_control.locomotion.walkers import cmu_humanoid

if __name__ == '__main__':
    initializer = stand.StandUpInitalizer()
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(initializer=initializer)
    arena = floors.Floor()
    task = stand.StandUp(walker=walker, arena=arena)
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)

    spec = env.action_spec()
    def random_policy(time_step):
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)
    viewer.launch(env, policy=random_policy)

from turtle import speed
from ml_collections import ConfigDict
from dm_control.locomotion.tasks import go_to_target
from humanoid_control.tasks import speed_control

def get_config(task_string):
    tasks = {
        'go_to_target': ConfigDict({
            'constructor': go_to_target.GoToTarget,
            'config': ConfigDict(dict(
                moving_target=True
            ))
        }),
        'speed_control': ConfigDict({
            'constructor': speed_control.SpeedControl,
            'config': ConfigDict(dict(
                max_speed=4.5,
                reward_margin=0.75,
                steps_before_changing_speed=166
            ))
        })
    }

    return tasks[task_string]

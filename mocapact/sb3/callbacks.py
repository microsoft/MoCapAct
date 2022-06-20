import os
import os.path as osp
from typing import Optional
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback

class SaveVecNormalizeCallback(BaseCallback):
    """
    Taken from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/callbacks.py
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class LogOnRolloutEndCallback(BaseCallback):
    def __init__(self, log_dir, verbose: float = 0):
        super().__init__(verbose)
        self.filename = osp.join(log_dir, 'last_time.txt')

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        with open(self.filename, 'w') as f:
            f.write(str(datetime.now().timestamp()))
            f.flush()


class SeedEnvCallback(BaseCallback):
    def __init__(self, seed: int, verbose: float = 0):
        super().__init__(verbose)
        self.seed = seed

    def _on_step(self) -> bool:
        return True

    def _on_event(self) -> bool:
        self.parent.eval_env.seed(self.seed)
        return True

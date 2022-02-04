import numpy as np

def get_exponential_fn(start: float, decay: float, lr_min: float):
    def func(progress_remaining: float) -> float:
        lr = start * np.exp(-(1-progress_remaining)*decay)
        return np.maximum(lr, lr_min)
    return func

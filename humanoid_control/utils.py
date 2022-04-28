import os.path as osp
from importlib import import_module

from absl import logging

from dm_control.locomotion.mocap import cmu_mocap_data, loader
from dm_control.locomotion.tasks.reference_pose.tracking import _MAX_END_STEP

def str_to_callable(callable_name):
    module_name, method_name = callable_name.rsplit('.', 1)
    module = import_module(module_name)
    method = getattr(module, method_name)
    return method

def load_absl_flags(fname):
    """
    Loads the abseil flags from a text file. Does not include booleans.
    """
    flags = dict()
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "=" in line:
                flag, val = line.split("=")
                flags[flag.strip("--")] = val.strip("\n")
    return flags

def log_flags(flags, log_dir):
    """Logs the value of each of the flags."""
    for k in dir(flags):
        if k != '?':
            flag = 'FLAGS.{}'.format(k)
            logging.info('{}: {}'.format(flag, eval(flag)))
    flags.append_flags_into_file(osp.join(log_dir, 'flags.txt'))

def get_clip_length(clip_id):
    """
    We subtract one from the end step due to how the ReferencePosesTask handles the
    last step of a reference trajectory.
    """
    clip_loader = loader.HDF5TrajectoryLoader(cmu_mocap_data.get_path_for_cmu(version='2020'))
    clip = clip_loader.get_trajectory(clip_id, start_step=0, end_step=_MAX_END_STEP)
    return clip.end_step-1

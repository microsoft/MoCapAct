import glob
import json
import numpy as np
import os.path as osp
import shutil

from absl import app, flags
from pathlib import Path

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "")
flags.DEFINE_string("output_path", None, "")

def get_expert_paths(input_dir):
    """
    For each clip in the input directories, gets the path of the expert.
    """
    clips = set()
    expert_paths, expert_metrics = {}, {}
    experiment_paths = [osp.dirname(path) for path in glob.iglob(f"{input_dir}/**/flags.txt", recursive=True)]
    for path in experiment_paths:
        with open(osp.join(path, 'clip_info.json')) as f:
            clip_info = json.load(f)
        clip_id = clip_info['clip_id']
        start_step = clip_info['start_step']
        end_step = clip_info['end_step']
        expert_name = f"{clip_id}-{start_step}-{end_step}"
        if osp.exists(osp.join(path, 'eval_random/evaluations.npz')):
            try:
                eval_npz = np.load(osp.join(path, 'eval_random/evaluations.npz'))
            except:
                continue
            clips.add(clip_id)
            idx = eval_npz['results'].mean(1).argmax()
            ret = eval_npz['results'][idx].mean()
            if expert_name not in expert_paths or ret > expert_metrics[expert_name]['ep_return'].mean():
                expert_paths[expert_name] = path
                expert_metrics[expert_name] = dict(
                    ep_return=eval_npz['results'][idx],
                    ep_length=eval_npz['ep_lengths'][idx],
                    ep_norm_return=eval_npz['results_norm'][idx],
                    ep_norm_length=eval_npz['ep_lengths_norm'][idx]
                )
    return expert_paths, expert_metrics, clips

def main(_):
    paths, _, _ = get_expert_paths(FLAGS.input_dir)
    for clip, old_path in paths.items():
        new_root = osp.join(FLAGS.output_path, clip)
        Path(new_root).mkdir(parents=True, exist_ok=True)
        shutil.copy(osp.join(old_path, 'clip_info.json'), new_root)
        shutil.copytree(osp.join(old_path, 'eval_random/model'), osp.join(new_root, 'eval_rsi/model'))

if __name__ == "__main__":
    app.run(main)

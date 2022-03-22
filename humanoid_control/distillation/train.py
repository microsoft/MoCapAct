import os
import os.path as osp
import random
import collections
from pathlib import Path
from datetime import datetime, timedelta
from absl import flags, app
from pandas import concat
from torch.utils.data.dataloader import DataLoader
import ml_collections
from ml_collections.config_flags import config_flags
import pytorch_lightning as pl

from stable_baselines3.common.running_mean_std import RunningMeanStd

from humanoid_control import observables
from humanoid_control import utils
from humanoid_control.distillation import callbacks
from humanoid_control.distillation import dataset

FLAGS = flags.FLAGS

# Path flags
flags.DEFINE_string("output_root", None, "Output directory to save the model and logs")
flags.DEFINE_list("train_dataset_paths", None, "Path(s) to training dataset(s)")
flags.DEFINE_list("val_dataset_paths", None, "Path(s) to validation dataset(s), if desired")
flags.DEFINE_bool("do_validation_loop", False, "Whether to run PyTorch Lightning's loop over the validation set")
flags.DEFINE_bool("randomly_load_hdf5", False, "Whether to randomize the order of hdf5 files before loading")
flags.DEFINE_integer("save_every_n_minutes", 60, "How often to save latest model")

# Training hyperparameters
flags.DEFINE_integer("n_hours", 24, "Number of hours to train")
flags.DEFINE_integer("batch_size", 64, "Batch size used during training")
flags.DEFINE_list("clip_ids", None, "List of clips to consider. By default, every clip.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_float("max_grad_norm", float('inf'), "Clip gradient norm")
flags.DEFINE_integer("n_workers", 8, "Number of workers for loading data")
flags.DEFINE_bool("preload_dataset", False, "Whether to preload the dataset to RAM")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_integer("progress_bar_refresh_rate", 1, "How often to refresh progress bar")
flags.DEFINE_bool("track_grad_norm", False, "Whether to log the gradient norm")
flags.DEFINE_float("temperature", None, "AWR temperature")

# Model hyperparameters
config_file = "humanoid_control/distillation/config.py"
config_flags.DEFINE_config_file("model", f"{config_file}:mlp_time_index", "Model architecture")
flags.DEFINE_string("gpus", "-1", "GPU configuration (https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices)")
flags.DEFINE_bool("normalize_obs", False, "Whether to normalize the input observation")
flags.DEFINE_string("load_path", None, "Load path to warm-start")
flags.DEFINE_bool("record_video", False, "Whether to record video for evaluation")

# Rollout evaluation hyperparameters
eval_config = ml_collections.ConfigDict()
eval_config.freq = int(1e5)
eval_config.n_episodes = 2500
eval_config.act_noise = 0.
eval_config.min_steps = 15
eval_config.termination_error_threshold = 0.3
eval_config.n_workers = 8
eval_config.seed = 0
flags.DEFINE_multi_enum("eval_mode", [], ["train_start", "train_random", "val_start", "val_random"], "What dataset and initialization to do evaluation on")
config_flags.DEFINE_config_dict("eval", eval_config)

flags.mark_flag_as_required("output_root")
flags.mark_flag_as_required("train_dataset_paths")

def main(_):
    output_dir = osp.join(FLAGS.output_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Log some stuff (but only in process 0)
    if os.getenv("LOCAL_RANK", "0") == "0":
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(FLAGS.flags_into_string())
        with open(osp.join(output_dir, 'flags.txt'), 'w') as f:
            f.write(FLAGS.flags_into_string())
        with open(osp.join(output_dir, 'model_constructor.txt'), 'w') as f:
            f.write(FLAGS.model.constructor)

    pl.seed_everything(FLAGS.seed, workers=True)

    # If desired, randomize order of dataset paths
    if FLAGS.randomly_load_hdf5:
        print(FLAGS.train_dataset_paths)
        random.shuffle(FLAGS.train_dataset_paths)
        print(FLAGS.train_dataset_paths)
        if FLAGS.val_dataset_paths is not None:
            random.shuffle(FLAGS.val_dataset_paths)

    # Make supervision dataset
    if hasattr(FLAGS.model.config, 'seq_steps'):
        seq_steps = FLAGS.model.config.seq_steps
    elif hasattr(FLAGS.model.config, 'block_size'):
        seq_steps = FLAGS.model.config.block_size
    else:
        seq_steps = 1

    train_dataset = dataset.ExpertDataset(
        FLAGS.train_dataset_paths,
        observables.MULTI_CLIP_OBSERVABLES_SANS_ID,
        FLAGS.clip_ids,
        min_seq_steps=seq_steps,
        max_seq_steps=seq_steps,
        normalize_obs=False, #FLAGS.normalize_obs,
        preload=FLAGS.preload_dataset,
        temperature=FLAGS.temperature,
        concat_observables=False
    )

    if FLAGS.val_dataset_paths is not None:
        val_dataset = dataset.ExpertDataset(
            FLAGS.val_dataset_paths,
            FLAGS.model.config.observables,
            FLAGS.clip_ids,
            min_seq_steps=seq_steps,
            max_seq_steps=seq_steps,
            normalize_obs=FLAGS.normalize_obs,
            preload=FLAGS.preload_dataset
        )

    if FLAGS.normalize_obs:
        obs_rms = {}
        observable_indices = train_dataset.observable_indices['walker']
        for k in observable_indices:
            rms = RunningMeanStd(shape=(len(observable_indices[k]),))
            rms.mean = train_dataset.obs_mean[observable_indices[k]]
            rms.var = train_dataset.obs_var[observable_indices[k]]
            obs_rms[f"walker/{k}"] = rms
    else:
        obs_rms = None

    # Make policy to be trained
    model_constructor = utils.str_to_callable(FLAGS.model.constructor)
    policy = model_constructor(
        train_dataset.full_observation_space,
        train_dataset.action_space,
        ref_steps=train_dataset.ref_steps,
        learning_rate=FLAGS.learning_rate,
        features_extractor_kwargs=dict(observable_keys=FLAGS.model.config.observables, obs_rms=obs_rms),
        **FLAGS.model.config
    )

    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                              batch_size=FLAGS.batch_size, num_workers=FLAGS.n_workers)

    ############
    # Callbacks
    ############
    train_callbacks = []
    # Saving latest model callback
    last_model_callback = pl.callbacks.ModelCheckpoint(
        dirpath=osp.join(output_dir, 'model'),
        save_top_k=0,
        save_last=True,
        train_time_interval=timedelta(minutes=FLAGS.save_every_n_minutes)
    )
    train_callbacks.append(last_model_callback)
    for eval_mode in FLAGS.eval_mode:
        is_train_dataset = (eval_mode.startswith("train"))
        always_init_at_clip_start = (eval_mode.endswith("start"))
        prefix = (
            ("train_" if is_train_dataset else "val_")
            + ("start" if always_init_at_clip_start else "random")
        )
        if not is_train_dataset and FLAGS.val_dataset_paths is None:
            continue
        if os.getenv("LOCAL_RANK", "0") == "0":
            Path(osp.join(output_dir, 'eval', prefix)).mkdir(parents=True, exist_ok=True)
        eval_dataset = train_dataset if is_train_dataset else val_dataset
        eval_callback = callbacks.PolicyEvaluationCallback(
            eval_dataset.all_clip_ids,
            eval_dataset.ref_steps[...],
            FLAGS.eval.n_episodes,
            FLAGS.eval.freq,
            FLAGS.eval.act_noise,
            FLAGS.eval.min_steps,
            FLAGS.eval.termination_error_threshold,
            always_init_at_clip_start,
            FLAGS.eval.n_workers,
            FLAGS.eval.seed,
            prefix + '_',
            osp.join(output_dir, 'eval', prefix),
            record_video=FLAGS.record_video,
            verbose=1
        )
        train_callbacks.append(eval_callback)
    csv_logger = pl.loggers.CSVLogger(output_dir, name='logs', version='')
    tb_logger = pl.loggers.TensorBoardLogger(output_dir, name='logs', version='')
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        gpus=FLAGS.gpus,
        strategy='ddp',
        track_grad_norm=2 if FLAGS.track_grad_norm else -1,
        max_time=timedelta(hours=FLAGS.n_hours),
        gradient_clip_val=FLAGS.max_grad_norm,
        progress_bar_refresh_rate=FLAGS.progress_bar_refresh_rate,
        deterministic=True,
        benchmark=True,
        callbacks=train_callbacks,
        logger=[csv_logger, tb_logger],
        profiler="simple"
    )
    trainer.fit(policy, train_loader, ckpt_path=FLAGS.load_path)

if __name__ == '__main__':
    app.run(main)

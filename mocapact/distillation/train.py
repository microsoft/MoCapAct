"""
Main script for training a policy given a dataset.
"""
import os
import os.path as osp
import random
from pathlib import Path
from datetime import datetime, timedelta
from absl import flags, app
import torch
from torch.utils.data.dataloader import DataLoader
import ml_collections
from ml_collections.config_flags import config_flags
import pytorch_lightning as pl

from stable_baselines3.common.running_mean_std import RunningMeanStd

from mocapact import observables
from mocapact import utils
from mocapact.distillation import callbacks
from mocapact.distillation import dataset

FLAGS = flags.FLAGS

# Path flags
flags.DEFINE_string("output_root", None, "Output directory to save the model and logs")
flags.DEFINE_string("dataset_metrics_path", None, "Path to load dataset metrics")
flags.DEFINE_list("train_dataset_paths", None, "Path(s) to training dataset(s)")
flags.DEFINE_list("val_dataset_paths", None, "Path(s) to validation dataset(s), if desired")
flags.DEFINE_list("extra_clips", None, "List of clip snippets to additionaly do evaluations on, if desired")
flags.DEFINE_bool("include_timestamp", True, "Whether to include timestamp in log directory")
flags.DEFINE_integer("validation_freq", None, "How often (in iterations) to do validation loop")
flags.DEFINE_integer("train_start_rollouts", -1, "Number of start rollouts to consider in training set")
flags.DEFINE_integer("train_rsi_rollouts", -1, "Number of RSI rollouts to consider in training set")
flags.DEFINE_integer("val_start_rollouts", -1, "Number of start rollouts to consider in validation set")
flags.DEFINE_integer("val_rsi_rollouts", -1, "Number of RSI rollouts to consider in validation set")
flags.DEFINE_bool("randomly_load_hdf5", False, "Whether to randomize the order of hdf5 files before loading")
flags.DEFINE_bool("clip_len_upsampling", False, "Compensate for shorter clips by upsampling")
flags.DEFINE_integer("save_every_n_minutes", 60, "How often to save latest model")

# Training hyperparameters
flags.DEFINE_integer("n_hours", 24, "Number of hours to train")
flags.DEFINE_integer("n_steps", None, "Alternatively, how many steps to run training")
flags.DEFINE_integer("batch_size", 64, "Batch size used during training")
flags.DEFINE_list("clip_ids", None, "List of clips to consider. By default, every clip.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_float("max_grad_norm", float('inf'), "Clip gradient norm")
flags.DEFINE_integer("n_workers", 8, "Number of workers for loading data")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_integer("progress_bar_refresh_rate", 1, "How often to refresh progress bar")
flags.DEFINE_bool("track_grad_norm", False, "Whether to log the gradient norm")
flags.DEFINE_bool("clip_weighted", False, "Whether to use a weight defined solely by the clip or also by the state and action")
flags.DEFINE_bool("advantage_weights", True, "Whether to use AWR or RWR")
flags.DEFINE_float("temperature", None, "Weighting temperature")
flags.DEFINE_bool("keep_hdf5s_open", False, "Whether to keep all HDF5s open (will cause memory leaks!)")

# Model hyperparameters
config_file = "mocapact/distillation/config.py"
config_flags.DEFINE_config_file("model", f"{config_file}:mlp_time_index", "Model architecture")
flags.DEFINE_list("gpus", None, "GPU configuration (https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus)")
flags.DEFINE_bool("normalize_obs", False, "Whether to normalize the input observation")
flags.DEFINE_string("load_path", None, "Load path to warm-start")
flags.DEFINE_bool("record_video", False, "Whether to record video for evaluation")

# Rollout evaluation hyperparameters
eval_config = ml_collections.ConfigDict()
eval_config.freq = int(1e5)
eval_config.n_episodes = 2500
eval_config.act_noise = 0.
eval_config.min_steps = 10
eval_config.termination_error_threshold = 0.3
eval_config.n_workers = 8
eval_config.seed = 0
eval_config.serial = False
eval_config.run_at_beginning = False
flags.DEFINE_multi_enum("eval_mode", [], ["train_start", "train_rsi", "val_start", "val_rsi", "clips_start", "clips_rsi"], "What dataset and initialization to do evaluation on")
config_flags.DEFINE_config_dict("eval", eval_config)

flags.mark_flag_as_required("output_root")
flags.mark_flag_as_required("train_dataset_paths")
flags.mark_flag_as_required("dataset_metrics_path")

def main(_):
    output_dir = FLAGS.output_root
    if FLAGS.include_timestamp:
        output_dir = osp.join(output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Make supervision dataset
    if hasattr(FLAGS.model.config, 'seq_steps'):
        seq_steps = FLAGS.model.config.seq_steps
    elif hasattr(FLAGS.model.config, 'block_size'):
        seq_steps = FLAGS.model.config.block_size
    else:
        seq_steps = 1


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
    train_dataset = dataset.ExpertDataset(
        FLAGS.train_dataset_paths,
        observables.MULTI_CLIP_OBSERVABLES_SANS_ID,
        FLAGS.dataset_metrics_path,
        FLAGS.clip_ids,
        min_seq_steps=seq_steps,
        max_seq_steps=seq_steps,
        n_start_rollouts=FLAGS.train_start_rollouts,
        n_rsi_rollouts=FLAGS.train_rsi_rollouts,
        normalize_obs=False,
        clip_len_upsampling=FLAGS.clip_len_upsampling,
        clip_weighted=FLAGS.clip_weighted,
        advantage_weights=FLAGS.advantage_weights,
        temperature=FLAGS.temperature,
        concat_observables=False,
        keep_hdf5s_open=FLAGS.keep_hdf5s_open
    )

    print("Train set size =", len(train_dataset))

    if FLAGS.val_dataset_paths is not None:
        val_dataset = dataset.ExpertDataset(
            FLAGS.val_dataset_paths,
            observables.MULTI_CLIP_OBSERVABLES_SANS_ID,
            FLAGS.dataset_metrics_path,
            FLAGS.clip_ids,
            min_seq_steps=seq_steps,
            max_seq_steps=seq_steps,
            n_start_rollouts=FLAGS.val_start_rollouts,
            n_rsi_rollouts=FLAGS.val_rsi_rollouts,
            normalize_obs=False,
            clip_weighted=FLAGS.clip_weighted,
            advantage_weights=FLAGS.advantage_weights,
            temperature=FLAGS.temperature,
            concat_observables=False,
            keep_hdf5s_open=FLAGS.keep_hdf5s_open
        )
        print("Validation set size =", len(val_dataset))

    if FLAGS.normalize_obs:
        obs_rms = {}
        for obs_key, obs_indices in train_dataset.observable_indices.items():
            rms = RunningMeanStd(shape=obs_indices.shape)
            rms.mean = train_dataset.proprio_mean[obs_indices]
            rms.var = train_dataset.proprio_var[obs_indices]
            rms.count = train_dataset.count
            obs_rms[obs_key] = rms
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
    if FLAGS.val_dataset_paths is not None and FLAGS.validation_freq is not None:
        val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.n_workers)
    else:
        val_loader = None

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
    if val_loader is not None: # Validation set callback
        train_callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=osp.join(output_dir, "eval/validation"),
            filename="best",
            monitor="val_loss/loss",
            save_top_k=1,
            every_n_epochs=1
        ))
    for eval_mode in FLAGS.eval_mode: # Policy evaluation callbacks
        is_train_dataset = eval_mode.startswith("train")
        always_init_at_clip_start = eval_mode.endswith("start")
        prefix = eval_mode
        if os.getenv("LOCAL_RANK", "0") == "0":
            Path(osp.join(output_dir, 'eval', prefix)).mkdir(parents=True, exist_ok=True)
        snippets = (train_dataset.clip_snippets_flat if is_train_dataset
                    else val_dataset.clip_snippets_flat if eval_mode.startswith("val")
                    else FLAGS.extra_clips)
        eval_callback = callbacks.PolicyEvaluationCallback(
            snippets,
            train_dataset.ref_steps,
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
            run_at_beginning=FLAGS.eval.run_at_beginning,
            serial_evaluation=FLAGS.eval.serial,
            record_video=FLAGS.record_video,
            verbose=1
        )
        train_callbacks.append(eval_callback)
    csv_logger = pl.loggers.CSVLogger(output_dir, name='logs', version='')
    tb_logger = pl.loggers.TensorBoardLogger(output_dir, name='logs', version='')
    gpus = -1 if FLAGS.gpus is None else [int(x) for x in FLAGS.gpus]
    multigpu = (gpus == -1 and torch.cuda.device_count() > 1) or (gpus != -1 and len(gpus) > 1)
    strategy = 'ddp' if multigpu else None
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        gpus=gpus,
        auto_select_gpus=True,
        strategy=strategy,
        track_grad_norm=2 if FLAGS.track_grad_norm else -1,
        max_steps=FLAGS.n_steps,
        max_time=timedelta(hours=FLAGS.n_hours),
        gradient_clip_val=FLAGS.max_grad_norm,
        progress_bar_refresh_rate=FLAGS.progress_bar_refresh_rate,
        val_check_interval=FLAGS.validation_freq if val_loader is not None else None,
        deterministic=True,
        benchmark=True,
        callbacks=train_callbacks,
        logger=[csv_logger, tb_logger]
    )
    trainer.fit(policy, train_loader, val_dataloaders=val_loader, ckpt_path=FLAGS.load_path)

if __name__ == '__main__':
    app.run(main)

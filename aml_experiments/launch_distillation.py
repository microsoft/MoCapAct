import os
import sys
from azureml.core import (
    Datastore, Dataset, Experiment, ScriptRunConfig, Workspace,
)

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.data import OutputFileDatasetConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__))))
import compute_manager
import environment_manager

interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
ws = Workspace(subscription_id="c8b7f913-60fb-4759-a310-fc5630e56f99",
               resource_group="dilbert-rg",
               workspace_name="dilbert-ws",
               auth=interactive_auth)

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

setup_cmds = ('pip install -e . && ')

script_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        setup_cmds + 'python', './humanoid_control/distillation/train.py',
        "--train_dataset_paths", Dataset.File.from_files((Datastore.get(ws, "humcontrolds"), 'humanoid_control/rollouts/few/CMU_108_20.hdf5')).as_download(), Dataset.File.from_files((Datastore.get(ws, "humcontrolds"), 'humanoid_control/rollouts/few/CMU_035_29.hdf5')).as_download(),
        "--val_dataset_paths", Dataset.File.from_files((Datastore.get(ws, "humcontrolds"), 'humanoid_control/rollouts/few/CMU_078_32.hdf5')).as_download(), Dataset.File.from_files((Datastore.get(ws, "humcontrolds"), 'humanoid_control/rollouts/few/CMU_016_58.hdf5')).as_download(),
        "--n_workers", "24",
        "--output_root", "./logs",
        "--learning_rate", "0.0005",
        "--model", "humanoid_control/distillation/config.py:hierarchical_rnn",
        "--model.config.activation_fn", "torch.nn.ELU",
        "--nomodel.config.squash_output",
        "--max_grad_norm", "1",
        "--normalize_obs",
        "--batch_size", "64",
        "--save_every_n_minutes", "60",
        "--n_hours", "2",
        "--eval_mode", "train_start", "--eval_mode", "train_random", "--eval_mode", "val_start", "--eval_mode", "val_random",
        "--eval.freq", "50000",
        "--eval.n_workers", "24",
        "--model.config.embed_size", "60",
        "--eval.n_episodes", "1000",
        "--dataset_metrics_path", Dataset.File.from_files(Datastore.get(ws, "humcontrolds").path('humanoid_control/distillation/few/dataset_metrics.npz')).as_download(),
    ],
    compute_target=compute_manager.create_compute_target(ws, 'gpu-NC24'),
    environment=environment_manager.create_env(ws, "hum-control-env", os.path.join(root_dir, 'requirements.txt'))
)

exp = Experiment(workspace=ws, name='hum-control-distillation')
exp.submit(config=script_run_config)

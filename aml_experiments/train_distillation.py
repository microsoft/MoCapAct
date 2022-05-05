import os
import sys
from azureml.core import (
    Datastore, Dataset, Environment, Experiment, ScriptRunConfig, Workspace, 
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
               auth=interactive_auth
)

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
output_ref = OutputFileDatasetConfig(destination=(ws.get_default_datastore(), 'locomotion'))
input_ref = Dataset.File.from_files(Datastore.get(ws, "locomotion").path('distillation/locomotion_dataset.hdf5')).as_download()

installation_cmds = ('python setup.py build_mjbindings --inplace=True --headers-dir=/opt/mujoco200_linux/include && ' +
                     'pip install -e . && ')

script_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        installation_cmds + 'python', './humanoid_control/distillation/train.py',
        "--train_dataset_paths", input_ref,
        "--output_root", output_ref,
        "--batch_size", "128",
        "--n_hours", "48",
        "--n_workers", "8",
        "--learning_rate", "3e-5",
        "--model", "./dm_control/locomotion/motion_completion/distillation/config.py:gpt",
        "--max_grad_norm", "1",
        "--normalize_obs",
        "--seq_steps", "32",
        "--eval.freq", "100000",
        "--save_every_n_minutes", "60",
        "--eval.n_workers", "32",
        "--eval_mode", "train_start",
        "--eval_mode", "train_random",
    ],
    compute_target=compute_manager.create_compute_target(ws, 'gpu-NC24'),
    environment=environment_manager.create_env_from_dockerfile(ws, "dm-control-env", os.path.join(root_dir, 'aml_experiments', 'dm-control.Dockerfile'))
)

exp = Experiment(workspace=ws, name='locomotion-distillation')
exp.submit(config=script_run_config)

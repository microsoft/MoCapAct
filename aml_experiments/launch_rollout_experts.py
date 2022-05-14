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
        setup_cmds + 'python', './humanoid_control/distillation/rollout_experts.py',
        "--input_dirs", Dataset.File.from_files((Datastore.get(ws, "humcontrolds"), 'humanoid_experts/CMU_016_22-0-82')).as_download(),
        "--output_path", OutputFileDatasetConfig(destination=(Datastore.get(ws, "humcontrolds"), 'humanoid_offline_rl/rollouts')).as_upload(),
        "--n_start_rollouts", "1000",
        "--n_random_rollouts", "1000",
        "--n_workers", "20",
        "--min_steps", "25"
    ],
    compute_target=compute_manager.create_compute_target(ws, 'cpu-ds15v2'),
    environment=environment_manager.create_env(ws, "hum-control-env", os.path.join(root_dir, 'requirements.txt'))
)

exp = Experiment(workspace=ws, name='hum-control-rollouts')
exp.submit(config=script_run_config)

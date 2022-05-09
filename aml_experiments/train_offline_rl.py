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
input_ref = Dataset.File.from_files(Datastore.get(ws, "locomotion").path('humanoid_offline_rl')).as_download()
output_ref = OutputFileDatasetConfig(destination=(ws.get_default_datastore(), 'locomotion'))

# setup_cmds = ('pip install wrapt ratelimit azureml-mlflow && ' +
#               'mkdir -p $(dirname $(dirname $(which python)))/lib/python3.7/site-packages && ' +
#               'cp ./aml_experiments/tensorboard_patcher.py $(dirname $(dirname $(which python)))/lib/python3.7/site-packages/usercustomize.py && ' +
#               'pip install -e . && ')

setup_cmds = ('pip install -e . && ')

script_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        setup_cmds + 'python', './humanoid_control/offline_rl/train_bcq.py',
        "--dataset_local_path", input_ref,
        # "--output_root", 'output_ref',
        "--output_root", './logs',
        "--train_dataset_files", "CMU_001_01-0-178.hdf5",
    ],
    compute_target=compute_manager.create_compute_target(ws, 'gpu-NC24'),
    environment=environment_manager.create_env(ws, "hum-control-env", os.path.join(root_dir, 'requirements.txt'))
)

exp = Experiment(workspace=ws, name='hum-control-offline-rl')
exp.submit(config=script_run_config)

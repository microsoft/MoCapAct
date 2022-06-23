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
        setup_cmds + 'python', './scripts/compress_files.py',
        "--input_path", Dataset.File.from_files((Datastore.get(ws, "mocapact"), 'dataset/large')).as_download(),
        "--output_path", OutputFileDatasetConfig(destination=(Datastore.get(ws, "mocapact"), 'compressed')),
        '--file_name', "large.tar.gz"
    ],
    compute_target=compute_manager.create_compute_target(ws, 'cpu-e96a-v4'),
    environment=environment_manager.create_env(ws, "hum-control-env", os.path.join(root_dir, 'requirements.txt'))
)

exp = Experiment(workspace=ws, name='hum-control-compress-files')
exp.submit(config=script_run_config)

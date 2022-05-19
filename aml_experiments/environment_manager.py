import os
from azureml.core.conda_dependencies import CondaDependencies

from azureml.core import Environment

def create_env(ws, env_name, requirements_file_path):
    # env = Environment.get(workspace=ws, name="AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu").clone(env_name)
    env = Environment(name=env_name)
    env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20220412.v1'

    conda_dependencies = CondaDependencies()
    conda_dependencies.set_python_version('3.7')
    conda_dependencies.add_channel('menpo')
    conda_dependencies.add_channel('conda-forge')

    conda_dependencies.add_conda_package('pip==21.2.2')
    conda_dependencies.add_conda_package('glew')
    conda_dependencies.add_conda_package('mesalib')
    conda_dependencies.add_conda_package('glfw3')
    conda_dependencies.add_conda_package('imageio-ffmpeg')

    for p in Environment.from_pip_requirements(name="myenv", file_path=requirements_file_path).python.conda_dependencies.pip_packages:
        conda_dependencies.add_pip_package(p)

    env.python.conda_dependencies = conda_dependencies
    env.environment_variables['MUJOCO_GL'] = 'osmesa'
    env.register(ws)
    return env

def create_env_from_dockerfile(ws, env_name, dockerfile_path):
    env = None
    with open(dockerfile_path, "r") as f:
        dockerfile=f.read()

    env = Environment(name=env_name)
    env.docker.base_image = None
    env.docker.base_dockerfile = dockerfile
    env.docker.enabled = True
    env.python.user_managed_dependencies = True

    env.register(ws)
    return env

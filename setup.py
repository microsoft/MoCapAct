from distutils.core import setup

setup(
    name='mocapact',
    version='0.1',
    author='Nolan Wagener',
    author_email='nolan.wagener@gatech.edu',
    license='MIT',
    packages=['mocapact'],
    install_requires=[
        'azure.storage.blob==12.9.0',
        'cloudpickle>=2.1.0',
        'dm_control',
        'h5py',
        'imageio',
        'imageio-ffmpeg',
        'ml_collections',
        'mujoco',
        'pytorch-lightning<1.7',
        'stable-baselines3'
    ]
)

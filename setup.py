from distutils.core import setup

setup(
    name='mocapact',
    version='0.1',
    author='Nolan Wagener',
    author_email='nolan.wagener@gatech.edu',
    packages=['mocapact'],
    install_requires=[
        'cloudpickle>=2.1.0',
        'dm_control<=1.0.2',
        'h5py',
        'imageio',
        'imageio-ffmpeg',
        'ml_collections',
        'mujoco<=2.1.5',
        'pytorch-lightning<=1.6.4',
        'stable-baselines3'
    ]
)

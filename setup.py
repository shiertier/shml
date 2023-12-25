from setuptools import setup, find_packages

setup(
    name='shml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'einops>=0.6.0',
        'huggingface_hub',
        'timm>=0.9.0',
        'torch==2.1.1'
    ],
)
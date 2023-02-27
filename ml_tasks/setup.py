from setuptools import setup, find_packages

setup(
    name='ml_tasks',
    version='0.0.1',
    description='ml tasks',
    package_data={'ml_tasks': ['py.typed']},
    packages=find_packages(),
)
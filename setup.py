from setuptools import setup, find_packages

setup(
    name='medical_pytorch',
    version='0.1',
    description='A project boundling several MIC libraries.',
    url='https://github.com/camgbus/medical_pytorch',
    keywords='python setuptools',
    packages=find_packages(include=['mp', 'mp.*']),
)
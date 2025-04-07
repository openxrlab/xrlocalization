import pytest
import shutil
import os
import subprocess

dataset_dir = 'tests/data'


@pytest.fixture(scope='session', autouse=True)
def fixture():
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    url = 'https://docs.google.com/uc?id=1vKfCDWtZ1ui5t5sYjlF_EAqj1mVM1zk-'
    command = ['gdown', url]
    subprocess.run(command, check=True)
    command = ['tar', '-xf', 'xrloc-test-meta.tar.gz']
    subprocess.run(command, check=True)
    command = ['mv', 'xrloc-test-meta', dataset_dir]
    subprocess.run(command, check=True)
    command = ['rm', 'xrloc-test-meta.tar.gz']
    subprocess.run(command, check=True)
    yield
    command = ['rm', '-rf', dataset_dir]
    subprocess.run(command, check=True)

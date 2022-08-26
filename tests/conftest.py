import pytest
import shutil
import os
import subprocess

dataset_dir = 'tests/data'

@pytest.fixture(scope='session', autouse=True)
def fixture():
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    url = 'https://sensear-map.oss-cn-hangzhou.aliyuncs.com/stloc_sdk/xrloc-test-data.tar.gz'
    command = ['wget', '--no-check-certificate', url]
    subprocess.run(command, check=True)
    command = ['tar', '-xf',  'xrloc-test-data.tar.gz']
    subprocess.run(command, check=True)
    command = ['mv', 'xrloc-test-data', dataset_dir]
    subprocess.run(command, check=True)
    command = ['rm', 'xrloc-test-data.tar.gz']
    subprocess.run(command, check=True)
    yield
    command = ['rm', '-rf', dataset_dir]
    subprocess.run(command, check=True)


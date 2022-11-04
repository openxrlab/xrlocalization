import pytest
import shutil
import os
import subprocess

dataset_dir = 'tests/data'


@pytest.fixture(scope='session', autouse=True)
def fixture():
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    url = 'https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrlocalization/meta/xrloc-test-meta.tar.gz'
    command = ['wget', '--no-check-certificate', url]
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

import os
from setuptools import setup, find_packages


def package_files(directory):
    items = []
    for (root, directories, filenames) in os.walk(directory):
        if len(filenames) == 0:
            continue
        item = (root, [])
        for filename in filenames:
            item[1].append(os.path.join(root, filename))
        items.append(item)
    return items


models = package_files('models/')
third_party = package_files('3rdparty/')


setup(
    name="xrloc",
    version='0.5.1',
    author="Hailin Yu",
    author_email="yuhailin@sensetime.com",
    description="XRLoc is a visual localization toolbox",
    url="http://staging.openxrlab.openxxlab.com",
    packages=find_packages(),

    data_files=[
        *models, *third_party
    ]
)


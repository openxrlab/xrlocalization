import os

import pytest

import xrloc.utils.image as image

image_dir = 'tests/data/query'


def test_read_image():
    image_path = os.path.join(image_dir, '000001.jpg')
    im = image.read_image(image_path, grayscale=False)
    assert im.shape == (1440, 1920, 3)
    im = image.read_image(image_path, grayscale=True)
    assert im.shape == (1440, 1920)
    image_path = os.path.join(image_dir, '000001.png')
    with pytest.raises(ValueError,
                       match='Cannot read image {}'.format(image_path)):
        image.read_image(image_path, grayscale=False)


def test_resize_image():
    image_path = os.path.join(image_dir, '000001.jpg')
    im = image.read_image(image_path, grayscale=False)
    rim, factor = image.image_resize(im, 640)
    assert rim.shape == (im.shape[0] / 3, im.shape[1] / 3, 3)
    assert abs(factor - 1 / 3) < 1e-6
    rim, factor = image.image_resize(im, 1930)
    assert rim.shape == im.shape
    assert factor == 1


def test_convert_gray_image():
    image_path = os.path.join(image_dir, '000001.jpg')
    im = image.read_image(image_path, grayscale=False)
    rim = image.convert_gray_image(im)
    assert rim.shape == (1440, 1920)
    im = image.read_image(image_path, grayscale=True)
    rim = image.convert_gray_image(im)
    assert rim.shape == (1440, 1920)


def test_image_dataset():
    data = image.ImageDataSet(image_dir, do_read=False)
    assert len(data) == 3
    data = image.ImageDataSet(image_dir, do_read=True)
    assert len(data) == 3
    assert data[0].shape == (1440, 1920, 3)

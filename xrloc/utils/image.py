import cv2 as cv

from xrloc.utils.miscs import glob_images


def read_image(path, grayscale=False):
    """ Read image
    Args:
        path (str): Path to image
        grayscale (bool): Read image with gray mode
    Returns:
        array[unit8]: Scalar value in [0, 255] and
            with RGB and HWC
    """
    mode = cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_COLOR
    image = cv.imread(str(path), mode)
    if image is None:
        raise ValueError('Cannot read image {}'.format(path))
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def image_resize(image, max_size):
    """Resize image
    Args:
        image (array[uint8]): Scalar value in [0, 255] and
            with (RGB or Gray) and HWC
        max_size (int): Max size of images
    Returns:
        array[uint8]: Resized image
    """
    height, width = image.shape[:2]
    if max(height, width) < max_size:
        return image, 1
    factor = max_size * 1.0 / max(height, width)
    height_scaled, width_scaled = int(height * factor), int(width * factor)
    image_resized = cv.resize(image, (width_scaled, height_scaled),
                              interpolation=cv.INTER_LINEAR)
    return image_resized, factor


def convert_gray_image(image):
    """Resize image
    Args:
        image (array[uint8]): Scalar value in [0, 255] and
            with (RGB or Gray) and HWC
    Returns:
        array[uint8]: Gray image
    """
    if len(image.shape) == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def show_image(name, image):
    """Show given image
    Args:
        name: Window name
        image (array[uint8]): Scalar value in [0, 255] and
            with (RGB or Gray) and HWC
    """
    if len(image.shape) == 3:
        image = image[:, :, ::-1]  # RGB to BGR
    cv.imshow(name, image)
    cv.waitKey(0)


def write_image(path, image):
    """Show given image
    Args:
        path: Image to save
        image (array[uint8]): Scalar value in [0, 255] and
            with (RGB or Gray) and HWC
    """
    if len(image.shape) == 3:
        image = image[:, :, ::-1]  # RGB to BGR
    cv.imwrite(path, image)


class ImageDataSet:
    """Image dataset.

    Args:
        image_dir (str): Path to directory including images
        do_read (str): Read image or not
    """
    def __init__(self, image_dir, do_read=True):
        self.paths = glob_images(image_dir)
        self.do_read = do_read

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.do_read:
            return read_image(self.paths[index])
        else:
            return self.paths[index]

import cv2 as cv
import copy


def draw_keypoint(image, kps, color=(255, 0, 0)):
    """Draw keypoint on image
    Args:
        image: RGB, HWC
        kps: N*2
        color: RGB
    """
    # markerType = None, markerSize = None, thickness = None, line_type = None
    img = copy.copy(image)
    for i in range(len(kps)):
        img = cv.drawMarker(img, (kps[i, 0], kps[i, 1]),
                            color=color,
                            markerType=cv.MARKER_CROSS,
                            markerSize=10,
                            thickness=1,
                            line_type=cv.LINE_AA)
    return img

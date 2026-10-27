import os

import cv2
import numpy as np


def cv_imread(path, flags=cv2.IMREAD_COLOR):
    path = os.fspath(path)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def cv_imwrite(path, image, params=None):
    path = os.fspath(path)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buffer = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buffer.tofile(path)
    return True

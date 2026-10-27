import numpy as np

from ok.util.image_io import cv_imread, cv_imwrite


def test_cv_image_io_supports_chinese_paths(tmp_path):
    folder = tmp_path / "自动化测试"
    path = folder / "模板截图.png"
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :] = (10, 120, 240)

    assert cv_imwrite(path, image)

    loaded = cv_imread(path)

    assert loaded is not None
    assert loaded.shape == image.shape
    assert loaded[0, 0].tolist() == [10, 120, 240]

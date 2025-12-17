from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6 import QtGui


def cv2_bgr_to_qimage(image: np.ndarray) -> QtGui.QImage:
    if image is None:
        return QtGui.QImage()

    if image.ndim == 2:
        if image.dtype != np.uint8:
            img = image.astype(np.uint8)
        else:
            img = image
        h, w = img.shape
        bytes_per_line = w
        return QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)

    if image.ndim == 3 and image.shape[2] == 4:
        if image.dtype != np.uint8:
            img = image.astype(np.uint8)
        else:
            img = image
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        h, w, ch = rgba.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgba.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)

    if image.ndim == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            img = image.astype(np.uint8)
        else:
            img = image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

    return QtGui.QImage()


def cv2_to_qpixmap(image: Optional[np.ndarray]) -> QtGui.QPixmap:
    if image is None:
        return QtGui.QPixmap()
    qimg = cv2_bgr_to_qimage(image)
    return QtGui.QPixmap.fromImage(qimg)

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import CrackDetectionPipeline
from src.pipeline.models import BoxResult


def cv2_to_qpixmap(image: np.ndarray) -> QtGui.QPixmap:
    if image is None:
        return QtGui.QPixmap()
    if len(image.shape) == 2:
        h, w = image.shape
        bytes_per_line = w
        qimg = QtGui.QImage(image.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)


def draw_boxes_on_image(image: np.ndarray, boxes: List[BoxResult]) -> np.ndarray:
    output = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = b.as_tuple()
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{b.prompt[:16]} {b.score:.2f}"
        cv2.putText(output, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return output


class DetectionWorker(QtCore.QThread):
    finished = QtCore.Signal(np.ndarray, list, str)
    failed = QtCore.Signal(str)

    def __init__(self, image_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.pipeline = CrackDetectionPipeline()

    def run(self) -> None:
        try:
            result = self.pipeline.run(str(self.image_path))
            overlay = draw_boxes_on_image(result.overlay_image, result.boxes)
            self.finished.emit(overlay, result.boxes, "Done")
        except Exception as exc:  # pragma: no cover - UI path
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Crack Detector (GroundingDINO + SAM)")
        self.resize(1100, 700)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setStyleSheet("QLabel { background: #222; color: #eee; }")

        self.status = QtWidgets.QLabel("Ready")

        load_btn = QtWidgets.QPushButton("Chọn ảnh")
        load_btn.clicked.connect(self.load_image)
        self.run_btn = QtWidgets.QPushButton("Chạy detect")
        self.run_btn.clicked.connect(self.run_detection)
        self.run_btn.setEnabled(False)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(load_btn)
        top_bar.addWidget(self.run_btn)
        top_bar.addStretch(1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.current_image_path: Path | None = None
        self.worker: DetectionWorker | None = None

    def load_image(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh crack",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return
        self.current_image_path = Path(file_path)
        pixmap = QtGui.QPixmap(file_path)
        if pixmap.isNull():
            self.status.setText("Không mở được ảnh.")
            return
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.status.setText(f"Đã chọn: {self.current_image_path.name}")
        self.run_btn.setEnabled(True)

    def run_detection(self) -> None:
        if not self.current_image_path:
            return
        self.run_btn.setEnabled(False)
        self.status.setText("Đang chạy...")
        self.worker = DetectionWorker(self.current_image_path)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_finished(self, overlay: np.ndarray, boxes: list, msg: str) -> None:
        pixmap = cv2_to_qpixmap(overlay)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.status.setText(f"{msg} - {len(boxes)} boxes")
        self.run_btn.setEnabled(True)

    def on_failed(self, err: str) -> None:
        self.status.setText(f"Lỗi: {err}")
        self.run_btn.setEnabled(True)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


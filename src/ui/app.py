from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import CrackDetectionPipeline
from src.pipeline.models import PipelineResult, RegionResult
from src.ui.utils import cv2_to_qpixmap


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _endpoints(skel: np.ndarray) -> np.ndarray:
    s = (skel > 0).astype(np.uint8)
    k = np.array(
        [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    neigh = cv2.filter2D(s, -1, k)
    return ((neigh == 11) & (s == 1)).astype(np.uint8)


def _overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    base = base_bgr.copy()
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return base
    overlay = base.copy()
    overlay[m > 0] = color
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)


class ImageView(QtWidgets.QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 180)
        self.setStyleSheet("QGraphicsView { background: #222; }")
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._has_image = False

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self.set_pixmap(QtGui.QPixmap())
            return
        self.set_pixmap(cv2_to_qpixmap(image))

    def set_pixmap(self, pixmap: QtGui.QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            self._pixmap_item.setPixmap(QtGui.QPixmap())
            self._scene.setSceneRect(QtCore.QRectF())
            self._has_image = False
            return
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self._has_image = True
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._has_image:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._has_image:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


def _overlay_on_base(base_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    base = base_bgr.copy()
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return base
    overlay = base.copy()
    overlay[m > 0] = color
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)


class RegionInspectorWidget(QtWidgets.QWidget):
    """Region inspector tab for geometry filtering review."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels(
            [
                "region_id",
                "kept",
                "dropped_reason",
                "area",
                "skeleton_length",
                "width_mean",
                "width_variance",
                "touch_border_ratio",
                "orientation_variance",
                "curvature_variance",
                "len_area_ratio",
            ],
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_row_clicked)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.filter_kept = QtWidgets.QCheckBox("Show kept")
        self.filter_kept.setChecked(True)
        self.filter_dropped = QtWidgets.QCheckBox("Show dropped")
        self.filter_dropped.setChecked(True)
        self.filter_reason = QtWidgets.QComboBox()
        self.filter_reason.addItem("All reasons", "")
        self.filter_kept.stateChanged.connect(self._refresh_table)
        self.filter_dropped.stateChanged.connect(self._refresh_table)
        self.filter_reason.currentIndexChanged.connect(self._refresh_table)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(self.filter_kept)
        filter_row.addWidget(self.filter_dropped)
        filter_row.addWidget(QtWidgets.QLabel("Dropped reason:"))
        filter_row.addWidget(self.filter_reason, 1)

        self.view_crop_overlay = ImageView()
        self.view_crop_mask = ImageView()
        self.view_skeleton = ImageView()
        self.view_full_highlight = ImageView()

        imgs_layout = QtWidgets.QHBoxLayout()
        imgs_layout.addWidget(self.view_crop_overlay, 1)
        imgs_layout.addWidget(self.view_crop_mask, 1)
        imgs_layout.addWidget(self.view_skeleton, 1)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(filter_row)
        main_layout.addWidget(self.table, 1)
        main_layout.addLayout(imgs_layout, 1)
        main_layout.addWidget(self.view_full_highlight, 1)
        self.setLayout(main_layout)

        self._result: Optional[PipelineResult] = None
        self._regions: List[RegionResult] = []

    def load_pipeline_result(self, result: PipelineResult) -> None:
        self._result = result
        self._regions = result.regions or []
        # populate reason filter
        reasons = sorted({str(r.dropped_reason) for r in self._regions if r.dropped_reason})
        self.filter_reason.blockSignals(True)
        self.filter_reason.clear()
        self.filter_reason.addItem("All reasons", "")
        for rs in reasons:
            self.filter_reason.addItem(rs, rs)
        self.filter_reason.blockSignals(False)
        self._refresh_table()

    def _refresh_table(self) -> None:
        regs = []
        show_kept = self.filter_kept.isChecked()
        show_dropped = self.filter_dropped.isChecked()
        reason_filter = self.filter_reason.currentData()
        for r in self._regions:
            kept = bool(r.kept)
            if kept and not show_kept:
                continue
            if (not kept) and not show_dropped:
                continue
            if reason_filter:
                if str(r.dropped_reason) != str(reason_filter):
                    continue
            regs.append(r)

        self.table.setRowCount(len(regs))
        for row, reg in enumerate(regs):
            rid = int(reg.region_id)
            kept = bool(reg.kept)
            reason = reg.dropped_reason
            area = int(reg.geometry.area)
            sk_len = float(reg.geometry.skeleton_length)
            width_mean = float(reg.geometry.width_mean)
            width_var = float(reg.geometry.width_var)
            touch_ratio = 0.0
            ori_var = 0.0
            curv_var = float(reg.geometry.curvature_var)
            len_area_ratio = float(reg.geometry.length_area_ratio)
            items = [
                QtWidgets.QTableWidgetItem(str(rid)),
                QtWidgets.QTableWidgetItem("yes" if kept else "no"),
                QtWidgets.QTableWidgetItem("" if kept else ("" if reason is None else str(reason))),
                QtWidgets.QTableWidgetItem(str(area)),
                QtWidgets.QTableWidgetItem(f"{sk_len:.2f}"),
                QtWidgets.QTableWidgetItem(f"{width_mean:.2f}"),
                QtWidgets.QTableWidgetItem(f"{width_var:.2f}"),
                QtWidgets.QTableWidgetItem(f"{touch_ratio:.3f}"),
                QtWidgets.QTableWidgetItem(f"{ori_var:.3f}"),
                QtWidgets.QTableWidgetItem(f"{curv_var:.3f}"),
                QtWidgets.QTableWidgetItem(f"{len_area_ratio:.4f}"),
            ]
            for c, it in enumerate(items):
                it.setData(QtCore.Qt.UserRole, rid)
                if not kept:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#999")))
                self.table.setItem(row, c, it)
        self.table.resizeColumnsToContents()

    def _on_row_clicked(self, row: int, col: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        rid = item.data(QtCore.Qt.UserRole)
        if rid is None:
            return
        self._show_region(int(rid))

    def _show_region(self, rid: int) -> None:
        if self._result is None:
            return
        reg = next((r for r in self._regions if int(r.region_id) == rid), None)
        if not reg:
            return

        base = self._result.images.get("input")
        if base is None:
            base = self._result.images.get("preprocessed")

        if base is None or not isinstance(base, np.ndarray) or reg.mask is None:
            self.view_crop_overlay.set_image(None)
            self.view_crop_mask.set_image(None)
            self.view_skeleton.set_image(None)
            self.view_full_highlight.set_image(None)
            return

        base_bgr = base if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        overlay_full = _overlay_on_base(base_bgr, reg.mask, (0, 0, 255), 0.4)
        self.view_full_highlight.set_image(overlay_full)

        x1, y1, x2, y2 = [int(v) for v in reg.bbox]
        x1 = max(0, min(x1, base_bgr.shape[1] - 1))
        x2 = max(0, min(x2, base_bgr.shape[1]))
        y1 = max(0, min(y1, base_bgr.shape[0] - 1))
        y2 = max(0, min(y2, base_bgr.shape[0]))
        if x2 <= x1 or y2 <= y1:
            self.view_crop_overlay.set_image(None)
            self.view_crop_mask.set_image(None)
            self.view_skeleton.set_image(None)
            return

        crop_img = base_bgr[y1:y2, x1:x2]
        crop_mask = (reg.mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        crop_overlay = _overlay_on_base(crop_img, crop_mask, (0, 0, 255), 0.4)
        self.view_crop_overlay.set_image(crop_overlay)

        cm_vis = (crop_mask * 255).astype(np.uint8)
        self.view_crop_mask.set_image(cm_vis)
        sk = _skeletonize(crop_mask)
        sk_rgb = cv2.cvtColor((sk * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        sk_rgb[sk > 0] = (0, 255, 255)
        self.view_skeleton.set_image(sk_rgb)


class DetectionWorker(QtCore.QThread):
    finished = QtCore.Signal(object, str)
    failed = QtCore.Signal(str)

    def __init__(self, image_path: Path, runtime: dict, parent=None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.runtime = runtime
        self.pipeline = CrackDetectionPipeline()

    def run(self) -> None:
        try:
            result = self.pipeline.run(str(self.image_path), runtime=self.runtime)
            self.finished.emit(result, "Done")
        except Exception as exc:  # pragma: no cover - UI path
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Crack Detector (GroundingDINO + SAM)")
        self.resize(1100, 700)

        self.status = QtWidgets.QLabel("Ready")

        self.tabs = QtWidgets.QTabWidget()
        self.view_input = ImageView()
        self.view_preprocess = ImageView()
        self.view_dino = ImageView()
        self.view_sam = ImageView()
        self.view_geometry_input = ImageView()
        self.view_geometry_kept = ImageView()
        self.view_final = ImageView()
        self.region_inspector = RegionInspectorWidget()
        self.tabs.addTab(self.view_input, "Input")
        self.tabs.addTab(self.view_preprocess, "Preprocess")
        self.tabs.addTab(self.view_dino, "DINO Boxes")
        self.tabs.addTab(self.view_sam, "SAM Raw Mask")
        self.tabs.addTab(self.view_geometry_input, "Geometry Input")
        self.tabs.addTab(self.view_geometry_kept, "Geometry Kept")
        self.tabs.addTab(self.view_final, "Final Overlay")
        self.tabs.addTab(self.region_inspector, "Region Inspector")

        self.chk_enable_geom = QtWidgets.QCheckBox("Enable geometry filter")
        self.chk_enable_geom.setChecked(True)
        self.chk_enable_geom.stateChanged.connect(self._schedule_rerun)

        self.grp_variants = QtWidgets.QGroupBox("Preprocess")
        v_layout = QtWidgets.QHBoxLayout()
        self.var_base = QtWidgets.QCheckBox("base")
        self.var_blur = QtWidgets.QCheckBox("blur_boost")
        self.var_ridge = QtWidgets.QCheckBox("ridge")
        self.var_base.setChecked(True)
        for cb in (self.var_base, self.var_blur, self.var_ridge):
            cb.stateChanged.connect(self._schedule_rerun)
            v_layout.addWidget(cb)
        self.grp_variants.setLayout(v_layout)

        self.cmb_prompt_mode = QtWidgets.QComboBox()
        self.cmb_prompt_mode.addItem("disabled")
        self.cmb_prompt_mode.addItem("one_pass")
        self.cmb_prompt_mode.addItem("multi_pass")
        self.cmb_prompt_mode.currentIndexChanged.connect(self._schedule_rerun)

        self.btn_save_bundle = QtWidgets.QPushButton("Save Debug Bundle")
        self.btn_save_bundle.clicked.connect(self.save_debug_bundle)

        self.chk_table_only_kept = QtWidgets.QCheckBox("Table: only kept")
        self.chk_table_only_dropped = QtWidgets.QCheckBox("Table: only dropped")
        self.chk_table_only_kept.stateChanged.connect(self.refresh_regions_table)
        self.chk_table_only_dropped.stateChanged.connect(self.refresh_regions_table)

        load_btn = QtWidgets.QPushButton("Chọn ảnh")
        load_btn.clicked.connect(self.load_image)
        self.run_btn = QtWidgets.QPushButton("Chạy detect")
        self.run_btn.clicked.connect(self.run_detection)
        self.run_btn.setEnabled(False)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(load_btn)
        top_bar.addWidget(self.run_btn)
        top_bar.addWidget(self.chk_enable_geom)
        top_bar.addWidget(self.chk_table_only_kept)
        top_bar.addWidget(self.chk_table_only_dropped)
        top_bar.addWidget(self.grp_variants)
        top_bar.addWidget(QtWidgets.QLabel("Prompt mode:"))
        top_bar.addWidget(self.cmb_prompt_mode)
        top_bar.addWidget(self.btn_save_bundle)
        top_bar.addStretch(1)

        self.metrics_group = QtWidgets.QGroupBox("Metrics")
        form = QtWidgets.QFormLayout()
        self.lbl_regions_before = QtWidgets.QLabel("-")
        self.lbl_regions_after = QtWidgets.QLabel("-")
        self.lbl_avg_width = QtWidgets.QLabel("-")
        self.lbl_avg_length = QtWidgets.QLabel("-")
        self.lbl_final_conf = QtWidgets.QLabel("-")
        self.lbl_fallback = QtWidgets.QLabel("-")
        form.addRow("Num regions (before):", self.lbl_regions_before)
        form.addRow("Num regions (after):", self.lbl_regions_after)
        form.addRow("Avg width:", self.lbl_avg_width)
        form.addRow("Avg length:", self.lbl_avg_length)
        form.addRow("Final confidence:", self.lbl_final_conf)
        form.addRow("Fallback used:", self.lbl_fallback)
        self.metrics_group.setLayout(form)

        self.details_group = QtWidgets.QGroupBox("Region Details")
        details_form = QtWidgets.QFormLayout()
        self._detail_labels: dict[str, QtWidgets.QLabel] = {}
        for key in (
            "region_id",
            "kept",
            "dropped_reason",
            "prompt_name",
            "variant_name",
            "area",
            "skeleton_length",
            "endpoint_count",
            "width_mean",
            "width_var",
            "length_area_ratio",
            "curvature_var",
            "dino_conf",
        ):
            lbl = QtWidgets.QLabel("-")
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self._detail_labels[key] = lbl
            details_form.addRow(f"{key}:", lbl)
        self.details_group.setLayout(details_form)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            [
                "region_id",
                "kept",
                "dropped_reason",
                "area",
                "skeleton_length",
                "width_mean",
                "endpoint_count",
                "dino_conf",
            ],
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self.on_region_clicked)
        self.table.horizontalHeader().setStretchLastSection(True)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.metrics_group)
        right.addWidget(self.details_group)
        right.addWidget(self.table, 1)
        right.addWidget(self.status)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.tabs, 2)
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right)
        main_layout.addWidget(right_container, 1)

        outer = QtWidgets.QVBoxLayout()
        outer.addLayout(top_bar)
        outer.addLayout(main_layout, 1)

        container = QtWidgets.QWidget()
        container.setLayout(outer)
        self.setCentralWidget(container)

        self.current_image_path: Path | None = None
        self.worker: DetectionWorker | None = None
        self.last_result: PipelineResult | None = None
        self.selected_region_id: int | None = None

        self._rerun_timer = QtCore.QTimer(self)
        self._rerun_timer.setSingleShot(True)
        self._rerun_timer.timeout.connect(self._rerun_if_possible)
        self.tabs.currentChanged.connect(self._refresh_highlight)

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
        self.view_input.set_pixmap(pixmap)
        self.status.setText(f"Đã chọn: {self.current_image_path.name}")
        self.run_btn.setEnabled(True)

    def run_detection(self) -> None:
        if not self.current_image_path:
            return
        self.run_btn.setEnabled(False)
        self.status.setText("Đang chạy...")
        runtime = {
            "enable_geometry_filter": bool(self.chk_enable_geom.isChecked()),
            "debug_enabled": False,
            "preprocess_variants": self._selected_variants(),
            "damage_prompt_mode": str(self.cmb_prompt_mode.currentText()),
        }
        self.worker = DetectionWorker(self.current_image_path, runtime)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_finished(self, result_obj: object, msg: str) -> None:
        if not isinstance(result_obj, PipelineResult):
            raise RuntimeError("Invalid pipeline result")
        self.last_result = result_obj
        self.selected_region_id = None
        self._set_region_details(None)

        self.view_input.set_image(self.last_result.images.get("input"))
        self.view_preprocess.set_image(self.last_result.images.get("preprocessed"))
        self.view_dino.set_image(self.last_result.images.get("dino_boxes_overlay") or self.last_result.images.get("dino_boxes"))
        self.view_sam.set_image(self.last_result.images.get("sam_raw_mask_viz") or self.last_result.images.get("sam_raw_mask"))
        self.view_geometry_input.set_image(self.last_result.images.get("geometry_input_viz"))
        self.view_geometry_kept.set_image(self.last_result.images.get("geometry_kept_mask_viz"))
        self.view_final.set_image(self.last_result.images.get("final_overlay"))
        self.refresh_metrics()
        self.refresh_regions_table()
        self.region_inspector.load_pipeline_result(self.last_result)
        self._refresh_highlight()

        self.status.setText(f"{msg}")
        self.run_btn.setEnabled(True)

    def on_failed(self, err: str) -> None:
        self.status.setText(f"Lỗi: {err}")
        self.run_btn.setEnabled(True)

    def refresh_metrics(self) -> None:
        if self.last_result is None:
            return
        m = self.last_result.metrics
        self.lbl_regions_before.setText(str(m.get("num_regions_before", "-")))
        self.lbl_regions_after.setText(str(m.get("num_regions_after", "-")))
        self.lbl_avg_width.setText(f"{float(m.get('avg_width', 0.0)):.2f}")
        self.lbl_avg_length.setText(f"{float(m.get('avg_length', 0.0)):.2f}")
        self.lbl_final_conf.setText(f"{float(m.get('final_confidence', 0.0)):.3f}")
        self.lbl_fallback.setText("Yes" if bool(m.get("fallback_used", False)) else "No")

    def refresh_regions_table(self) -> None:
        if self.last_result is None:
            return
        regions_all = list(self.last_result.regions or [])
        only_kept = bool(self.chk_table_only_kept.isChecked())
        only_dropped = bool(self.chk_table_only_dropped.isChecked())
        regions: list[RegionResult] = []
        for reg in regions_all:
            if only_kept and not bool(reg.kept):
                continue
            if only_dropped and bool(reg.kept):
                continue
            regions.append(reg)

        regions.sort(key=lambda r: (not bool(r.kept), -float(r.geometry.skeleton_length)))

        self.table.setRowCount(len(regions))
        for r, reg in enumerate(regions):
            items = [
                QtWidgets.QTableWidgetItem(str(int(reg.region_id))),
                QtWidgets.QTableWidgetItem("true" if reg.kept else "false"),
                QtWidgets.QTableWidgetItem("" if reg.dropped_reason is None else str(reg.dropped_reason)),
                QtWidgets.QTableWidgetItem(str(int(reg.geometry.area))),
                QtWidgets.QTableWidgetItem(f"{float(reg.geometry.skeleton_length):.2f}"),
                QtWidgets.QTableWidgetItem(f"{float(reg.geometry.width_mean):.2f}"),
                QtWidgets.QTableWidgetItem(str(int(reg.geometry.endpoint_count))),
                QtWidgets.QTableWidgetItem("" if reg.scores.dino_conf is None else f"{float(reg.scores.dino_conf):.3f}"),
            ]
            for c, it in enumerate(items):
                it.setData(QtCore.Qt.UserRole, int(reg.region_id))
                self.table.setItem(r, c, it)
                if not reg.kept:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#aaa")))

        self.table.resizeColumnsToContents()

    def on_region_clicked(self, row: int, col: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        region_id = item.data(QtCore.Qt.UserRole)
        if region_id is None:
            return
        self.selected_region_id = int(region_id)
        reg = next((r for r in (self.last_result.regions or []) if int(r.region_id) == int(self.selected_region_id)), None) if self.last_result is not None else None
        self._set_region_details(reg)
        self._refresh_highlight()

    def _set_region_details(self, reg: RegionResult | None) -> None:
        if reg is None:
            for lbl in self._detail_labels.values():
                lbl.setText("-")
            return

        self._detail_labels["region_id"].setText(str(int(reg.region_id)))
        self._detail_labels["kept"].setText("true" if reg.kept else "false")
        self._detail_labels["dropped_reason"].setText("" if reg.dropped_reason is None else str(reg.dropped_reason))
        self._detail_labels["prompt_name"].setText("" if reg.meta.prompt_name is None else str(reg.meta.prompt_name))
        self._detail_labels["variant_name"].setText("" if reg.meta.variant_name is None else str(reg.meta.variant_name))
        self._detail_labels["area"].setText(str(int(reg.geometry.area)))
        self._detail_labels["skeleton_length"].setText(f"{float(reg.geometry.skeleton_length):.2f}")
        self._detail_labels["endpoint_count"].setText(str(int(reg.geometry.endpoint_count)))
        self._detail_labels["width_mean"].setText(f"{float(reg.geometry.width_mean):.2f}")
        self._detail_labels["width_var"].setText(f"{float(reg.geometry.width_var):.2f}")
        self._detail_labels["length_area_ratio"].setText(f"{float(reg.geometry.length_area_ratio):.4f}")
        self._detail_labels["curvature_var"].setText(f"{float(reg.geometry.curvature_var):.4f}")
        self._detail_labels["dino_conf"].setText("" if reg.scores.dino_conf is None else f"{float(reg.scores.dino_conf):.3f}")

    def _selected_variants(self) -> list[str]:
        out: list[str] = []
        if self.var_base.isChecked():
            out.append("base")
        if self.var_blur.isChecked():
            out.append("blur_boost")
        if self.var_ridge.isChecked():
            out.append("ridge")
        if not out:
            out.append("base")
        return out

    def _schedule_rerun(self) -> None:
        self._rerun_timer.start(350)

    def _rerun_if_possible(self) -> None:
        if self.current_image_path is None:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.run_btn.isEnabled():
            return
        self.run_detection()

    def _active_stage_key(self) -> str:
        w = self.tabs.currentWidget()
        if w is self.view_input:
            return "input"
        if w is self.view_preprocess:
            return "preprocessed"
        if w is self.view_dino:
            return "dino_boxes_overlay"
        if w is self.view_sam:
            return "sam_raw_mask_viz"
        if w is self.view_geometry_input:
            return "geometry_input_viz"
        if w is self.view_geometry_kept:
            return "geometry_kept_mask_viz"
        if w is self.view_final:
            return "final_overlay"
        return "final_overlay"

    def _refresh_highlight(self) -> None:
        if self.last_result is None:
            return
        stage_key = self._active_stage_key()
        base = self.last_result.images.get(stage_key)
        if base is None:
            base = self.last_result.images.get("final_overlay")
        if base is None:
            return

        out = base.copy() if isinstance(base, np.ndarray) else None
        if out is None:
            return

        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        if self.selected_region_id is not None:
            reg = next((r for r in (self.last_result.regions or []) if int(r.region_id) == int(self.selected_region_id)), None)
            if reg is not None and isinstance(reg.mask, np.ndarray):
                mask = reg.mask
                if mask.shape[:2] != out.shape[:2]:
                    mask = cv2.resize((mask > 0).astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
                out = _overlay_on_base(out, mask, (0, 255, 255), 0.35)
                x1, y1, x2, y2 = [int(v) for v in reg.bbox]
                scale = float(self.last_result.metrics.get("preprocess_scale", 1.0) or 1.0)
                if stage_key != "input" and scale > 0:
                    x1 = int(x1 * scale)
                    x2 = int(x2 * scale)
                    y1 = int(y1 * scale)
                    y2 = int(y2 * scale)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                if reg.dropped_reason:
                    cv2.putText(out, str(reg.dropped_reason), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        if stage_key == "input":
            self.view_input.set_image(out)
        elif stage_key == "preprocessed":
            self.view_preprocess.set_image(out)
        elif stage_key == "dino_boxes_overlay":
            self.view_dino.set_image(out)
        elif stage_key == "sam_raw_mask_viz":
            self.view_sam.set_image(out)
        elif stage_key == "geometry_input_viz":
            self.view_geometry_input.set_image(out)
        elif stage_key == "geometry_kept_mask_viz":
            self.view_geometry_kept.set_image(out)
        else:
            self.view_final.set_image(out)

    def save_debug_bundle(self) -> None:
        if self.last_result is None:
            return
        export_masks, export_crops = self._ask_export_region_assets()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder", str(Path.cwd()))
        if not out_dir:
            return
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        meta_path = out_path / "pipeline_result.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.last_result.to_dict(shallow=True), f, indent=2)

        stage_keys = [
            "input",
            "preprocessed",
            "dino_boxes_overlay",
            "sam_raw_mask_viz",
            "geometry_input_viz",
            "geometry_kept_mask_viz",
            "final_overlay",
        ]
        for k in stage_keys:
            img = self.last_result.images.get(k)
            if img is None or not isinstance(img, np.ndarray):
                continue
            ext = ".png" if img.ndim == 2 else ".jpg"
            cv2.imwrite(str(out_path / f"{k}{ext}"), img)

        if export_masks or export_crops:
            regions_dir = out_path / "regions"
            regions_dir.mkdir(parents=True, exist_ok=True)

            base_img = self.last_result.images.get("input")
            if base_img is None:
                base_img = self.last_result.images.get("preprocessed")
            if base_img is not None and base_img.ndim == 2:
                base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

            for reg in (self.last_result.regions or []):
                rid = int(reg.region_id)
                if export_masks and isinstance(reg.mask, np.ndarray):
                    mask_vis = ((reg.mask > 0).astype(np.uint8) * 255)
                    cv2.imwrite(str(regions_dir / f"region_{rid:04d}_mask.png"), mask_vis)

                if export_crops and base_img is not None:
                    x1, y1, x2, y2 = [int(v) for v in reg.bbox]
                    x1 = max(0, min(x1, base_img.shape[1] - 1))
                    x2 = max(0, min(x2, base_img.shape[1]))
                    y1 = max(0, min(y1, base_img.shape[0] - 1))
                    y2 = max(0, min(y2, base_img.shape[0]))
                    if x2 > x1 and y2 > y1:
                        crop = base_img[y1:y2, x1:x2]
                        cv2.imwrite(str(regions_dir / f"region_{rid:04d}_crop.jpg"), crop)

    def _ask_export_region_assets(self) -> tuple[bool, bool]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Save Debug Bundle")
        layout = QtWidgets.QVBoxLayout()
        chk_masks = QtWidgets.QCheckBox("Export per-region masks")
        chk_crops = QtWidgets.QCheckBox("Export per-region crops")
        chk_masks.setChecked(True)
        chk_crops.setChecked(True)
        layout.addWidget(chk_masks)
        layout.addWidget(chk_crops)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        dlg.setLayout(layout)

        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return False, False
        return bool(chk_masks.isChecked()), bool(chk_crops.isChecked())


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


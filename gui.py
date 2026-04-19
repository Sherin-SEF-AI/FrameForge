"""
gui.py
──────
MainWindow and all supporting classes for FrameForge.

New in this version
-------------------
EditMode              — Enum for the three correction modes (SELECT / DRAW_BOX /
                        DRAW_POLYGON).
FrameViewer           — Interactive canvas: click-to-select, draw-box drag,
                        draw-polygon click sequence, selection highlight.
ConfidenceHistogramDialog — QPainter histogram of detection confidences.
ClassFilterDialog     — Per-class include/exclude checkboxes.
ReviewDialog          — Table of exported label files with seek-to-frame.
StatisticsDialog      — Dataset-level class distribution bar chart.
AddDetectionDialog    — Assign class + confidence to a manually drawn shape.
ExportAllWorker       — Updated to accept stride and enabled_classes.
MainWindow            — Corrections section, stride spinbox, class filter,
                        CVAT/LabelStudio export, keyboard shortcuts.
"""

import base64
import copy
import datetime
import enum
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QScrollArea, QDialog,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QSlider, QListWidget, QListWidgetItem, QLineEdit, QTextEdit,
    QProgressDialog, QMessageBox, QFileDialog, QRadioButton,
    QSizePolicy, QStatusBar, QFrame, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush,
    QKeySequence, QShortcut,
)

from video_handler import VideoHandler
from inference_engine import (
    InferenceEngine, InferenceResult,
    GroundedSAMEngine, SemanticSegmentationEngine, SemanticResult,
    SAMBoxRefiner, SAM2PropagationEngine,
)
from export_handler import ExportHandler
from taxonomy import GROUNDED_SAM_PROMPT, CLASS_NAMES as TAXONOMY_CLASS_NAMES

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Enums                                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class EditMode(enum.Enum):
    """Correction mode active in the FrameViewer canvas."""
    SELECT      = "select"
    DRAW_BOX    = "draw_box"
    DRAW_POLYGON = "draw_polygon"


# ═══════════════════════════════════════════════════════════════════════════ #
#  Worker threads                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class ModelLoaderWorker(QThread):
    """
    Background thread that loads a YOLO / FastSAM model without blocking
    the GUI event loop.

    Signals
    -------
    finished : (bool, str)
        ``True`` on success with a status message; ``False`` with an error.
    """

    finished = pyqtSignal(bool, str)

    def __init__(self, engine: InferenceEngine, model_name: str):
        """
        Parameters
        ----------
        engine : InferenceEngine
            Shared inference engine instance.
        model_name : str
            Weights filename to load.
        """
        super().__init__()
        self._engine     = engine
        self._model_name = model_name

    def run(self):
        """Load the model and emit ``finished``."""
        try:
            ok = self._engine.load_model(self._model_name)
            if ok:
                self.finished.emit(True, f"Loaded on {self._engine.device().upper()}")
            else:
                self.finished.emit(False, f"Failed to load {self._model_name}")
        except Exception as exc:
            self.finished.emit(False, str(exc))


# --------------------------------------------------------------------------- #

class InferenceWorker(QThread):
    """
    Background thread for single-frame segmentation inference.

    Signals
    -------
    finished : (object, object)
        ``(InferenceResult, annotated_bgr_frame)`` on success.
    error : str
        Exception description on failure.
    """

    finished = pyqtSignal(object)   # InferenceResult only
    error    = pyqtSignal(str)

    def __init__(
        self, engine: InferenceEngine, frame: np.ndarray,
        conf: float, iou: float, imgsz: int = 640,
    ):
        """
        Parameters
        ----------
        engine : InferenceEngine
            Loaded inference engine.
        frame : np.ndarray
            BGR frame (caller must pass a copy).
        conf : float
            Confidence threshold.
        iou : float
            IoU threshold.
        imgsz : int
            Inference image size (square).
        """
        super().__init__()
        self._engine = engine
        self._frame  = frame
        self._conf   = conf
        self._iou    = iou
        self._imgsz  = imgsz

    def run(self):
        """Run inference and emit ``finished`` or ``error``."""
        try:
            result = self._engine.infer(
                self._frame, self._conf, self._iou, self._imgsz
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# --------------------------------------------------------------------------- #

class ExportAllWorker(QThread):
    """
    Background thread that exports every (or every Nth) frame with inference.

    Signals
    -------
    progress : (int, int)   current, total
    finished : str          completion message
    error    : str          error description
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(
        self,
        export_handler: ExportHandler,
        video_path: str,
        total_frames: int,
        inference_engine: InferenceEngine,
        conf: float,
        iou: float,
        stride: int = 1,
        enabled_classes: "set[int] | None" = None,
    ):
        super().__init__()
        self._export        = export_handler
        self._video_path    = video_path
        self._total_frames  = total_frames
        self._engine        = inference_engine
        self._conf          = conf
        self._iou           = iou
        self._stride        = stride
        self._classes       = enabled_classes
        self._cancelled     = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """Open a private VideoCapture (thread-safe) and export frames."""
        cap = None
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error.emit(f"Cannot open video: {self._video_path}")
                return
            frames_to_export = list(range(0, self._total_frames, max(1, self._stride)))
            n = len(frames_to_export)
            for pos, idx in enumerate(frames_to_export):
                if self._cancelled:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    try:
                        result = self._engine.infer(frame, conf=self._conf, iou=self._iou)
                        if self._classes is not None:
                            result = ExportHandler.filter_result(result, self._classes)
                        self._export.save_yolo(frame, result, idx)
                    except Exception as exc:
                        logger.warning("Export failed at frame %d: %s", idx, exc)
                self.progress.emit(pos + 1, n)
            self.finished.emit("All frames exported successfully.")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


# ═══════════════════════════════════════════════════════════════════════════ #
#  New engine worker threads                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

class GSAMModelLoader(QThread):
    """Background loader for GroundedSAMEngine."""
    finished = pyqtSignal(bool, str)

    def __init__(self, engine: GroundedSAMEngine, sam_ckpt: str, sam_variant: str):
        super().__init__()
        self._engine  = engine
        self._ckpt    = sam_ckpt
        self._variant = sam_variant

    def run(self):
        ok, msg = self._engine.load(self._ckpt, self._variant)
        self.finished.emit(ok, msg)


class GSAMInferenceWorker(QThread):
    """Background inference worker for GroundedSAMEngine."""
    finished = pyqtSignal(object)   # InferenceResult only
    error    = pyqtSignal(str)

    def __init__(
        self,
        engine:         GroundedSAMEngine,
        frame:          np.ndarray,
        text_prompt:    str,
        box_threshold:  float,
        text_threshold: float,
    ):
        super().__init__()
        self._engine    = engine
        self._frame     = frame
        self._prompt    = text_prompt
        self._box_thr   = box_threshold
        self._text_thr  = text_threshold

    def run(self):
        try:
            result = self._engine.infer(
                self._frame, self._prompt, self._box_thr, self._text_thr
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class SemanticModelLoader(QThread):
    """Background loader for SemanticSegmentationEngine."""
    finished = pyqtSignal(bool, str)

    def __init__(self, engine: SemanticSegmentationEngine, model_id: str):
        super().__init__()
        self._engine   = engine
        self._model_id = model_id

    def run(self):
        ok, msg = self._engine.load(self._model_id)
        self.finished.emit(ok, msg)


class SemanticInferenceWorker(QThread):
    """Background inference worker for SemanticSegmentationEngine."""
    finished = pyqtSignal(object)   # SemanticResult
    error    = pyqtSignal(str)

    def __init__(self, engine: SemanticSegmentationEngine, frame: np.ndarray):
        super().__init__()
        self._engine = engine
        self._frame  = frame

    def run(self):
        try:
            result = self._engine.infer(self._frame)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class SemanticBatchWorker(QThread):
    """
    Runs semantic segmentation on every frame in the video sequentially.

    Opens its own private cv2.VideoCapture from the video path so it never
    shares the main thread's capture object — cv2.VideoCapture is not
    thread-safe and sharing it causes a segfault.

    Results are emitted per-frame via frame_done; the worker never accumulates
    them, avoiding OOM on long videos. finished emits only the frame count.
    """
    progress   = pyqtSignal(int, int)    # (done, total)
    frame_done = pyqtSignal(int, object) # (frame_idx, SemanticResult)
    finished   = pyqtSignal(int)         # total frames processed
    error      = pyqtSignal(str)

    def __init__(self, engine: SemanticSegmentationEngine,
                 video_path: str, total_frames: int, stride: int = 1,
                 already_cached: "set[int] | None" = None):
        super().__init__()
        self._engine          = engine
        self._video_path      = video_path
        self._total_frames    = total_frames
        self._stride          = max(1, stride)
        self._cancel          = False
        self._already_cached  = already_cached or set()

    def cancel(self):
        self._cancel = True

    def run(self):
        import torch as _torch
        cap = None
        try:
            # Own private capture — no sharing with the main thread
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error.emit(f"Cannot open video: {self._video_path}")
                return

            indices = list(range(0, self._total_frames, self._stride))
            processed = 0
            for done, idx in enumerate(indices, 1):
                if self._cancel:
                    break
                if idx in self._already_cached:
                    self.progress.emit(done, len(indices))
                    processed += 1
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                result = self._engine.infer(frame)
                self.frame_done.emit(idx, result)
                self.progress.emit(done, len(indices))
                processed += 1
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            self.finished.emit(processed)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


class SAMRefineLoadWorker(QThread):
    """Loads SAMBoxRefiner in the background."""
    finished = pyqtSignal(bool, str)

    def __init__(self, refiner: SAMBoxRefiner, checkpoint: str,
                 variant: str, use_mobile: bool):
        super().__init__()
        self._refiner    = refiner
        self._checkpoint = checkpoint
        self._variant    = variant
        self._use_mobile = use_mobile

    def run(self):
        ok, msg = self._refiner.load(self._checkpoint, self._variant, self._use_mobile)
        self.finished.emit(ok, msg)


class SAMRefineWorker(QThread):
    """Runs SAMBoxRefiner.refine() on a single frame."""
    finished = pyqtSignal(object)   # InferenceResult
    error    = pyqtSignal(str)

    def __init__(self, refiner: SAMBoxRefiner, frame: np.ndarray,
                 result: InferenceResult):
        super().__init__()
        self._refiner = refiner
        self._frame   = frame
        self._result  = result

    def run(self):
        try:
            refined = self._refiner.refine(self._frame, self._result)
            self.finished.emit(refined)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainingWorker(QThread):
    """
    Runs YOLO training as a subprocess and streams output line by line.
    Emits log_line for each output line, finished on success, error on failure.
    """
    log_line = pyqtSignal(str)
    finished = pyqtSignal(str)   # message with best.pt path
    error    = pyqtSignal(str)

    def __init__(self, data_yaml: str, base_model: str, epochs: int,
                 batch: int, imgsz: int, project_dir: str):
        super().__init__()
        self._data_yaml   = data_yaml
        self._base_model  = base_model
        self._epochs      = epochs
        self._batch       = batch
        self._imgsz       = imgsz
        self._project_dir = project_dir
        self._process     = None

    def cancel(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run(self):
        cmd = [
            sys.executable, "-m", "ultralytics", "train",
            f"model={self._base_model}",
            f"data={self._data_yaml}",
            f"epochs={self._epochs}",
            f"batch={self._batch}",
            f"imgsz={self._imgsz}",
            f"project={self._project_dir}",
            "name=frameforge_train",
            "exist_ok=True",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(self._process.stdout.readline, ""):
                self.log_line.emit(line.rstrip())
            self._process.wait()
            if self._process.returncode == 0:
                best = os.path.join(
                    self._project_dir, "frameforge_train", "weights", "best.pt"
                )
                if os.path.exists(best):
                    self.finished.emit(f"Training complete. Best model: {best}")
                else:
                    self.finished.emit("Training complete.")
            else:
                self.error.emit(
                    f"Training process exited with code {self._process.returncode}"
                )
        except Exception as exc:
            self.error.emit(str(exc))


# --------------------------------------------------------------------------- #

class PropagationWorker(QThread):
    """
    Background thread for SAM 2 temporal mask propagation.

    Extracts frames from the source video into a temp directory, runs the SAM 2
    video predictor seeded with the keyframe annotations, and emits each
    propagated frame result as it arrives so the UI stays responsive.

    Signals
    -------
    progress   : (int, int)    (done, total) frame count
    frame_done : (int, object) (global_frame_idx, InferenceResult)
    finished   : int           total number of frames propagated
    error      : str           exception message on failure
    """

    progress   = pyqtSignal(int, int)
    frame_done = pyqtSignal(int, object)
    finished   = pyqtSignal(int)
    error      = pyqtSignal(str)

    def __init__(
        self,
        engine:          SAM2PropagationEngine,
        video_path:      str,
        keyframe_idx:    int,
        keyframe_result: InferenceResult,
        start_idx:       int,
        end_idx:         int,
        reverse:         bool = False,
    ):
        super().__init__()
        self._engine          = engine
        self._video_path      = video_path
        self._keyframe_idx    = keyframe_idx
        self._keyframe_result = keyframe_result
        self._start_idx       = start_idx
        self._end_idx         = end_idx
        self._reverse         = reverse
        self._cancel_flag     = [False]   # mutable flag shared with engine

    def cancel(self):
        self._cancel_flag[0] = True

    def run(self):
        done_count = [0]
        total_frames = self._end_idx - self._start_idx + 1

        def on_progress(done: int, total: int):
            done_count[0] = done
            self.progress.emit(done, total)

        try:
            results = self._engine.propagate(
                video_path      = self._video_path,
                keyframe_idx    = self._keyframe_idx,
                keyframe_result = self._keyframe_result,
                start_idx       = self._start_idx,
                end_idx         = self._end_idx,
                reverse         = self._reverse,
                progress_cb     = on_progress,
                cancel_flag     = self._cancel_flag,
            )
            for global_idx, result in sorted(results.items()):
                self.frame_done.emit(global_idx, result)
            self.finished.emit(len(results))
        except Exception as exc:
            self.error.emit(str(exc))


class ActiveLearningWorker(QThread):
    """
    Scans the in-memory frame store and exported label files to flag frames
    that are likely to need human review.
    """
    finished = pyqtSignal(list)   # list[dict]

    def __init__(
        self,
        frame_store:    dict,
        labels_dir:     str,
        conf_threshold: float,
        total_frames:   int,
    ):
        super().__init__()
        self._store      = frame_store
        self._labels_dir = labels_dir
        self._conf_thr   = conf_threshold
        self._total      = total_frames

    def run(self):
        flagged: list[dict] = []
        seen: set[int]      = set()

        # --- Scan in-memory frame store (has confidence data) ---
        for fid, result in self._store.items():
            seen.add(fid)
            issues:   list[str] = []
            det_count = len(result.class_ids)
            avg_conf  = (
                sum(result.confidences) / len(result.confidences)
                if result.confidences else 0.0
            )

            if det_count == 0:
                issues.append("No detections")
            if avg_conf > 0 and avg_conf < self._conf_thr:
                issues.append(f"Low confidence ({avg_conf:.2f})")
            # Flag single-class frames with many detections (likely noise)
            if det_count > 10 and len(set(result.class_ids)) == 1:
                issues.append("Single-class flood")

            if issues:
                priority = 1.0 - avg_conf if avg_conf > 0 else 1.0
                flagged.append({
                    "frame_idx": fid,
                    "priority":  round(priority, 3),
                    "issues":    ", ".join(issues),
                    "det_count": det_count,
                    "avg_conf":  round(avg_conf, 3),
                    "source":    "memory",
                })

        # --- Scan exported label files for frames not in memory ---
        ldir = Path(self._labels_dir)
        if ldir.exists():
            for fpath in sorted(ldir.glob("frame_*.txt")):
                try:
                    idx = int(fpath.stem.replace("frame_", ""))
                except ValueError:
                    continue
                if idx in seen:
                    continue
                lines = [
                    ln for ln in fpath.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
                issues = []
                if not lines:
                    issues.append("Empty label file")
                if issues:
                    flagged.append({
                        "frame_idx": idx,
                        "priority":  1.0,
                        "issues":    ", ".join(issues),
                        "det_count": len(lines),
                        "avg_conf":  0.0,
                        "source":    "disk",
                    })

        flagged.sort(key=lambda x: x["priority"], reverse=True)
        self.finished.emit(flagged)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Active Learning Dialog                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class ActiveLearningDialog(QDialog):
    """
    Shows the prioritised list of frames flagged for human review by the
    ActiveLearningWorker.  Double-click or 'Go to Frame' seeks to that frame.
    """

    seek_requested = pyqtSignal(int)

    def __init__(self, flagged: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Active Learning Review Queue")
        self.setMinimumSize(620, 440)
        layout = QVBoxLayout(self)

        lbl = QLabel(
            f"{len(flagged)} frame(s) flagged for review  "
            f"(sorted by priority — highest first)"
        )
        lbl.setStyleSheet("color:#DDAA00;padding:4px;")
        layout.addWidget(lbl)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Frame #", "Priority", "Detections", "Avg Conf", "Issues", "Source"]
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        self._table.setRowCount(len(flagged))
        red    = QColor("#CC4444")
        orange = QColor("#CC8800")
        for row, item in enumerate(flagged):
            vals = [
                str(item["frame_idx"]),
                f"{item['priority']:.3f}",
                str(item["det_count"]),
                f"{item['avg_conf']:.3f}",
                item["issues"],
                item["source"],
            ]
            color = red if item["priority"] >= 0.9 else orange
            for col, val in enumerate(vals):
                cell = QTableWidgetItem(val)
                cell.setForeground(color)
                self._table.setItem(row, col, cell)

        btn_row   = QHBoxLayout()
        btn_go    = QPushButton("Go to Frame")
        btn_csv   = QPushButton("Export CSV")
        btn_close = QPushButton("Close")
        btn_go.clicked.connect(self._on_goto)
        btn_csv.clicked.connect(lambda: self._export_csv(flagged))
        btn_close.clicked.connect(self.accept)
        self._table.doubleClicked.connect(self._on_goto)
        btn_row.addWidget(btn_go)
        btn_row.addWidget(btn_csv)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _on_goto(self):
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if item:
            try:
                self.seek_requested.emit(int(item.text()))
            except ValueError:
                pass

    def _export_csv(self, flagged: list[dict]):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Review CSV", "review_queue.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            lines = ["frame_idx,priority,det_count,avg_conf,issues,source"]
            for item in flagged:
                lines.append(
                    f"{item['frame_idx']},{item['priority']},"
                    f"{item['det_count']},{item['avg_conf']},"
                    f"\"{item['issues']}\",{item['source']}"
                )
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "CSV Export Error", str(exc))


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dialogs                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class _HistogramCanvas(QWidget):
    """
    Internal widget that paints a confidence-score bar histogram using
    QPainter.  Used by ConfidenceHistogramDialog.
    """

    def __init__(self, confidences: list[float], parent=None):
        """
        Parameters
        ----------
        confidences : list[float]
            Raw confidence scores in [0, 1].
        """
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._confs = confidences
        self._bins  = 20

    def paintEvent(self, event):
        """Draw the histogram bar chart with QPainter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h  = self.width(), self.height()
        pad   = 32
        painter.fillRect(0, 0, w, h, QColor("#1E1E1E"))

        if not self._confs:
            painter.setPen(QColor("#888888"))
            painter.drawText(w // 2 - 50, h // 2, "No detections")
            painter.end()
            return

        # Compute bin counts
        bins = [0] * self._bins
        for c in self._confs:
            b = min(int(c * self._bins), self._bins - 1)
            bins[b] += 1

        max_count = max(bins) if any(bins) else 1
        chart_w   = w - 2 * pad
        chart_h   = h - 2 * pad
        bar_w     = chart_w / self._bins

        bar_color = QColor("#2A82DA")
        for i, count in enumerate(bins):
            bar_h = int(count / max_count * chart_h)
            x = pad + int(i * bar_w)
            y = pad + chart_h - bar_h
            painter.fillRect(x, y, max(1, int(bar_w) - 1), bar_h, bar_color)

        # Axes
        painter.setPen(QPen(QColor("#464649"), 1))
        painter.drawLine(pad, pad, pad, pad + chart_h)
        painter.drawLine(pad, pad + chart_h, pad + chart_w, pad + chart_h)

        # X-axis labels
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(QColor("#DCDCDC"))
        for v in (0, 5, 10):
            x = pad + int(v / 10.0 * chart_w)
            painter.drawText(x - 8, pad + chart_h + 14, f"{v / 10:.1f}")

        # Mean line
        mean_val = sum(self._confs) / len(self._confs)
        mean_x   = pad + int(mean_val * chart_w)
        painter.setPen(QPen(QColor("#FF8800"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(mean_x, pad, mean_x, pad + chart_h)

        painter.end()


class ConfidenceHistogramDialog(QDialog):
    """
    Modal dialog showing a bar histogram of detection confidence values
    for the current inference result.
    """

    def __init__(self, confidences: list[float], parent=None):
        """
        Parameters
        ----------
        confidences : list[float]
            Confidence scores to visualise.
        """
        super().__init__(parent)
        self.setWindowTitle("Confidence Distribution")
        self.setMinimumSize(420, 300)
        layout = QVBoxLayout(self)

        layout.addWidget(_HistogramCanvas(confidences))

        if confidences:
            mean_c = sum(confidences) / len(confidences)
            info   = (f"Count: {len(confidences)}   "
                      f"Mean: {mean_c:.3f}   "
                      f"Min: {min(confidences):.3f}   "
                      f"Max: {max(confidences):.3f}")
        else:
            info = "No detections in current result."
        lbl = QLabel(info)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# --------------------------------------------------------------------------- #

class ClassFilterDialog(QDialog):
    """
    Dialog that lets the user check/uncheck individual YOLO classes.
    Checked classes are included in display and export; unchecked are hidden.
    """

    def __init__(
        self,
        known_classes: dict[int, str],
        enabled: "set[int] | None",
        parent=None,
    ):
        """
        Parameters
        ----------
        known_classes : dict[int, str]
            Mapping of class_id -> class_name accumulated from inference runs.
        enabled : set[int] or None
            Currently enabled class IDs.  ``None`` means all enabled.
        """
        super().__init__(parent)
        self.setWindowTitle("Class Filter")
        self.setMinimumSize(280, 340)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Checked classes are shown and exported:"))

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for cid, cname in sorted(known_classes.items()):
            item = QListWidgetItem(f"[{cid}]  {cname}")
            item.setData(Qt.ItemDataRole.UserRole, cid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = (enabled is None) or (cid in enabled)
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            self._list.addItem(item)
        layout.addWidget(self._list)

        toggle_row = QHBoxLayout()
        btn_all  = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none.clicked.connect(lambda: self._set_all(False))
        toggle_row.addWidget(btn_all)
        toggle_row.addWidget(btn_none)
        layout.addLayout(toggle_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _set_all(self, state: bool):
        """Check or uncheck all items."""
        cs = Qt.CheckState.Checked if state else Qt.CheckState.Unchecked
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(cs)

    def get_enabled_classes(self) -> "set[int] | None":
        """
        Return the set of checked class IDs, or ``None`` when all are checked
        (meaning no filter is active).
        """
        enabled: set[int] = set()
        total = self._list.count()
        for i in range(total):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                enabled.add(item.data(Qt.ItemDataRole.UserRole))
        # None = no filter (all enabled)
        return None if len(enabled) == total else enabled


# --------------------------------------------------------------------------- #

class ReviewDialog(QDialog):
    """
    Dialog that lists all exported label files so the engineer can see which
    frames have annotations and which are empty, and jump to any frame.
    """

    seek_requested = pyqtSignal(int)

    def __init__(self, labels_dir: str, parent=None):
        """
        Parameters
        ----------
        labels_dir : str
            Path to the ``labels/`` sub-directory inside the output directory.
        """
        super().__init__(parent)
        self.setWindowTitle("Review Exported Labels")
        self.setMinimumSize(520, 440)
        layout = QVBoxLayout(self)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Frame #", "File", "Detections", "Status"]
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        btn_go  = QPushButton("Go to Frame")
        btn_go.clicked.connect(self._on_goto)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_go)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self._populate(labels_dir)
        self._table.doubleClicked.connect(self._on_goto)

    def _populate(self, labels_dir: str):
        """Scan *labels_dir* for .txt files and fill the table."""
        ldir = Path(labels_dir)
        rows: list[tuple[int, str, int]] = []

        if ldir.exists():
            for f in sorted(ldir.glob("frame_*.txt")):
                try:
                    idx   = int(f.stem.replace("frame_", ""))
                    lines = [
                        ln for ln in f.read_text(encoding="utf-8").splitlines()
                        if ln.strip()
                    ]
                    rows.append((idx, f.name, len(lines)))
                except (ValueError, OSError):
                    pass

        self._table.setRowCount(len(rows))
        red = QColor("#CC4444")
        for row_i, (idx, fname, count) in enumerate(rows):
            self._table.setItem(row_i, 0, QTableWidgetItem(str(idx)))
            self._table.setItem(row_i, 1, QTableWidgetItem(fname))
            self._table.setItem(row_i, 2, QTableWidgetItem(str(count)))
            status_item = QTableWidgetItem("OK" if count > 0 else "Empty")
            self._table.setItem(row_i, 3, status_item)
            if count == 0:
                for col in range(4):
                    it = self._table.item(row_i, col)
                    if it:
                        it.setForeground(red)

    def _on_goto(self):
        """Emit seek_requested for the selected row."""
        row = self._table.currentRow()
        if row < 0:
            return
        idx_item = self._table.item(row, 0)
        if idx_item:
            try:
                self.seek_requested.emit(int(idx_item.text()))
            except ValueError:
                pass


# --------------------------------------------------------------------------- #

class _ClassBarChart(QWidget):
    """Internal bar-chart widget used by StatisticsDialog."""

    def __init__(
        self, class_counts: dict[int, int],
        known_classes: dict[int, str], parent=None,
    ):
        """
        Parameters
        ----------
        class_counts : dict[int, int]
            Mapping of class_id -> detection count.
        known_classes : dict[int, str]
            Mapping of class_id -> class_name for labels.
        """
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._counts = class_counts
        self._names  = known_classes

    def paintEvent(self, event):
        """Draw bars proportional to per-class detection count."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h   = self.width(), self.height()
        pad    = 40
        painter.fillRect(0, 0, w, h, QColor("#1E1E1E"))

        if not self._counts:
            painter.setPen(QColor("#888888"))
            painter.drawText(w // 2 - 30, h // 2, "No data")
            painter.end()
            return

        items     = sorted(self._counts.items())
        n         = len(items)
        max_count = max(v for _, v in items) or 1
        chart_w   = w - 2 * pad
        chart_h   = h - 2 * pad
        bar_w     = chart_w / n

        palette = [
            QColor("#2A82DA"), QColor("#44CC44"), QColor("#CC4444"),
            QColor("#DDAA00"), QColor("#AA44CC"), QColor("#44AACC"),
        ]

        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)

        for i, (cid, count) in enumerate(items):
            bar_h = int(count / max_count * chart_h)
            x     = pad + int(i * bar_w)
            y     = pad + chart_h - bar_h
            color = palette[i % len(palette)]
            painter.fillRect(x + 2, y, max(1, int(bar_w) - 4), bar_h, color)

            name = self._names.get(cid, str(cid))[:10]
            painter.setPen(QColor("#DCDCDC"))
            painter.save()
            painter.translate(x + int(bar_w) // 2, pad + chart_h + 5)
            painter.rotate(40)
            painter.drawText(0, 0, name)
            painter.restore()

            if bar_h > 14:
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(x + 4, y + bar_h - 4, str(count))

        painter.end()


class StatisticsDialog(QDialog):
    """
    Modal dialog showing dataset-level statistics: total frames, total
    detections, empty frames, and a per-class bar chart.
    """

    def __init__(
        self, labels_dir: str, known_classes: dict[int, str], parent=None
    ):
        """
        Parameters
        ----------
        labels_dir : str
            Path to the ``labels/`` sub-directory.
        known_classes : dict[int, str]
            Accumulated class ID -> name mapping.
        """
        super().__init__(parent)
        self.setWindowTitle("Dataset Statistics")
        self.setMinimumSize(480, 400)
        layout = QVBoxLayout(self)

        stats = self._compute(labels_dir)

        summary = QLabel(
            f"Annotated frames:      {stats['total_frames']}\n"
            f"Total detections:      {stats['total_detections']}\n"
            f"Empty frames:          {stats['empty_frames']}\n"
            f"Avg detections/frame:  {stats['avg_per_frame']:.2f}"
        )
        summary.setStyleSheet("font-family: monospace; padding: 8px;")
        layout.addWidget(summary)

        if stats["class_counts"]:
            layout.addWidget(
                _ClassBarChart(stats["class_counts"], known_classes)
            )
        else:
            layout.addWidget(QLabel("No label files found in the output directory."))

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    @staticmethod
    def _compute(labels_dir: str) -> dict:
        """
        Read all YOLO .txt label files and compute aggregate statistics.

        Parameters
        ----------
        labels_dir : str
            Path to the labels directory.

        Returns
        -------
        dict
            Keys: ``total_frames``, ``empty_frames``, ``total_detections``,
            ``avg_per_frame``, ``class_counts``.
        """
        ldir         = Path(labels_dir)
        class_counts: dict[int, int] = {}
        total_frames = empty_frames = total_detections = 0

        if ldir.exists():
            for f in ldir.glob("frame_*.txt"):
                total_frames += 1
                try:
                    lines = [
                        ln for ln in f.read_text(encoding="utf-8").splitlines()
                        if ln.strip()
                    ]
                    if not lines:
                        empty_frames += 1
                    else:
                        total_detections += len(lines)
                        for line in lines:
                            parts = line.split()
                            if parts:
                                cid = int(parts[0])
                                class_counts[cid] = class_counts.get(cid, 0) + 1
                except Exception:
                    pass

        return {
            "total_frames":      total_frames,
            "empty_frames":      empty_frames,
            "total_detections":  total_detections,
            "avg_per_frame":     total_detections / max(total_frames, 1),
            "class_counts":      class_counts,
        }


# --------------------------------------------------------------------------- #

class AddDetectionDialog(QDialog):
    """
    Small dialog that appears after the user draws a bounding box or polygon,
    asking for a class name and confidence score.
    """

    def __init__(self, known_names: list[str], parent=None):
        """
        Parameters
        ----------
        known_names : list[str]
            Class names seen in previous inference runs (pre-populates combo).
        """
        super().__init__(parent)
        self.setWindowTitle("Add Detection")
        self.setMinimumWidth(240)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Class name:"))
        self._combo = QComboBox()
        self._combo.setEditable(True)
        for name in sorted(set(known_names)):
            self._combo.addItem(name)
        layout.addWidget(self._combo)

        layout.addWidget(QLabel("Confidence:"))
        self._spin = QDoubleSpinBox()
        self._spin.setRange(0.0, 1.0)
        self._spin.setValue(1.0)
        self._spin.setSingleStep(0.05)
        self._spin.setDecimals(2)
        layout.addWidget(self._spin)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def class_name(self) -> str:
        """Return the entered or selected class name."""
        return self._combo.currentText().strip()

    def confidence(self) -> float:
        """Return the entered confidence score."""
        return self._spin.value()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Interactive FrameViewer                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class FrameViewer(QLabel):
    """
    Interactive video frame canvas that supports three edit modes:

    SELECT
        Left-click emits ``image_clicked`` with the image-space coordinates.
        A gold rectangle is drawn over the selected detection.

    DRAW_BOX
        Click-and-drag draws a white dashed rectangle.  On mouse-release,
        ``bbox_drawn`` is emitted with the image-space coordinates.

    DRAW_POLYGON
        Each left-click places a vertex (green dot).  Double-click closes
        the polygon and emits ``polygon_completed``.  ESC cancels.

    Signals
    -------
    image_clicked : (int, int)       image-space (x, y) on SELECT click
    bbox_drawn    : (int, int, int, int)  (x1, y1, x2, y2) image coords
    polygon_completed : list         list of (x, y) image-coord tuples
    """

    image_clicked     = pyqtSignal(int, int)
    bbox_drawn        = pyqtSignal(int, int, int, int)
    polygon_completed = pyqtSignal(list)

    def __init__(self, parent=None):
        """Initialise with SELECT mode, dark background, mouse tracking on."""
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1E1E1E;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._mode         = EditMode.SELECT
        self._selected_box: tuple | None = None   # (x1,y1,x2,y2) image coords
        self._drag_start:   tuple | None = None   # widget coords
        self._drag_end:     tuple | None = None   # widget coords
        self._poly_pts:     list         = []     # widget coord list
        self._cursor_pos:   tuple | None = None   # widget coords (polygon preview)

        # Zoom / pan state
        self._zoom:             float        = 1.0
        self._pan_x:            float        = 0.0
        self._pan_y:            float        = 0.0
        self._mid_drag_start:   tuple | None = None   # widget coords at drag start
        self._mid_drag_pan_start: tuple      = (0.0, 0.0)

    # ── Public API ──────────────────────────────────────────────────────

    def set_frame_pixmap(self, pixmap: QPixmap):
        """Replace the displayed frame and trigger a repaint."""
        self._pixmap = pixmap
        self.update()

    def set_edit_mode(self, mode: EditMode):
        """
        Switch to a new edit mode, clearing any in-progress drawing.

        Parameters
        ----------
        mode : EditMode
            Target mode.
        """
        self._mode       = mode
        self._drag_start = None
        self._drag_end   = None
        self._poly_pts   = []
        self._cursor_pos = None
        cursor = (
            Qt.CursorShape.CrossCursor
            if mode in (EditMode.DRAW_BOX, EditMode.DRAW_POLYGON)
            else Qt.CursorShape.ArrowCursor
        )
        self.setCursor(cursor)
        self.update()

    def set_selection(self, box_img: "tuple | None"):
        """
        Set the gold selection highlight.

        Parameters
        ----------
        box_img : tuple or None
            ``(x1, y1, x2, y2)`` in image pixel coordinates, or ``None``
            to clear the highlight.
        """
        self._selected_box = box_img
        self.update()

    def cancel_drawing(self):
        """Cancel any in-progress draw-box or draw-polygon operation."""
        self._drag_start = None
        self._drag_end   = None
        self._poly_pts   = []
        self._cursor_pos = None
        self.update()

    def reset_zoom(self):
        """Reset zoom to 1× and clear pan offset."""
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ── Coordinate helpers ───────────────────────────────────────────────

    def _scale_info(self) -> "tuple | None":
        """
        Return ``(ox, oy, sx, sy)`` mapping image coords to widget coords,
        incorporating the current zoom level and pan offset.

        Returns
        -------
        tuple or None
            ``None`` when no pixmap is loaded.
        """
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return None
        ww, wh = self.width(), self.height()
        # Base scale: fit image into widget preserving aspect ratio
        base_scale = min(ww / pw, wh / ph)
        # Apply zoom
        sx = sy = base_scale * self._zoom
        # Compute origin so image is centred + pan offset applied
        ox = (ww - pw * sx) / 2.0 + self._pan_x
        oy = (wh - ph * sy) / 2.0 + self._pan_y
        return (ox, oy, sx, sy)

    def _w2i(self, wx: float, wy: float) -> "tuple[int,int] | tuple[None,None]":
        """Convert widget coordinates to image coordinates."""
        info = self._scale_info()
        if info is None:
            return None, None
        ox, oy, sx, sy = info
        ix = int((wx - ox) / sx)
        iy = int((wy - oy) / sy)
        if self._pixmap:
            ix = max(0, min(ix, self._pixmap.width()  - 1))
            iy = max(0, min(iy, self._pixmap.height() - 1))
        return ix, iy

    def _i2w(self, ix: float, iy: float) -> "tuple[int,int] | tuple[None,None]":
        """Convert image coordinates to widget coordinates."""
        info = self._scale_info()
        if info is None:
            return None, None
        ox, oy, sx, sy = info
        return int(ox + ix * sx), int(oy + iy * sy)

    # ── Mouse events ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Handle left-click for SELECT/DRAW_BOX/DRAW_POLYGON, right-click to undo polygon vertex, middle-click to start pan."""
        if event.button() == Qt.MouseButton.LeftButton:
            wx, wy = event.position().x(), event.position().y()
            if self._mode == EditMode.SELECT:
                ix, iy = self._w2i(wx, wy)
                if ix is not None:
                    self.image_clicked.emit(ix, iy)
            elif self._mode == EditMode.DRAW_BOX:
                self._drag_start = (wx, wy)
                self._drag_end   = (wx, wy)
            elif self._mode == EditMode.DRAW_POLYGON:
                self._poly_pts.append((wx, wy))
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            # Undo last polygon vertex
            if self._mode == EditMode.DRAW_POLYGON and self._poly_pts:
                self._poly_pts.pop()
                self.update()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._mid_drag_start    = (event.position().x(), event.position().y())
            self._mid_drag_pan_start = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Update drag endpoint, polygon cursor preview, or pan offset."""
        wx, wy = event.position().x(), event.position().y()
        self._cursor_pos = (wx, wy)
        if self._mode == EditMode.DRAW_BOX and self._drag_start:
            self._drag_end = (wx, wy)
        if self._mid_drag_start is not None:
            dx = wx - self._mid_drag_start[0]
            dy = wy - self._mid_drag_start[1]
            self._pan_x = self._mid_drag_pan_start[0] + dx
            self._pan_y = self._mid_drag_pan_start[1] + dy
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finalise a drawn bounding box and emit bbox_drawn, or end pan."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == EditMode.DRAW_BOX and self._drag_start:
                wx, wy = event.position().x(), event.position().y()
                sx, sy = self._drag_start
                ex, ey = wx, wy
                ix1, iy1 = self._w2i(min(sx, ex), min(sy, ey))
                ix2, iy2 = self._w2i(max(sx, ex), max(sy, ey))
                self._drag_start = None
                self._drag_end   = None
                self.update()
                if (ix1 is not None and ix2 is not None
                        and (ix2 - ix1) > 5 and (iy2 - iy1) > 5):
                    self.bbox_drawn.emit(ix1, iy1, ix2, iy2)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._mid_drag_start = None
            # Restore cursor
            cursor = (
                Qt.CursorShape.CrossCursor
                if self._mode in (EditMode.DRAW_BOX, EditMode.DRAW_POLYGON)
                else Qt.CursorShape.ArrowCursor
            )
            self.setCursor(cursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Zoom in/out centred on the cursor position."""
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        cx = event.position().x()
        cy = event.position().y()

        info = self._scale_info()
        if info is None:
            super().wheelEvent(event)
            return
        ox, oy, sx, _ = info

        # Image coord currently under cursor
        if sx == 0:
            super().wheelEvent(event)
            return
        ix = (cx - ox) / sx
        iy = (cy - oy) / sx  # sy == sx (square pixels)

        self._zoom = max(0.1, min(self._zoom * factor, 30.0))

        # Recompute pan so that (ix, iy) stays under the cursor
        pw = self._pixmap.width() if self._pixmap else 1
        ph = self._pixmap.height() if self._pixmap else 1
        base_scale = min(self.width() / pw, self.height() / ph) if pw and ph else 1.0
        new_sx = base_scale * self._zoom
        self._pan_x = cx - ix * new_sx - (self.width()  - pw * new_sx) / 2.0
        self._pan_y = cy - iy * new_sx - (self.height() - ph * new_sx) / 2.0
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        """Close an in-progress polygon on double-click (>= 3 vertices)."""
        if (event.button() == Qt.MouseButton.LeftButton
                and self._mode == EditMode.DRAW_POLYGON
                and len(self._poly_pts) >= 3):
            image_pts = []
            for wx, wy in self._poly_pts:
                ix, iy = self._w2i(wx, wy)
                if ix is not None:
                    image_pts.append((ix, iy))
            self._poly_pts   = []
            self._cursor_pos = None
            self.update()
            if len(image_pts) >= 3:
                self.polygon_completed.emit(image_pts)
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """ESC cancels any in-progress drawing."""
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_drawing()
        super().keyPressEvent(event)

    # ── Paint ────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        """
        Draw the frame pixmap, then overlay: selection highlight,
        drag rectangle, and polygon vertices/edges.
        """
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Base frame — draw at zoom/pan-aware position and size
        info = self._scale_info()
        if info is None:
            painter.end()
            return
        ox, oy, sx, sy = info
        pw, ph = self._pixmap.width(), self._pixmap.height()
        painter.drawPixmap(
            int(ox), int(oy), int(pw * sx), int(ph * sy), self._pixmap
        )

        # Selection highlight (gold)
        if self._selected_box is not None:
            info = self._scale_info()
            if info:
                o_x, o_y, sx, sy = info
                x1, y1, x2, y2 = self._selected_box
                wx1 = int(o_x + x1 * sx)
                wy1 = int(o_y + y1 * sy)
                wx2 = int(o_x + x2 * sx)
                wy2 = int(o_y + y2 * sy)
                painter.setPen(QPen(QColor("#FFD700"), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(wx1, wy1, wx2 - wx1, wy2 - wy1)

        # Draw-box rubber-band
        if self._drag_start and self._drag_end:
            sx, sy = self._drag_start
            ex, ey = self._drag_end
            painter.setPen(QPen(QColor("#FFFFFF"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                int(min(sx, ex)), int(min(sy, ey)),
                int(abs(ex - sx)), int(abs(ey - sy)),
            )

        # Draw-polygon vertices + edges
        if self._poly_pts:
            green = QColor("#00FF88")
            painter.setPen(QPen(green, 2))
            painter.setBrush(QBrush(green))
            for i, (px, py) in enumerate(self._poly_pts):
                painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)
                if i > 0:
                    ppx, ppy = self._poly_pts[i - 1]
                    painter.drawLine(int(ppx), int(ppy), int(px), int(py))
            # Preview edge to current cursor
            if self._cursor_pos:
                cx, cy   = self._cursor_pos
                lx, ly   = self._poly_pts[-1]
                painter.setPen(
                    QPen(green, 1, Qt.PenStyle.DashLine)
                )
                painter.drawLine(int(lx), int(ly), int(cx), int(cy))

        painter.end()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def _section_label(title: str) -> QLabel:
    """
    Return a bold QLabel section header with a 1 px bottom border.

    Parameters
    ----------
    title : str
        Section title text.

    Returns
    -------
    QLabel
        Styled header widget.
    """
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "font-weight: bold;"
        "border-bottom: 1px solid #464649;"
        "padding-bottom: 3px;"
        "margin-top: 8px;"
        "margin-bottom: 2px;"
    )
    return lbl


# ═══════════════════════════════════════════════════════════════════════════ #
#  MainWindow                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class MainWindow(QMainWindow):
    """
    Primary application window for FrameForge.

    Orchestrates VideoHandler, InferenceEngine, and ExportHandler through
    PyQt6 signals/slots and background QThread workers.  All long-running
    operations run in worker threads so the GUI never freezes.
    """

    def __init__(self):
        """Initialise all components, build the UI, wire signals, start timers."""
        super().__init__()
        self.setWindowTitle("FrameForge — Auto-Label & Pseudo-Label Studio")
        self.resize(1280, 720)

        # Core components
        self._video           = VideoHandler()
        self._video_file_path = ""
        self._engine = InferenceEngine()
        self._export = ExportHandler()

        # Application state
        self._current_frame_idx:  int                     = 0
        self._current_frame_bgr:  np.ndarray | None       = None
        self._current_result:     InferenceResult | None  = None
        self._output_dir:         str                     = ""
        self._is_playing:         bool                    = False
        self._last_infer_time:    float                   = 0.0
        self._last_frame_time:    float                   = 0.0
        self._fps_window:         deque                   = deque(maxlen=10)

        # Correction state
        self._selected_detection: int                     = -1
        self._edit_mode:          EditMode                = EditMode.SELECT
        self._enabled_classes:    "set[int] | None"       = None
        self._known_classes:      dict[int, str]          = {}

        # Per-frame annotation store (frame_idx -> InferenceResult)
        self._frame_store:  dict[int, InferenceResult]   = {}
        # Undo/redo stacks per frame (frame_idx -> list of InferenceResult snapshots)
        self._undo_stacks:  dict[int, list]              = {}
        self._redo_stacks:  dict[int, list]              = {}

        # Grounded SAM state
        self._gsam_engine:        GroundedSAMEngine           = GroundedSAMEngine()
        self._gsam_model_worker:  GSAMModelLoader   | None    = None
        self._gsam_infer_worker:  GSAMInferenceWorker | None  = None

        # Semantic segmentation state
        self._sem_engine:         SemanticSegmentationEngine  = SemanticSegmentationEngine()
        self._sem_model_worker:   SemanticModelLoader | None  = None
        self._sem_infer_worker:   SemanticInferenceWorker | None = None
        self._sem_batch_worker:   SemanticBatchWorker | None  = None
        self._current_semantic:   SemanticResult | None       = None
        self._semantic_store:     dict                        = {}   # {frame_idx: SemanticResult}
        self._view_semantic:      bool                        = False

        # Active learning state
        self._al_worker:          ActiveLearningWorker | None = None
        self._flagged_frames:     list[dict]                  = []

        # SAM Box Refiner state
        self._sam_refiner:             SAMBoxRefiner               = SAMBoxRefiner()
        self._sam_refine_load_worker:  SAMRefineLoadWorker | None  = None
        self._sam_refine_worker:       SAMRefineWorker | None      = None

        # Training state
        self._train_worker:  TrainingWorker | None = None

        # SAM 2 temporal propagation state
        self._sam2_engine:       SAM2PropagationEngine       = SAM2PropagationEngine()
        self._propagation_worker: PropagationWorker | None   = None

        # Worker references
        self._model_worker:  ModelLoaderWorker  | None = None
        self._infer_worker:  InferenceWorker    | None = None
        self._export_worker: ExportAllWorker    | None = None

        self._build_ui()
        self._build_status_bar()
        self._wire_signals()
        self._wire_shortcuts()

        # Timers
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._advance_frame)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start()

    # ══════════════════════════════════════════════════════════════════ #
    #  UI construction                                                    #
    # ══════════════════════════════════════════════════════════════════ #

    def _build_ui(self):
        """Lay out the splitter, left scroll panel, and right viewer panel."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(200)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(2)

        self._build_file_section(left_layout)
        self._build_model_section(left_layout)
        self._build_inference_section(left_layout)
        self._build_corrections_section(left_layout)
        self._build_grounded_sam_section(left_layout)
        self._build_sam_refine_section(left_layout)
        self._build_semantic_section(left_layout)
        self._build_export_section(left_layout)
        self._build_active_learning_section(left_layout)
        self._build_training_section(left_layout)
        self._build_propagation_section(left_layout)
        self._build_log_section(left_layout)
        left_layout.addStretch(1)
        left_scroll.setWidget(left_widget)

        # Right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        # View mode toggle — Instance (default) vs Semantic
        view_bar = QHBoxLayout()
        view_bar.addWidget(QLabel("View:"))
        self._btn_view_instance = QPushButton("Instance")
        self._btn_view_semantic = QPushButton("Semantic")
        for btn in (self._btn_view_instance, self._btn_view_semantic):
            btn.setCheckable(True)
            btn.setFixedHeight(22)
        self._btn_view_instance.setChecked(True)
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        self._view_group.addButton(self._btn_view_instance, 0)
        self._view_group.addButton(self._btn_view_semantic, 1)
        view_bar.addWidget(self._btn_view_instance)
        view_bar.addWidget(self._btn_view_semantic)
        view_bar.addStretch()
        right_layout.addLayout(view_bar)

        self._frame_viewer = FrameViewer()
        right_layout.addWidget(self._frame_viewer, stretch=1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        right_layout.addWidget(self._slider)

        self._lbl_frame_pos = QLabel("Frame 0 / 0")
        self._lbl_frame_pos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self._lbl_frame_pos)

        right_layout.addLayout(self._build_transport_bar())

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 900])

    def _build_file_section(self, layout: QVBoxLayout):
        """Add File section widgets."""
        layout.addWidget(_section_label("File"))
        self._btn_open_video = QPushButton("Open Video")
        layout.addWidget(self._btn_open_video)
        self._lbl_filename = QLabel("No file loaded")
        self._lbl_filename.setWordWrap(True)
        self._lbl_filename.setStyleSheet("color:#888888;font-size:8pt;")
        layout.addWidget(self._lbl_filename)
        self._btn_set_output = QPushButton("Set Output Dir")
        layout.addWidget(self._btn_set_output)
        self._lbl_output_dir = QLabel("Not set")
        self._lbl_output_dir.setWordWrap(True)
        self._lbl_output_dir.setStyleSheet("color:#888888;font-size:8pt;")
        layout.addWidget(self._lbl_output_dir)
        session_row = QHBoxLayout()
        self._btn_save_session = QPushButton("Save Session")
        self._btn_load_session = QPushButton("Load Session")
        self._btn_save_session.setToolTip(
            "Save all in-memory annotations to a JSON session file"
        )
        self._btn_load_session.setToolTip(
            "Load annotations from a previously saved session file"
        )
        session_row.addWidget(self._btn_save_session)
        session_row.addWidget(self._btn_load_session)
        layout.addLayout(session_row)

    def _build_model_section(self, layout: QVBoxLayout):
        """Add Model section widgets."""
        layout.addWidget(_section_label("Model"))
        self._combo_model = QComboBox()
        self._combo_model.addItems(
            ["yolo11n-seg.pt", "FastSAM-s.pt", "yolo11s-seg.pt"]
        )
        layout.addWidget(self._combo_model)
        self._btn_load_model = QPushButton("Load Model")
        layout.addWidget(self._btn_load_model)
        self._lbl_model_status = QLabel("Not Loaded")
        self._lbl_model_status.setStyleSheet("color:#CC4444;")
        layout.addWidget(self._lbl_model_status)

    def _build_inference_section(self, layout: QVBoxLayout):
        """Add Inference section widgets including stride and class filter."""
        layout.addWidget(_section_label("Inference"))

        self._chk_auto_infer = QCheckBox("Auto-Infer on Seek")
        layout.addWidget(self._chk_auto_infer)

        self._btn_run_inference = QPushButton("Run Inference (current frame)")
        layout.addWidget(self._btn_run_inference)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Conf:"))
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.0, 1.0)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setValue(0.35)
        self._spin_conf.setDecimals(2)
        conf_row.addWidget(self._spin_conf)
        layout.addLayout(conf_row)

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IoU:"))
        self._spin_iou = QDoubleSpinBox()
        self._spin_iou.setRange(0.0, 1.0)
        self._spin_iou.setSingleStep(0.05)
        self._spin_iou.setValue(0.45)
        self._spin_iou.setDecimals(2)
        iou_row.addWidget(self._spin_iou)
        layout.addLayout(iou_row)

        stride_row = QHBoxLayout()
        stride_row.addWidget(QLabel("Stride:"))
        self._spin_stride = QSpinBox()
        self._spin_stride.setRange(1, 300)
        self._spin_stride.setValue(1)
        self._spin_stride.setToolTip(
            "Export every Nth frame (1 = every frame)"
        )
        stride_row.addWidget(self._spin_stride)
        layout.addLayout(stride_row)

        self._btn_class_filter = QPushButton("Class Filter")
        self._btn_class_filter.setToolTip(
            "Choose which YOLO classes to show and export"
        )
        layout.addWidget(self._btn_class_filter)

        self._lbl_filter_status = QLabel("Filter: all classes")
        self._lbl_filter_status.setStyleSheet("color:#888888;font-size:8pt;")
        layout.addWidget(self._lbl_filter_status)

        imgsz_row = QHBoxLayout()
        imgsz_row.addWidget(QLabel("ImgSz:"))
        self._spin_imgsz = QSpinBox()
        self._spin_imgsz.setRange(320, 1280)
        self._spin_imgsz.setSingleStep(32)
        self._spin_imgsz.setValue(640)
        self._spin_imgsz.setToolTip(
            "Inference image size (square). Higher = more detail, more VRAM."
        )
        imgsz_row.addWidget(self._spin_imgsz)
        layout.addLayout(imgsz_row)

        self._btn_show_histogram = QPushButton("Confidence Histogram")
        layout.addWidget(self._btn_show_histogram)

    def _build_corrections_section(self, layout: QVBoxLayout):
        """Add Corrections section: mode buttons, delete, reassign, overlay toggle."""
        layout.addWidget(_section_label("Corrections"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(3)
        self._btn_mode_select  = QPushButton("Select")
        self._btn_mode_box     = QPushButton("Box")
        self._btn_mode_polygon = QPushButton("Polygon")
        for btn in (self._btn_mode_select, self._btn_mode_box,
                    self._btn_mode_polygon):
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            mode_row.addWidget(btn)
        self._btn_mode_select.setChecked(True)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._btn_mode_select,  0)
        self._mode_group.addButton(self._btn_mode_box,     1)
        self._mode_group.addButton(self._btn_mode_polygon, 2)
        layout.addLayout(mode_row)

        self._btn_delete_det = QPushButton("Delete Selected")
        layout.addWidget(self._btn_delete_det)

        self._btn_reassign = QPushButton("Reassign Class")
        layout.addWidget(self._btn_reassign)

        undo_row = QHBoxLayout()
        self._btn_undo = QPushButton("Undo")
        self._btn_redo = QPushButton("Redo")
        self._btn_undo.setToolTip("Ctrl+Z — undo last annotation change")
        self._btn_redo.setToolTip("Ctrl+Y — redo last undone change")
        undo_row.addWidget(self._btn_undo)
        undo_row.addWidget(self._btn_redo)
        layout.addLayout(undo_row)

        self._chk_show_overlay = QCheckBox("Show Overlay")
        self._chk_show_overlay.setChecked(True)
        layout.addWidget(self._chk_show_overlay)

        self._lbl_sel_info = QLabel("Nothing selected")
        self._lbl_sel_info.setWordWrap(True)
        self._lbl_sel_info.setStyleSheet("color:#888888;font-size:8pt;")
        layout.addWidget(self._lbl_sel_info)

    def _build_grounded_sam_section(self, layout: QVBoxLayout):
        """Add Grounded SAM section: text prompt, thresholds, load/run."""
        layout.addWidget(_section_label("Grounded SAM"))

        avail = GroundedSAMEngine.dino_available()
        if not avail:
            lbl = QLabel("Install: pip install transformers timm pillow\n(+ segment-anything for masks)")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#CC4444;font-size:8pt;")
            layout.addWidget(lbl)

        layout.addWidget(QLabel("Text prompt:"))
        self._edit_gsam_prompt = QLineEdit()
        self._edit_gsam_prompt.setText(GROUNDED_SAM_PROMPT)
        self._edit_gsam_prompt.setToolTip(
            "Period-separated class names sent to Grounding DINO.\n"
            "Edit to focus on specific classes."
        )
        layout.addWidget(self._edit_gsam_prompt)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Box:"))
        self._spin_gsam_box_thr = QDoubleSpinBox()
        self._spin_gsam_box_thr.setRange(0.05, 0.95)
        self._spin_gsam_box_thr.setSingleStep(0.05)
        self._spin_gsam_box_thr.setValue(0.35)
        self._spin_gsam_box_thr.setDecimals(2)
        thr_row.addWidget(self._spin_gsam_box_thr)
        thr_row.addWidget(QLabel("Txt:"))
        self._spin_gsam_txt_thr = QDoubleSpinBox()
        self._spin_gsam_txt_thr.setRange(0.05, 0.95)
        self._spin_gsam_txt_thr.setSingleStep(0.05)
        self._spin_gsam_txt_thr.setValue(0.25)
        self._spin_gsam_txt_thr.setDecimals(2)
        thr_row.addWidget(self._spin_gsam_txt_thr)
        layout.addLayout(thr_row)

        sam_row = QHBoxLayout()
        sam_row.addWidget(QLabel("SAM ckpt:"))
        self._edit_sam_ckpt = QLineEdit()
        self._edit_sam_ckpt.setPlaceholderText("sam_vit_b_01ec64.pth (optional)")
        self._edit_sam_ckpt.setToolTip(
            "Path to a SAM checkpoint .pth file.\n"
            "Leave blank for boxes-only (no masks)."
        )
        sam_row.addWidget(self._edit_sam_ckpt)
        btn_browse_sam = QPushButton("…")
        btn_browse_sam.setFixedWidth(26)
        btn_browse_sam.clicked.connect(self._on_browse_sam_ckpt)
        sam_row.addWidget(btn_browse_sam)
        layout.addLayout(sam_row)

        self._combo_sam_variant = QComboBox()
        self._combo_sam_variant.addItems(["vit_b", "vit_l", "vit_h"])
        self._combo_sam_variant.setToolTip("SAM model variant (must match checkpoint)")
        layout.addWidget(self._combo_sam_variant)

        gsam_btn_row = QHBoxLayout()
        self._btn_load_gsam = QPushButton("Load G-SAM")
        self._btn_run_gsam  = QPushButton("Run G-SAM")
        self._btn_run_gsam.setEnabled(False)
        gsam_btn_row.addWidget(self._btn_load_gsam)
        gsam_btn_row.addWidget(self._btn_run_gsam)
        layout.addLayout(gsam_btn_row)

        self._lbl_gsam_status = QLabel("Not loaded")
        self._lbl_gsam_status.setStyleSheet("color:#CC4444;font-size:8pt;")
        self._lbl_gsam_status.setWordWrap(True)
        layout.addWidget(self._lbl_gsam_status)

    def _build_sam_refine_section(self, layout: QVBoxLayout):
        """Add SAM Box Refine section — auto-segment YOLO boxes with SAM."""
        layout.addWidget(_section_label("SAM Box Refine"))

        # Backend selector
        backend_row = QHBoxLayout()
        self._radio_mobile_sam = QRadioButton("MobileSAM")
        self._radio_sam        = QRadioButton("SAM")
        self._radio_mobile_sam.setChecked(True)
        backend_row.addWidget(self._radio_mobile_sam)
        backend_row.addWidget(self._radio_sam)
        layout.addLayout(backend_row)
        self._radio_mobile_sam.toggled.connect(self._on_sam_refine_backend_changed)

        # SAM checkpoint (hidden when MobileSAM is selected)
        self._sam_refine_ckpt_row = QHBoxLayout()
        self._edit_refine_ckpt = QLineEdit()
        self._edit_refine_ckpt.setPlaceholderText("sam_vit_b_01ec64.pth")
        self._sam_refine_ckpt_row.addWidget(self._edit_refine_ckpt)
        btn_browse_refine = QPushButton("…")
        btn_browse_refine.setFixedWidth(26)
        btn_browse_refine.clicked.connect(self._on_browse_refine_ckpt)
        self._sam_refine_ckpt_row.addWidget(btn_browse_refine)
        layout.addLayout(self._sam_refine_ckpt_row)

        # SAM variant (hidden for MobileSAM)
        self._combo_refine_variant = QComboBox()
        self._combo_refine_variant.addItems(["vit_b", "vit_l", "vit_h"])
        layout.addWidget(self._combo_refine_variant)

        # Initially hidden for MobileSAM
        self._edit_refine_ckpt.setVisible(False)
        btn_browse_refine.setVisible(False)
        self._combo_refine_variant.setVisible(False)
        self._refine_sam_browse_btn = btn_browse_refine  # keep ref for visibility toggle

        refine_btn_row = QHBoxLayout()
        self._btn_load_refiner    = QPushButton("Load SAM Refiner")
        self._btn_auto_segment    = QPushButton("Auto-Segment Boxes")
        self._btn_auto_segment.setEnabled(False)
        self._btn_auto_segment.setToolTip(
            "Feeds every YOLO detection box into SAM to generate\n"
            "tight pixel-accurate masks — replaces rectangular masks."
        )
        refine_btn_row.addWidget(self._btn_load_refiner)
        refine_btn_row.addWidget(self._btn_auto_segment)
        layout.addLayout(refine_btn_row)

        self._lbl_refiner_status = QLabel("Not loaded")
        self._lbl_refiner_status.setStyleSheet("color:#CC4444;font-size:8pt;")
        self._lbl_refiner_status.setWordWrap(True)
        layout.addWidget(self._lbl_refiner_status)

    def _build_training_section(self, layout: QVBoxLayout):
        """Add Training Launcher section — one-click YOLO fine-tuning."""
        layout.addWidget(_section_label("Train Model"))

        # Data directory (output_dir which contains images/ and labels/)
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Data:"))
        self._edit_train_data = QLineEdit()
        self._edit_train_data.setPlaceholderText("output_dir (images/ + labels/)")
        self._edit_train_data.setToolTip(
            "Root dataset folder containing images/ and labels/ sub-dirs.\n"
            "Leave blank to auto-use the configured output directory."
        )
        data_row.addWidget(self._edit_train_data)
        btn_browse_data = QPushButton("…")
        btn_browse_data.setFixedWidth(26)
        btn_browse_data.clicked.connect(self._on_browse_train_data)
        data_row.addWidget(btn_browse_data)
        layout.addLayout(data_row)

        # Base model
        layout.addWidget(QLabel("Base model:"))
        self._combo_train_model = QComboBox()
        self._combo_train_model.addItems([
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11m-seg.pt",
            "yolo11l-seg.pt",
        ])
        self._combo_train_model.setToolTip(
            "Starting weights for fine-tuning.\n"
            "n=nano (fastest), s=small, m=medium, l=large (best accuracy)."
        )
        layout.addWidget(self._combo_train_model)

        # Hyperparameters row 1
        hp_row1 = QHBoxLayout()
        hp_row1.addWidget(QLabel("Epochs:"))
        self._spin_train_epochs = QSpinBox()
        self._spin_train_epochs.setRange(1, 1000)
        self._spin_train_epochs.setValue(50)
        hp_row1.addWidget(self._spin_train_epochs)
        hp_row1.addWidget(QLabel("Batch:"))
        self._spin_train_batch = QSpinBox()
        self._spin_train_batch.setRange(1, 128)
        self._spin_train_batch.setValue(8)
        hp_row1.addWidget(self._spin_train_batch)
        layout.addLayout(hp_row1)

        # Hyperparameters row 2
        hp_row2 = QHBoxLayout()
        hp_row2.addWidget(QLabel("Imgsz:"))
        self._spin_train_imgsz = QSpinBox()
        self._spin_train_imgsz.setRange(320, 1280)
        self._spin_train_imgsz.setSingleStep(32)
        self._spin_train_imgsz.setValue(640)
        hp_row2.addWidget(self._spin_train_imgsz)
        layout.addLayout(hp_row2)

        self._btn_start_train = QPushButton("Start Training")
        self._btn_start_train.setToolTip(
            "Generates data.yaml and launches: yolo train ..."
        )
        layout.addWidget(self._btn_start_train)

        self._lbl_train_status = QLabel("")
        self._lbl_train_status.setStyleSheet("color:#AAAAAA;font-size:8pt;")
        self._lbl_train_status.setWordWrap(True)
        layout.addWidget(self._lbl_train_status)

        # Training log — compact, scrolls automatically
        layout.addWidget(QLabel("Training log:"))
        self._txt_train_log = QTextEdit()
        self._txt_train_log.setReadOnly(True)
        self._txt_train_log.setMaximumHeight(180)
        self._txt_train_log.setStyleSheet(
            "font-family:monospace;font-size:7pt;background:#1e1e1e;color:#cccccc;"
        )
        layout.addWidget(self._txt_train_log)

    def _build_semantic_section(self, layout: QVBoxLayout):
        """Add Semantic Segmentation section: load, run, model selector."""
        layout.addWidget(_section_label("Semantic Seg"))

        avail = SemanticSegmentationEngine.is_available()
        if not avail:
            lbl = QLabel("Install: pip install transformers pillow")
            lbl.setStyleSheet("color:#CC4444;font-size:8pt;")
            layout.addWidget(lbl)

        self._combo_sem_model = QComboBox()
        self._combo_sem_model.addItems([
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        ])
        self._combo_sem_model.setToolTip("HuggingFace model — auto-downloaded on first use")
        layout.addWidget(self._combo_sem_model)

        sem_btn_row = QHBoxLayout()
        self._btn_load_sem = QPushButton("Load Semantic")
        self._btn_run_sem  = QPushButton("Run Semantic")
        self._btn_run_sem.setEnabled(False)
        sem_btn_row.addWidget(self._btn_load_sem)
        sem_btn_row.addWidget(self._btn_run_sem)
        layout.addLayout(sem_btn_row)

        self._btn_run_sem_all = QPushButton("Run All Frames")
        self._btn_run_sem_all.setEnabled(False)
        self._btn_run_sem_all.setToolTip(
            "Run semantic segmentation on every frame in the video (uses current stride)"
        )
        layout.addWidget(self._btn_run_sem_all)

        self._btn_save_semantic = QPushButton("Save Semantic PNG")
        self._btn_save_semantic.setEnabled(False)
        self._btn_save_semantic.setToolTip(
            "Save label-ID map + colour PNG to output_dir/semantic/"
        )
        layout.addWidget(self._btn_save_semantic)

        self._lbl_sem_status = QLabel("Not loaded")
        self._lbl_sem_status.setStyleSheet("color:#CC4444;font-size:8pt;")
        self._lbl_sem_status.setWordWrap(True)
        layout.addWidget(self._lbl_sem_status)

        self._lbl_sem_batch_progress = QLabel("")
        self._lbl_sem_batch_progress.setStyleSheet("color:#AAAAAA;font-size:8pt;")
        self._lbl_sem_batch_progress.setWordWrap(True)
        layout.addWidget(self._lbl_sem_batch_progress)

    def _build_active_learning_section(self, layout: QVBoxLayout):
        """Add Active Learning section: scan + review flagged frames."""
        layout.addWidget(_section_label("Active Learning"))

        al_thr_row = QHBoxLayout()
        al_thr_row.addWidget(QLabel("Flag conf <"))
        self._spin_al_conf = QDoubleSpinBox()
        self._spin_al_conf.setRange(0.0, 1.0)
        self._spin_al_conf.setSingleStep(0.05)
        self._spin_al_conf.setValue(0.50)
        self._spin_al_conf.setDecimals(2)
        self._spin_al_conf.setToolTip(
            "Frames with average detection confidence below this value are flagged"
        )
        al_thr_row.addWidget(self._spin_al_conf)
        layout.addLayout(al_thr_row)

        self._btn_al_scan   = QPushButton("Scan & Flag Frames")
        self._btn_al_review = QPushButton("Review Flagged Frames")
        self._btn_al_review.setEnabled(False)
        layout.addWidget(self._btn_al_scan)
        layout.addWidget(self._btn_al_review)

        self._lbl_al_status = QLabel("No scan run yet")
        self._lbl_al_status.setStyleSheet("color:#888888;font-size:8pt;")
        self._lbl_al_status.setWordWrap(True)
        layout.addWidget(self._lbl_al_status)

    def _build_export_section(self, layout: QVBoxLayout):
        """Add Export section with YOLO, COCO, CVAT, LabelStudio, Review, Stats."""
        layout.addWidget(_section_label("Export"))
        self._btn_save_yolo  = QPushButton("Save Pseudo-Label (YOLO)")
        self._btn_save_coco  = QPushButton("Save Pseudo-Label (COCO)")
        self._btn_save_cvat  = QPushButton("Save Pseudo-Label (CVAT)")
        self._btn_save_ls    = QPushButton("Save Pseudo-Label (LabelStudio)")
        self._btn_export_all = QPushButton("Export All Frames")
        self._btn_review     = QPushButton("Review Exported Labels")
        self._btn_stats      = QPushButton("Dataset Statistics")
        for btn in (
            self._btn_save_yolo, self._btn_save_coco,
            self._btn_save_cvat, self._btn_save_ls,
            self._btn_export_all, self._btn_review, self._btn_stats,
        ):
            layout.addWidget(btn)

    def _build_propagation_section(self, layout: QVBoxLayout):
        """Add Temporal Propagation (SAM 2) section."""
        layout.addWidget(_section_label("Temporal Propagation (SAM 2)"))

        # Checkpoint row
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Ckpt:"))
        self._edit_sam2_ckpt = QLineEdit()
        self._edit_sam2_ckpt.setPlaceholderText("sam2_hiera_tiny.pt")
        self._edit_sam2_ckpt.setToolTip(
            "Path to SAM 2 checkpoint (.pt).  Download from:\n"
            "https://github.com/facebookresearch/segment-anything-2/releases"
        )
        ckpt_row.addWidget(self._edit_sam2_ckpt)
        self._btn_browse_sam2_ckpt = QPushButton("…")
        self._btn_browse_sam2_ckpt.setFixedWidth(26)
        ckpt_row.addWidget(self._btn_browse_sam2_ckpt)
        layout.addLayout(ckpt_row)

        # Variant + load row
        variant_row = QHBoxLayout()
        variant_row.addWidget(QLabel("Size:"))
        self._combo_sam2_variant = QComboBox()
        for v in ("tiny", "small", "base+", "large"):
            self._combo_sam2_variant.addItem(v)
        variant_row.addWidget(self._combo_sam2_variant)
        self._btn_load_sam2 = QPushButton("Load SAM 2")
        variant_row.addWidget(self._btn_load_sam2)
        layout.addLayout(variant_row)

        self._lbl_sam2_status = QLabel("Not loaded")
        self._lbl_sam2_status.setStyleSheet("color:#888888;font-size:8pt;")
        layout.addWidget(self._lbl_sam2_status)

        # Propagation range
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range:"))
        self._spin_prop_range = QSpinBox()
        self._spin_prop_range.setRange(1, 500)
        self._spin_prop_range.setValue(60)
        self._spin_prop_range.setToolTip(
            "Number of frames to propagate from the current frame."
        )
        range_row.addWidget(self._spin_prop_range)
        range_row.addWidget(QLabel("frames"))
        layout.addLayout(range_row)

        # Propagation direction buttons
        self._btn_prop_forward  = QPushButton("▶  Propagate Forward")
        self._btn_prop_backward = QPushButton("◀  Propagate Backward")
        self._btn_prop_both     = QPushButton("◀▶  Propagate Both")
        for btn in (self._btn_prop_forward, self._btn_prop_backward, self._btn_prop_both):
            btn.setEnabled(False)
            layout.addWidget(btn)

        self._btn_prop_cancel = QPushButton("Cancel Propagation")
        self._btn_prop_cancel.setEnabled(False)
        layout.addWidget(self._btn_prop_cancel)

        self._lbl_prop_progress = QLabel("—")
        self._lbl_prop_progress.setStyleSheet("color:#888888;font-size:8pt;")
        layout.addWidget(self._lbl_prop_progress)

        # Wire local signals
        self._btn_browse_sam2_ckpt.clicked.connect(self._on_browse_sam2_ckpt)
        self._btn_load_sam2.clicked.connect(self._on_load_sam2)
        self._btn_prop_forward.clicked.connect(
            lambda: self._on_propagate(direction="forward")
        )
        self._btn_prop_backward.clicked.connect(
            lambda: self._on_propagate(direction="backward")
        )
        self._btn_prop_both.clicked.connect(
            lambda: self._on_propagate(direction="both")
        )
        self._btn_prop_cancel.clicked.connect(self._on_propagate_cancel)

    def _build_log_section(self, layout: QVBoxLayout):
        """Add Log section."""
        layout.addWidget(_section_label("Log"))
        self._log_list = QListWidget()
        self._log_list.setMaximumHeight(200)
        self._log_list.setStyleSheet("font-size:8pt;")
        layout.addWidget(self._log_list)

    def _build_transport_bar(self) -> QHBoxLayout:
        """Build and return the playback transport button row."""
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self._btn_first  = QPushButton("|◀")
        self._btn_back10 = QPushButton("◀")
        self._btn_prev   = QPushButton("◁")
        self._btn_next   = QPushButton("▷")
        self._btn_fwd10  = QPushButton("▶")
        self._btn_last   = QPushButton("▶|")
        self._btn_play   = QPushButton("▶  Play")
        for btn in (
            self._btn_first, self._btn_back10, self._btn_prev,
            self._btn_next,  self._btn_fwd10,  self._btn_last,
            self._btn_play,
        ):
            btn.setFixedHeight(28)
            bar.addWidget(btn)
        return bar

    def _build_status_bar(self):
        """Create the four permanent status-bar labels."""
        sb: QStatusBar = self.statusBar()
        sb.setStyleSheet("QStatusBar{border-top:1px solid #464649;}")

        def _lbl(t: str) -> QLabel:
            l = QLabel(t)
            l.setStyleSheet("padding:0 8px;")
            return l

        def _sep() -> QFrame:
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setStyleSheet("color:#464649;")
            return f

        self._sb_device = _lbl("Device: CPU")
        self._sb_vram   = _lbl("VRAM: 0 MB")
        self._sb_fps    = _lbl("FPS: --")
        self._sb_frame  = _lbl("Frame: 0 / 0")
        for w in (self._sb_device, _sep(), self._sb_vram,
                  _sep(), self._sb_fps, _sep(), self._sb_frame):
            sb.addPermanentWidget(w)

    # ══════════════════════════════════════════════════════════════════ #
    #  Signal wiring                                                      #
    # ══════════════════════════════════════════════════════════════════ #

    def _wire_signals(self):
        """Connect all widget signals to slot methods."""
        # File
        self._btn_open_video.clicked.connect(self._on_open_video)
        self._btn_set_output.clicked.connect(self._on_set_output_dir)
        self._btn_save_session.clicked.connect(self._on_save_session)
        self._btn_load_session.clicked.connect(self._on_load_session)
        # Model
        self._btn_load_model.clicked.connect(self._on_load_model)
        # Inference
        self._btn_run_inference.clicked.connect(self._on_run_inference)
        self._btn_class_filter.clicked.connect(self._on_class_filter)
        self._btn_show_histogram.clicked.connect(self._on_show_histogram)
        # Corrections
        self._mode_group.idClicked.connect(self._on_mode_changed)
        self._btn_delete_det.clicked.connect(self._on_delete_detection)
        self._btn_reassign.clicked.connect(self._on_reassign_class)
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_redo.clicked.connect(self._on_redo)
        self._chk_show_overlay.toggled.connect(lambda _: self._refresh_display())
        # FrameViewer interactions
        self._frame_viewer.image_clicked.connect(self._on_viewer_clicked)
        self._frame_viewer.bbox_drawn.connect(self._on_bbox_drawn)
        self._frame_viewer.polygon_completed.connect(self._on_polygon_completed)
        # Grounded SAM
        self._btn_load_gsam.clicked.connect(self._on_load_gsam)
        self._btn_run_gsam.clicked.connect(self._on_run_gsam)
        # SAM Box Refiner
        self._btn_load_refiner.clicked.connect(self._on_load_sam_refiner)
        self._btn_auto_segment.clicked.connect(self._on_auto_segment)
        # Training
        self._btn_start_train.clicked.connect(self._on_train_start)
        # Semantic
        self._btn_load_sem.clicked.connect(self._on_load_semantic)
        self._btn_run_sem.clicked.connect(self._on_run_semantic)
        self._btn_run_sem_all.clicked.connect(self._on_run_semantic_all)
        self._btn_save_semantic.clicked.connect(self._on_save_semantic)
        # View mode
        self._view_group.idClicked.connect(self._on_view_mode_changed)
        # Active learning
        self._btn_al_scan.clicked.connect(self._on_al_scan)
        self._btn_al_review.clicked.connect(self._on_al_review)
        # Export
        self._btn_save_yolo.clicked.connect(self._on_save_yolo)
        self._btn_save_coco.clicked.connect(self._on_save_coco)
        self._btn_save_cvat.clicked.connect(self._on_save_cvat)
        self._btn_save_ls.clicked.connect(self._on_save_labelstudio)
        self._btn_export_all.clicked.connect(self._on_export_all)
        self._btn_review.clicked.connect(self._on_review_labels)
        self._btn_stats.clicked.connect(self._on_dataset_stats)
        # Slider + transport
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        self._btn_first.clicked.connect(lambda: self._seek_to(0))
        self._btn_back10.clicked.connect(
            lambda: self._seek_to(self._current_frame_idx - 10)
        )
        self._btn_prev.clicked.connect(
            lambda: self._seek_to(self._current_frame_idx - 1)
        )
        self._btn_next.clicked.connect(
            lambda: self._seek_to(self._current_frame_idx + 1)
        )
        self._btn_fwd10.clicked.connect(
            lambda: self._seek_to(self._current_frame_idx + 10)
        )
        self._btn_last.clicked.connect(
            lambda: self._seek_to(self._video.total_frames() - 1)
        )
        self._btn_play.clicked.connect(self._on_play_pause)

    def _wire_shortcuts(self):
        """Install keyboard shortcuts on the main window."""
        def _sc(key, slot):
            qs = QShortcut(QKeySequence(key), self)
            qs.activated.connect(slot)

        _sc("A",             lambda: self._kb_seek(-1))
        _sc("D",             lambda: self._kb_seek(+1))
        _sc("I",             self._kb_infer)
        _sc("S",             self._kb_save)
        _sc("Ctrl+Z",        self._on_undo)
        _sc("Ctrl+Y",        self._on_redo)
        _sc("Ctrl+Shift+Z",  self._on_redo)
        _sc(Qt.Key.Key_Space,  self._on_play_pause)
        _sc(Qt.Key.Key_Delete, self._on_delete_key)

    # ── Keyboard-shortcut guards ─────────────────────────────────────────

    def _text_widget_focused(self) -> bool:
        """Return True when a text-input widget currently has keyboard focus."""
        focused = self.focusWidget()
        return isinstance(focused, (QDoubleSpinBox, QSpinBox, QComboBox))

    def _kb_seek(self, delta: int):
        """Seek by *delta* frames only when no text widget has focus."""
        if not self._text_widget_focused():
            self._seek_to(self._current_frame_idx + delta)

    def _on_delete_key(self):
        """Delete selected detection, or prompt to discard all annotations."""
        if self._text_widget_focused():
            return
        if self._selected_detection >= 0:
            self._on_delete_detection()
        else:
            self._on_discard_frame()

    def _kb_infer(self):
        """Run inference shortcut guard."""
        if not self._text_widget_focused():
            self._on_run_inference()

    def _kb_save(self):
        """Save YOLO shortcut guard."""
        if not self._text_widget_focused():
            self._on_save_yolo()

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — file                                                       #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_open_video(self):
        """Open a file dialog, load the video, and display frame 0."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI);;All Files (*)",
        )
        if not path:
            return
        if self._is_playing:
            self._on_play_pause()
        # Fix 3: cancel all running background workers before switching video
        for worker_attr in (
            "_infer_worker", "_gsam_infer_worker", "_export_worker",
            "_sem_batch_worker", "_sam_refine_worker", "_train_worker",
            "_propagation_worker",
        ):
            w = getattr(self, worker_attr, None)
            if w is not None and w.isRunning():
                if hasattr(w, "cancel"):
                    w.cancel()
                else:
                    w.quit()
        ok = self._video.open(path)
        if not ok:
            QMessageBox.critical(self, "Error",
                                 f"Could not open video file:\n{path}")
            self._log(f"ERROR: Could not open video — {path}")
            return
        self._video_file_path = path   # stored for thread-safe batch workers
        total = self._video.total_frames()
        self._slider.setMaximum(max(0, total - 1))
        self._slider.setValue(0)
        self._current_frame_idx  = 0
        self._current_result     = None
        self._selected_detection = -1
        self._frame_store.clear()
        self._undo_stacks.clear()
        self._redo_stacks.clear()
        basename = os.path.basename(path)
        self._lbl_filename.setText(basename)
        w, h = self._video.resolution()
        self._log(
            f"Opened: {basename}  "
            f"[{total} frames @ {self._video.fps():.1f} fps  {w}x{h}]"
        )
        self._display_frame(0)

    def _on_set_output_dir(self):
        """Open a directory picker and configure the ExportHandler."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if not path:
            return
        try:
            self._export.set_output_dir(path)
            self._output_dir = path
            self._lbl_output_dir.setText(os.path.basename(path) or path)
            self._log(f"Output dir set: {path}")
        except PermissionError:
            QMessageBox.critical(
                self, "Permission Error",
                f"Cannot write to:\n{path}\n\nPlease choose a different directory.",
            )
            self._log(f"ERROR: Output directory not writable — {path}")

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — model                                                      #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_load_model(self):
        """Spawn ModelLoaderWorker to load the selected model."""
        model_name = self._combo_model.currentText()
        self._btn_load_model.setEnabled(False)
        self._lbl_model_status.setText("Loading…")
        self._lbl_model_status.setStyleSheet("color:#DDAA00;")
        self._log(f"Loading model: {model_name} …")
        self._model_worker = ModelLoaderWorker(self._engine, model_name)
        self._model_worker.finished.connect(self._on_model_loaded)
        self._model_worker.start()

    def _on_model_loaded(self, success: bool, message: str):
        """Handle ModelLoaderWorker.finished."""
        self._btn_load_model.setEnabled(True)
        if success:
            self._lbl_model_status.setText(message)
            self._lbl_model_status.setStyleSheet("color:#44CC44;")
            self._log(f"Model ready — {message}")
        else:
            self._lbl_model_status.setText("Load Failed")
            self._lbl_model_status.setStyleSheet("color:#CC4444;")
            self._log(f"ERROR: {message}")
        self._update_status_bar()

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — inference                                                  #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_run_inference(self):
        """Spawn InferenceWorker for the current frame."""
        if not self._engine.is_loaded():
            self._log("Please load a model first.")
            return
        if self._current_frame_bgr is None:
            self._log("No frame available — open a video first.")
            return
        if self._infer_worker and self._infer_worker.isRunning():
            return
        self._btn_run_inference.setEnabled(False)
        self._infer_worker = InferenceWorker(
            self._engine, self._current_frame_bgr.copy(),
            self._spin_conf.value(), self._spin_iou.value(),
            imgsz=self._spin_imgsz.value(),
        )
        self._infer_worker.finished.connect(self._on_inference_done)
        self._infer_worker.error.connect(self._on_inference_error)
        self._infer_worker.start()

    def _on_inference_done(self, result: InferenceResult):
        """
        Handle InferenceWorker.finished: store result, update known classes,
        clear selection, refresh the display, and persist to frame store.
        """
        self._current_result     = result
        self._selected_detection = -1
        self._frame_viewer.set_selection(None)
        self._lbl_sel_info.setText("Nothing selected")
        self._btn_run_inference.setEnabled(True)
        self._last_infer_time = time.monotonic()
        for cid, cname in zip(result.class_ids, result.class_names):
            self._known_classes[cid] = cname
        # Persist fresh inference result to frame store
        self._frame_store[self._current_frame_idx] = self._deep_copy_result(result)
        # Clear undo/redo for this frame (new inference replaces corrections)
        self._undo_stacks.pop(self._current_frame_idx, None)
        self._redo_stacks.pop(self._current_frame_idx, None)
        self._update_propagation_buttons()
        self._refresh_display()
        self._log(
            f"Inference: {len(result.class_ids)} detection(s) "
            f"on frame {self._current_frame_idx}"
        )

    def _on_inference_error(self, msg: str):
        """Handle InferenceWorker.error."""
        self._btn_run_inference.setEnabled(True)
        self._log(f"ERROR during inference: {msg}")

    def _maybe_auto_infer(self):
        """Trigger auto-inference if enabled and >= 200 ms since last run."""
        if not self._chk_auto_infer.isChecked():
            return
        if not self._engine.is_loaded():
            return
        if self._infer_worker and self._infer_worker.isRunning():
            return
        if time.monotonic() - self._last_infer_time < 0.2:
            return
        self._on_run_inference()

    def _on_class_filter(self):
        """Open ClassFilterDialog and apply the result."""
        if not self._known_classes:
            self._log("Run inference at least once to populate the class list.")
            return
        dlg = ClassFilterDialog(self._known_classes, self._enabled_classes, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._enabled_classes = dlg.get_enabled_classes()
            if self._enabled_classes is None:
                self._lbl_filter_status.setText("Filter: all classes")
            else:
                names = [
                    self._known_classes.get(c, str(c))
                    for c in sorted(self._enabled_classes)
                ]
                self._lbl_filter_status.setText(
                    f"Filter: {', '.join(names)}"
                )
            self._refresh_display()

    def _on_show_histogram(self):
        """Show the confidence histogram for the current result."""
        if self._current_result is None:
            self._log("Run inference first.")
            return
        confs = list(self._current_result.confidences)
        ConfidenceHistogramDialog(confs, self).exec()

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — correction tools                                           #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_mode_changed(self, mode_id: int):
        """Switch FrameViewer edit mode based on button group ID."""
        mapping = {
            0: EditMode.SELECT,
            1: EditMode.DRAW_BOX,
            2: EditMode.DRAW_POLYGON,
        }
        self._edit_mode = mapping.get(mode_id, EditMode.SELECT)
        self._frame_viewer.set_edit_mode(self._edit_mode)
        if self._edit_mode != EditMode.SELECT:
            self._selected_detection = -1
            self._frame_viewer.set_selection(None)
            self._lbl_sel_info.setText("Nothing selected")

    def _on_viewer_clicked(self, img_x: int, img_y: int):
        """
        Handle a click in SELECT mode: find which detection (if any) was
        clicked and highlight it.
        """
        if self._current_result is None or self._edit_mode != EditMode.SELECT:
            return
        disp = self._get_display_indices()
        for actual_i in disp:
            if actual_i >= len(self._current_result.boxes_xyxy):
                continue
            x1, y1, x2, y2 = self._current_result.boxes_xyxy[actual_i]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self._selected_detection = actual_i
                self._frame_viewer.set_selection(
                    tuple(self._current_result.boxes_xyxy[actual_i])
                )
                cname = (self._current_result.class_names[actual_i]
                         if actual_i < len(self._current_result.class_names)
                         else "?")
                conf  = (self._current_result.confidences[actual_i]
                         if actual_i < len(self._current_result.confidences)
                         else 0.0)
                self._lbl_sel_info.setText(
                    f"#{actual_i}: {cname} ({conf:.2f})"
                )
                return
        # Clicked empty area — deselect
        self._selected_detection = -1
        self._frame_viewer.set_selection(None)
        self._lbl_sel_info.setText("Nothing selected")

    def _on_delete_detection(self):
        """Remove the selected detection from the current result."""
        if self._current_result is None or self._selected_detection < 0:
            self._log("Select a detection first.")
            return
        idx = self._selected_detection
        if idx >= len(self._current_result.class_ids):
            return
        self._push_undo()
        cname = (self._current_result.class_names[idx]
                 if idx < len(self._current_result.class_names) else "?")
        self._current_result.boxes_xyxy.pop(idx)
        self._current_result.class_ids.pop(idx)
        self._current_result.confidences.pop(idx)
        if idx < len(self._current_result.masks_binary):
            self._current_result.masks_binary.pop(idx)
        if idx < len(self._current_result.class_names):
            self._current_result.class_names.pop(idx)
        self._selected_detection = -1
        self._frame_viewer.set_selection(None)
        self._lbl_sel_info.setText("Nothing selected")
        self._refresh_display()
        self._log(f"Deleted detection #{idx} ({cname})")

    def _on_discard_frame(self):
        """Clear all annotations for the current frame after confirmation."""
        if self._current_result is None:
            return
        reply = QMessageBox.question(
            self, "Discard Annotations",
            f"Remove all annotations from frame {self._current_frame_idx}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._current_result     = None
            self._selected_detection = -1
            self._frame_viewer.set_selection(None)
            self._lbl_sel_info.setText("Nothing selected")
            self._refresh_display()
            self._log(f"Discarded annotations on frame {self._current_frame_idx}")

    def _on_reassign_class(self):
        """Open a dialog to change the class of the selected detection."""
        if self._current_result is None or self._selected_detection < 0:
            self._log("Select a detection first.")
            return
        idx  = self._selected_detection
        dlg  = AddDetectionDialog(list(self._known_classes.values()), self)
        dlg.setWindowTitle("Reassign Class")
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._push_undo()
            new_name = dlg.class_name()
            new_conf = dlg.confidence()
            if not new_name:
                return
            # Find or create a class ID for this name
            new_id = next(
                (k for k, v in self._known_classes.items() if v == new_name),
                max(self._known_classes.keys(), default=-1) + 1,
            )
            self._known_classes[new_id] = new_name
            if idx < len(self._current_result.class_ids):
                self._current_result.class_ids[idx]    = new_id
            if idx < len(self._current_result.class_names):
                self._current_result.class_names[idx]  = new_name
            if idx < len(self._current_result.confidences):
                self._current_result.confidences[idx]  = new_conf
            self._lbl_sel_info.setText(f"#{idx}: {new_name} ({new_conf:.2f})")
            self._refresh_display()
            self._log(f"Reassigned #{idx} to {new_name}")

    def _on_bbox_drawn(self, x1: int, y1: int, x2: int, y2: int):
        """Add a manually drawn bounding box as a new detection."""
        if self._current_frame_bgr is None:
            return
        if self._current_result is None:
            h, w = self._current_frame_bgr.shape[:2]
            self._current_result = InferenceResult([], [], [], [], [], (h, w))
        dlg = AddDetectionDialog(list(self._known_classes.values()), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cname = dlg.class_name()
            conf  = dlg.confidence()
            if not cname:
                return
            self._push_undo()
            cid = next(
                (k for k, v in self._known_classes.items() if v == cname),
                max(self._known_classes.keys(), default=-1) + 1,
            )
            self._known_classes[cid] = cname
            # Build a filled rectangular mask for the drawn box
            h, w = self._current_frame_bgr.shape[:2]
            rect_mask = np.zeros((h, w), dtype=np.uint8)
            rect_mask[
                max(0, y1):min(h, y2 + 1),
                max(0, x1):min(w, x2 + 1),
            ] = 255
            self._current_result.boxes_xyxy.append(
                [float(x1), float(y1), float(x2), float(y2)]
            )
            self._current_result.class_ids.append(cid)
            self._current_result.confidences.append(conf)
            self._current_result.masks_binary.append(rect_mask)
            self._current_result.class_names.append(cname)
            self._refresh_display()
            self._log(f"Added box: {cname} [{x1},{y1},{x2},{y2}]")

    def _on_polygon_completed(self, pts: list):
        """Add a manually drawn polygon as a new detection with a binary mask."""
        if self._current_frame_bgr is None or len(pts) < 3:
            return
        if self._current_result is None:
            h, w = self._current_frame_bgr.shape[:2]
            self._current_result = InferenceResult([], [], [], [], [], (h, w))
        dlg = AddDetectionDialog(list(self._known_classes.values()), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cname = dlg.class_name()
            conf  = dlg.confidence()
            if not cname:
                return
            self._push_undo()
            cid = next(
                (k for k, v in self._known_classes.items() if v == cname),
                max(self._known_classes.keys(), default=-1) + 1,
            )
            self._known_classes[cid] = cname
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            h, w = self._current_frame_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], 255)

            self._current_result.boxes_xyxy.append(
                [float(x1), float(y1), float(x2), float(y2)]
            )
            self._current_result.class_ids.append(cid)
            self._current_result.confidences.append(conf)
            self._current_result.masks_binary.append(mask)
            self._current_result.class_names.append(cname)
            self._refresh_display()
            self._log(f"Added polygon: {cname} ({len(pts)} pts)")

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — export                                                     #
    # ══════════════════════════════════════════════════════════════════ #

    def _export_preflight(self) -> bool:
        """Return True when output dir, frame, and result are all ready."""
        if not self._output_dir:
            self._log("Set an output directory first.")
            return False
        if self._current_frame_bgr is None:
            self._log("No frame loaded — open a video first.")
            return False
        if self._current_result is None:
            self._log("Run inference on this frame before saving.")
            return False
        return True

    def _filtered_result(self) -> InferenceResult:
        """Return the current result with the active class filter applied."""
        return ExportHandler.filter_result(
            self._current_result, self._enabled_classes
        )

    def _on_save_yolo(self):
        """Save current frame + result as a YOLO annotation."""
        if not self._export_preflight():
            return
        try:
            path = self._export.save_yolo(
                self._current_frame_bgr,
                self._filtered_result(),
                self._current_frame_idx,
            )
            self._log(f"Saved YOLO: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR saving YOLO: {exc}")

    def _on_save_coco(self):
        """Save current frame + result as a COCO annotation."""
        if not self._export_preflight():
            return
        try:
            path = self._export.save_coco(
                self._current_frame_bgr,
                self._filtered_result(),
                self._current_frame_idx,
                self._lbl_filename.text(),
            )
            self._log(f"Saved COCO: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR saving COCO: {exc}")

    def _on_save_cvat(self):
        """Save current frame + result as a CVAT XML annotation."""
        if not self._export_preflight():
            return
        try:
            path = self._export.save_cvat(
                self._current_frame_bgr,
                self._filtered_result(),
                self._current_frame_idx,
                self._lbl_filename.text(),
            )
            self._log(f"Saved CVAT: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR saving CVAT: {exc}")

    def _on_save_labelstudio(self):
        """Save current frame + result as a LabelStudio task."""
        if not self._export_preflight():
            return
        try:
            path = self._export.save_labelstudio(
                self._current_frame_bgr,
                self._filtered_result(),
                self._current_frame_idx,
                self._lbl_filename.text(),
            )
            self._log(f"Saved LabelStudio: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR saving LabelStudio: {exc}")

    def _on_export_all(self):
        """Open a progress dialog and spawn ExportAllWorker."""
        if not self._engine.is_loaded():
            self._log("Please load a model first.")
            return
        if self._video.total_frames() == 0:
            self._log("Open a video first.")
            return
        if not self._output_dir:
            self._log("Set an output directory first.")
            return
        if not self._video_file_path:
            self._log("Video path unavailable — re-open the video file.")
            return

        # Fix 9: warn if existing labels will be overwritten
        labels_dir = Path(self._output_dir) / "labels"
        if labels_dir.exists() and any(labels_dir.glob("frame_*.txt")):
            reply = QMessageBox.question(
                self, "Overwrite Labels?",
                f"Existing label files in:\n{labels_dir}\n\n"
                "Export All will overwrite them. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        stride       = self._spin_stride.value()
        n_export     = len(range(0, self._video.total_frames(), stride))
        self._progress_dlg = QProgressDialog(
            "Exporting frames…", "Cancel", 0, n_export, self
        )
        self._progress_dlg.setWindowTitle("Export All Frames")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setValue(0)
        self._progress_dlg.show()

        self._export_worker = ExportAllWorker(
            self._export,
            self._video_file_path,
            self._video.total_frames(),
            self._engine,
            self._spin_conf.value(),
            self._spin_iou.value(),
            stride=stride,
            enabled_classes=self._enabled_classes,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._progress_dlg.canceled.connect(self._export_worker.cancel)
        self._export_worker.start()
        self._log(f"Export started — stride={stride}, "
                  f"{n_export} frames to process …")

    def _on_export_progress(self, current: int, total: int):
        """Update the export progress dialog."""
        if hasattr(self, "_progress_dlg") and self._progress_dlg.isVisible():
            self._progress_dlg.setValue(current)

    def _on_export_finished(self, message: str):
        """Handle export completion."""
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._log(f"Export complete — {message}")

    def _on_export_error(self, msg: str):
        """Handle export error."""
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._log(f"ERROR during export: {msg}")
        QMessageBox.critical(self, "Export Error", msg)

    def _on_review_labels(self):
        """Open ReviewDialog to browse exported label files."""
        if not self._output_dir:
            self._log("Set an output directory first.")
            return
        labels_dir = str(Path(self._output_dir) / "labels")
        dlg = ReviewDialog(labels_dir, self)
        dlg.seek_requested.connect(self._seek_to)
        dlg.exec()

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — Grounded SAM                                               #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_browse_sam_ckpt(self):
        """Open a file picker for the SAM checkpoint."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM Checkpoint", "",
            "SAM Checkpoint (*.pth);;All Files (*)"
        )
        if path:
            self._edit_sam_ckpt.setText(path)

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — SAM Box Refiner                                            #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_sam_refine_backend_changed(self, mobile_checked: bool):
        """Show/hide SAM-specific widgets when backend radio changes."""
        sam_mode = not mobile_checked
        self._edit_refine_ckpt.setVisible(sam_mode)
        self._refine_sam_browse_btn.setVisible(sam_mode)
        self._combo_refine_variant.setVisible(sam_mode)

    def _on_browse_refine_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM Checkpoint", "", "SAM Checkpoint (*.pth);;All Files (*)"
        )
        if path:
            self._edit_refine_ckpt.setText(path)

    def _on_load_sam_refiner(self):
        """Load SAM or MobileSAM in background."""
        use_mobile = self._radio_mobile_sam.isChecked()
        checkpoint = "" if use_mobile else self._edit_refine_ckpt.text().strip()
        variant    = self._combo_refine_variant.currentText()
        self._btn_load_refiner.setEnabled(False)
        self._lbl_refiner_status.setText("Loading…")
        self._lbl_refiner_status.setStyleSheet("color:#DDAA00;font-size:8pt;")
        self._sam_refine_load_worker = SAMRefineLoadWorker(
            self._sam_refiner, checkpoint, variant, use_mobile
        )
        self._sam_refine_load_worker.finished.connect(self._on_sam_refiner_loaded)
        self._sam_refine_load_worker.start()

    def _on_sam_refiner_loaded(self, success: bool, message: str):
        self._btn_load_refiner.setEnabled(True)
        if success:
            self._lbl_refiner_status.setText(message)
            self._lbl_refiner_status.setStyleSheet("color:#44CC44;font-size:8pt;")
            self._btn_auto_segment.setEnabled(True)
            self._log(f"SAM Refiner ready — {message}")
        else:
            self._lbl_refiner_status.setText(message)
            self._lbl_refiner_status.setStyleSheet("color:#CC4444;font-size:8pt;")
            self._log(f"ERROR loading SAM Refiner: {message}")

    def _on_auto_segment(self):
        """Feed current YOLO boxes into SAM → replace masks with tight polygons."""
        if not self._sam_refiner.is_loaded():
            self._log("Load SAM Refiner first.")
            return
        if self._current_result is None or not self._current_result.boxes_xyxy:
            self._log("Run YOLO inference first (no boxes to refine).")
            return
        if self._current_frame_bgr is None:
            return
        if self._sam_refine_worker and self._sam_refine_worker.isRunning():
            return
        self._push_undo()
        self._btn_auto_segment.setEnabled(False)
        self._lbl_refiner_status.setText("Refining masks…")
        self._sam_refine_worker = SAMRefineWorker(
            self._sam_refiner,
            self._current_frame_bgr.copy(),
            self._deep_copy_result(self._current_result),
        )
        self._sam_refine_worker.finished.connect(self._on_sam_refine_done)
        self._sam_refine_worker.error.connect(self._on_sam_refine_error)
        self._sam_refine_worker.start()
        self._log("Auto-segmenting boxes with SAM…")

    def _on_sam_refine_done(self, result: InferenceResult):
        self._btn_auto_segment.setEnabled(True)
        n = len(result.masks_binary)
        self._current_result = result
        self._frame_store[self._current_frame_idx] = result
        self._lbl_refiner_status.setText(f"Done — {n} mask(s) refined")
        self._lbl_refiner_status.setStyleSheet("color:#44CC44;font-size:8pt;")
        self._log(f"SAM refined {n} mask(s) on frame {self._current_frame_idx}")
        self._refresh_display()

    def _on_sam_refine_error(self, msg: str):
        self._btn_auto_segment.setEnabled(True)
        self._lbl_refiner_status.setText(f"Error: {msg}")
        self._lbl_refiner_status.setStyleSheet("color:#CC4444;font-size:8pt;")
        self._log(f"ERROR SAM refine: {msg}")

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — Training Launcher                                          #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_browse_train_data(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Root Directory")
        if path:
            self._edit_train_data.setText(path)

    def _on_train_start(self):
        """Start or stop YOLO training."""
        # If already running — cancel
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.cancel()
            self._btn_start_train.setText("Start Training")
            self._lbl_train_status.setText("Cancelled.")
            return

        # Resolve dataset directory
        data_dir = self._edit_train_data.text().strip() or self._output_dir
        if not data_dir or not os.path.isdir(str(data_dir)):
            self._log("Set a valid dataset directory (or output dir) first.")
            return

        # Ensure images/ and labels/ subdirs exist
        images_dir = os.path.join(str(data_dir), "images")
        labels_dir = os.path.join(str(data_dir), "labels")
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            self._log(
                "Dataset directory must contain images/ and labels/ sub-folders.\n"
                "Run 'Export All Frames' first."
            )
            return

        # Fix 10: pre-flight check — require at least one non-empty label file
        label_files = list(Path(labels_dir).glob("*.txt"))
        non_empty   = [f for f in label_files if f.stat().st_size > 0]
        if not non_empty:
            QMessageBox.warning(
                self, "No Labels Found",
                f"No non-empty label files found in:\n{labels_dir}\n\n"
                "Run inference and export at least a few frames before training."
            )
            self._log("Training aborted — no label files found in dataset.")
            return

        # Generate data.yaml
        try:
            from taxonomy import CLASS_NAMES as _CN
            yaml_path = self._export.generate_data_yaml(_CN)
            self._log(f"data.yaml written: {yaml_path}")
        except Exception as exc:
            self._log(f"ERROR generating data.yaml: {exc}")
            return

        base_model  = self._combo_train_model.currentText()
        epochs      = self._spin_train_epochs.value()
        batch       = self._spin_train_batch.value()
        imgsz       = self._spin_train_imgsz.value()
        project_dir = str(data_dir)

        self._txt_train_log.clear()
        self._btn_start_train.setText("Stop Training")
        self._lbl_train_status.setText(
            f"Training {base_model} for {epochs} epochs…"
        )
        self._lbl_train_status.setStyleSheet("color:#DDAA00;font-size:8pt;")

        self._train_worker = TrainingWorker(
            yaml_path, base_model, epochs, batch, imgsz, project_dir
        )
        self._train_worker.log_line.connect(self._on_train_log)
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.start()

    def _on_train_log(self, line: str):
        """Append a training output line to the log box."""
        self._txt_train_log.append(line)
        # Auto-scroll to bottom
        sb = self._txt_train_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_train_done(self, message: str):
        self._btn_start_train.setText("Start Training")
        self._lbl_train_status.setText("Done!")
        self._lbl_train_status.setStyleSheet("color:#44CC44;font-size:8pt;")
        self._log(message)
        # Offer to load the trained model
        best_pt = ""
        if "best model:" in message.lower():
            best_pt = message.split("Best model:")[-1].strip()
        if best_pt and os.path.exists(best_pt):
            reply = QMessageBox.question(
                self, "Training Complete",
                f"Training finished.\nLoad best.pt into FrameForge now?\n{best_pt}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._combo_model.setCurrentText("")
                self._combo_model.setEditText(best_pt) if hasattr(
                    self._combo_model, "setEditText"
                ) else self._combo_model.addItem(best_pt)
                self._combo_model.setCurrentText(best_pt)

    def _on_train_error(self, msg: str):
        self._btn_start_train.setText("Start Training")
        self._lbl_train_status.setText(f"Error!")
        self._lbl_train_status.setStyleSheet("color:#CC4444;font-size:8pt;")
        self._log(f"Training ERROR: {msg}")

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — Temporal Propagation (SAM 2)                              #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_browse_sam2_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM 2 Checkpoint", "",
            "SAM 2 Checkpoint (*.pt *.pth);;All Files (*)"
        )
        if path:
            self._edit_sam2_ckpt.setText(path)

    def _on_load_sam2(self):
        """Load SAM 2 video predictor in a background thread."""
        checkpoint = self._edit_sam2_ckpt.text().strip()
        if not checkpoint:
            QMessageBox.warning(
                self, "No Checkpoint",
                "Enter or browse to a SAM 2 checkpoint file (.pt).\n\n"
                "Download from:\n"
                "https://github.com/facebookresearch/segment-anything-2/releases"
            )
            return
        variant = self._combo_sam2_variant.currentText()
        self._btn_load_sam2.setEnabled(False)
        self._lbl_sam2_status.setText("Loading…")
        self._lbl_sam2_status.setStyleSheet("color:#DDAA00;font-size:8pt;")

        # Load in a lightweight worker so UI stays responsive
        class _SAM2LoadWorker(QThread):
            finished = pyqtSignal(bool, str)
            def __init__(self, engine, ckpt, variant):
                super().__init__()
                self._engine  = engine
                self._ckpt    = ckpt
                self._variant = variant
            def run(self):
                ok, msg = self._engine.load(self._ckpt, self._variant)
                self.finished.emit(ok, msg)

        self._sam2_load_worker = _SAM2LoadWorker(self._sam2_engine, checkpoint, variant)
        self._sam2_load_worker.finished.connect(self._on_sam2_loaded)
        self._sam2_load_worker.start()
        self._log(f"Loading SAM 2 ({variant})…")

    def _on_sam2_loaded(self, success: bool, message: str):
        self._btn_load_sam2.setEnabled(True)
        if success:
            self._lbl_sam2_status.setText(message)
            self._lbl_sam2_status.setStyleSheet("color:#44CC44;font-size:8pt;")
            self._update_propagation_buttons()
            self._log(f"SAM 2 ready: {message}")
        else:
            self._lbl_sam2_status.setText(f"Error: {message[:60]}")
            self._lbl_sam2_status.setStyleSheet("color:#CC4444;font-size:8pt;")
            self._log(f"ERROR loading SAM 2: {message}")

    def _update_propagation_buttons(self):
        """Enable propagation buttons only when SAM 2 is loaded and frame has annotations."""
        can_propagate = (
            self._sam2_engine.is_loaded()
            and self._current_result is not None
            and len(self._current_result.class_ids) > 0
            and bool(self._video_file_path)
        )
        for btn in (self._btn_prop_forward, self._btn_prop_backward, self._btn_prop_both):
            btn.setEnabled(can_propagate)

    def _on_propagate(self, direction: str):
        """Start SAM 2 propagation in the given direction."""
        if self._propagation_worker and self._propagation_worker.isRunning():
            return

        if not self._sam2_engine.is_loaded():
            self._log("Load SAM 2 first.")
            return
        if self._current_result is None or not self._current_result.class_ids:
            self._log("Run inference on the current frame first (need annotations to propagate).")
            return
        if not self._video_file_path:
            self._log("Open a video first.")
            return

        total   = self._video.total_frames()
        cur_idx = self._current_frame_idx
        rng     = self._spin_prop_range.value()

        if direction == "both":
            # Run forward first, then backward
            self._log(f"Propagating both directions — {rng} frames each…")
            self._start_propagation_run(cur_idx, cur_idx, min(cur_idx + rng, total - 1), reverse=False, chain_backward=True)
        elif direction == "forward":
            end_idx = min(cur_idx + rng, total - 1)
            if end_idx <= cur_idx:
                self._log("Already at last frame — nothing to propagate forward.")
                return
            self._log(f"Propagating forward: frame {cur_idx} → {end_idx}…")
            self._start_propagation_run(cur_idx, cur_idx, end_idx, reverse=False, chain_backward=False)
        else:  # backward
            start_idx = max(cur_idx - rng, 0)
            if start_idx >= cur_idx:
                self._log("Already at first frame — nothing to propagate backward.")
                return
            self._log(f"Propagating backward: frame {cur_idx} → {start_idx}…")
            self._start_propagation_run(cur_idx, start_idx, cur_idx, reverse=True, chain_backward=False)

    def _start_propagation_run(
        self, keyframe_idx: int, start_idx: int, end_idx: int,
        reverse: bool, chain_backward: bool
    ):
        """Spawn a PropagationWorker for one direction."""
        self._prop_chain_backward    = chain_backward
        self._prop_keyframe_idx      = keyframe_idx
        self._prop_keyframe_result   = self._deep_copy_result(self._current_result)

        for btn in (self._btn_prop_forward, self._btn_prop_backward, self._btn_prop_both):
            btn.setEnabled(False)
        self._btn_prop_cancel.setEnabled(True)
        self._lbl_prop_progress.setText("Starting…")
        self._lbl_prop_progress.setStyleSheet("color:#DDAA00;font-size:8pt;")

        self._propagation_worker = PropagationWorker(
            engine          = self._sam2_engine,
            video_path      = self._video_file_path,
            keyframe_idx    = keyframe_idx,
            keyframe_result = self._prop_keyframe_result,
            start_idx       = start_idx,
            end_idx         = end_idx,
            reverse         = reverse,
        )
        self._propagation_worker.progress.connect(self._on_prop_progress)
        self._propagation_worker.frame_done.connect(self._on_prop_frame_done)
        self._propagation_worker.finished.connect(
            lambda n: self._on_prop_finished(n, chain_backward=chain_backward,
                                              keyframe_idx=keyframe_idx)
        )
        self._propagation_worker.error.connect(self._on_prop_error)
        self._propagation_worker.start()

    def _on_prop_progress(self, done: int, total: int):
        pct = int(100 * done / total) if total else 0
        self._lbl_prop_progress.setText(f"Frame {done}/{total}  ({pct}%)")

    def _on_prop_frame_done(self, frame_idx: int, result: InferenceResult):
        """Store each propagated frame into the annotation frame_store."""
        # Never overwrite the keyframe's original human-reviewed annotation
        if frame_idx == self._current_frame_idx:
            return
        self._frame_store[frame_idx] = result
        # If this is the currently displayed frame refresh it live
        if frame_idx == self._current_frame_idx:
            self._current_result = self._deep_copy_result(result)
            self._refresh_display()

    def _on_prop_finished(self, n: int, chain_backward: bool, keyframe_idx: int):
        if chain_backward:
            # First pass (forward) done → now run backward
            rng       = self._spin_prop_range.value()
            start_idx = max(keyframe_idx - rng, 0)
            if start_idx < keyframe_idx:
                self._lbl_prop_progress.setText(
                    f"Forward done ({n} frames). Running backward…"
                )
                self._log(f"Forward propagation done ({n} frames). Now propagating backward…")
                self._start_propagation_run(
                    keyframe_idx, start_idx, keyframe_idx,
                    reverse=True, chain_backward=False
                )
                return
        # All done
        self._btn_prop_cancel.setEnabled(False)
        self._update_propagation_buttons()
        self._lbl_prop_progress.setText(f"Done — {n} frames propagated.")
        self._lbl_prop_progress.setStyleSheet("color:#44CC44;font-size:8pt;")
        self._log(f"Propagation complete: {n} frames added to annotation store.")

    def _on_prop_error(self, msg: str):
        self._btn_prop_cancel.setEnabled(False)
        self._update_propagation_buttons()
        self._lbl_prop_progress.setText(f"Error: {msg[:60]}")
        self._lbl_prop_progress.setStyleSheet("color:#CC4444;font-size:8pt;")
        self._log(f"ERROR during propagation: {msg}")

    def _on_propagate_cancel(self):
        if self._propagation_worker and self._propagation_worker.isRunning():
            self._propagation_worker.cancel()
        self._btn_prop_cancel.setEnabled(False)
        self._update_propagation_buttons()
        self._lbl_prop_progress.setText("Cancelled.")
        self._lbl_prop_progress.setStyleSheet("color:#888888;font-size:8pt;")
        self._log("Propagation cancelled.")

    def _on_load_gsam(self):
        """Load Grounding DINO + optional SAM in background."""
        self._btn_load_gsam.setEnabled(False)
        self._lbl_gsam_status.setText("Loading…")
        self._lbl_gsam_status.setStyleSheet("color:#DDAA00;font-size:8pt;")
        sam_ckpt    = self._edit_sam_ckpt.text().strip()
        sam_variant = self._combo_sam_variant.currentText()
        self._gsam_model_worker = GSAMModelLoader(
            self._gsam_engine, sam_ckpt, sam_variant
        )
        self._gsam_model_worker.finished.connect(self._on_gsam_loaded)
        self._gsam_model_worker.start()
        self._log("Loading Grounded SAM…")

    def _on_gsam_loaded(self, success: bool, message: str):
        """Handle GSAMModelLoader.finished."""
        self._btn_load_gsam.setEnabled(True)
        if success:
            self._lbl_gsam_status.setText(message)
            self._lbl_gsam_status.setStyleSheet("color:#44CC44;font-size:8pt;")
            self._btn_run_gsam.setEnabled(True)
            self._log(f"Grounded SAM ready — {message}")
        else:
            self._lbl_gsam_status.setText(message)
            self._lbl_gsam_status.setStyleSheet("color:#CC4444;font-size:8pt;")
            self._log(f"ERROR loading Grounded SAM: {message}")

    def _on_run_gsam(self):
        """Run Grounded SAM inference on the current frame."""
        if not self._gsam_engine.is_loaded():
            self._log("Load Grounded SAM model first.")
            return
        if self._current_frame_bgr is None:
            self._log("Open a video first.")
            return
        if self._gsam_infer_worker and self._gsam_infer_worker.isRunning():
            return
        self._btn_run_gsam.setEnabled(False)
        self._lbl_gsam_status.setText("Running inference…")
        self._lbl_gsam_status.setStyleSheet("color:#DDAA00;font-size:8pt;")
        prompt = self._edit_gsam_prompt.text().strip() or GROUNDED_SAM_PROMPT
        self._gsam_infer_worker = GSAMInferenceWorker(
            self._gsam_engine,
            self._current_frame_bgr.copy(),
            prompt,
            self._spin_gsam_box_thr.value(),
            self._spin_gsam_txt_thr.value(),
        )
        self._gsam_infer_worker.finished.connect(self._on_gsam_done)
        self._gsam_infer_worker.error.connect(self._on_gsam_error)
        self._gsam_infer_worker.start()
        self._log(f"Grounded SAM inference running…")

    def _on_gsam_done(self, result: InferenceResult):
        """Handle GSAMInferenceWorker.finished — store as current result."""
        self._current_result     = result
        self._selected_detection = -1
        self._frame_viewer.set_selection(None)
        self._lbl_sel_info.setText("Nothing selected")
        self._btn_run_gsam.setEnabled(True)
        self._lbl_gsam_status.setText(self._gsam_engine.status())
        self._lbl_gsam_status.setStyleSheet("color:#44CC44;font-size:8pt;")
        for cid, cname in zip(result.class_ids, result.class_names):
            self._known_classes[cid] = cname
        self._frame_store[self._current_frame_idx] = self._deep_copy_result(result)
        self._undo_stacks.pop(self._current_frame_idx, None)
        self._redo_stacks.pop(self._current_frame_idx, None)
        self._update_auto_segment_btn()
        self._update_propagation_buttons()
        self._refresh_display()
        self._log(
            f"Grounded SAM: {len(result.class_ids)} detection(s) "
            f"on frame {self._current_frame_idx}"
        )

    def _on_gsam_error(self, msg: str):
        self._btn_run_gsam.setEnabled(True)
        self._lbl_gsam_status.setText(f"Error: {msg[:60]}")
        self._lbl_gsam_status.setStyleSheet("color:#CC4444;font-size:8pt;")
        self._log(f"ERROR during Grounded SAM inference: {msg}")

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — Semantic Segmentation                                      #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_load_semantic(self):
        """Load SegFormer model in background."""
        model_id = self._combo_sem_model.currentText()
        if self._sem_engine.is_loaded() and model_id in self._sem_engine.status():
            self._log(f"Semantic model already loaded: {self._sem_engine.status()}")
            return
        self._btn_load_sem.setEnabled(False)
        self._lbl_sem_status.setText("Downloading/loading model…")
        self._lbl_sem_status.setStyleSheet("color:#DDAA00;font-size:8pt;")
        model_id = self._combo_sem_model.currentText()
        self._sem_model_worker = SemanticModelLoader(self._sem_engine, model_id)
        self._sem_model_worker.finished.connect(self._on_semantic_loaded)
        self._sem_model_worker.start()
        self._log(f"Loading semantic model: {model_id} …")

    def _on_semantic_loaded(self, success: bool, message: str):
        self._btn_load_sem.setEnabled(True)
        if success:
            self._lbl_sem_status.setText(message)
            self._lbl_sem_status.setStyleSheet("color:#44CC44;font-size:8pt;")
            self._btn_run_sem.setEnabled(True)
            self._btn_run_sem_all.setEnabled(True)
            self._log(f"Semantic model ready — {message}")
        else:
            self._lbl_sem_status.setText(message)
            self._lbl_sem_status.setStyleSheet("color:#CC4444;font-size:8pt;")
            self._log(f"ERROR loading semantic model: {message}")

    def _on_run_semantic(self):
        """Run semantic segmentation on the current frame."""
        if not self._sem_engine.is_loaded():
            self._log("Load semantic model first.")
            return
        if self._current_frame_bgr is None:
            self._log("Open a video first.")
            return
        if self._sem_infer_worker and self._sem_infer_worker.isRunning():
            return
        self._btn_run_sem.setEnabled(False)
        self._sem_infer_worker = SemanticInferenceWorker(
            self._sem_engine, self._current_frame_bgr.copy()
        )
        self._sem_infer_worker.finished.connect(self._on_semantic_done)
        self._sem_infer_worker.error.connect(self._on_semantic_error)
        self._sem_infer_worker.start()
        self._log("Semantic segmentation running…")

    def _on_semantic_done(self, result: SemanticResult):
        self._current_semantic = result
        self._semantic_store[self._current_frame_idx] = result
        self._btn_run_sem.setEnabled(True)
        self._btn_save_semantic.setEnabled(True)
        self._log(
            f"Semantic seg done on frame {self._current_frame_idx} "
            f"({len(result.class_names)} classes)"
        )
        # Auto-switch to semantic view
        if not self._view_semantic:
            self._btn_view_semantic.setChecked(True)
            self._view_semantic = True
        self._refresh_display()

    def _on_semantic_error(self, msg: str):
        self._btn_run_sem.setEnabled(True)
        self._log(f"ERROR during semantic segmentation: {msg}")

    # ── Batch semantic ────────────────────────────────────────────────────

    def _on_run_semantic_all(self):
        """Start batch semantic segmentation across all frames."""
        if not self._sem_engine.is_loaded():
            self._log("Load semantic model first.")
            return
        if not self._video_file_path:
            self._log("Open a video first.")
            return
        if self._sem_batch_worker and self._sem_batch_worker.isRunning():
            # Cancel running batch
            self._sem_batch_worker.cancel()
            self._btn_run_sem_all.setText("Run All Frames")
            self._lbl_sem_batch_progress.setText("Cancelled.")
            if hasattr(self, "_spin_stride"):
                self._spin_stride.setEnabled(True)
            return
        stride = self._spin_stride.value() if hasattr(self, "_spin_stride") else 1
        # Fix 14: lock stride spinbox while batch is running
        if hasattr(self, "_spin_stride"):
            self._spin_stride.setEnabled(False)
        self._sem_batch_worker = SemanticBatchWorker(
            self._sem_engine,
            self._video_file_path,
            self._video.total_frames(),
            stride,
            already_cached=set(self._semantic_store.keys()),
        )
        self._sem_batch_worker.progress.connect(self._on_sem_batch_progress)
        self._sem_batch_worker.frame_done.connect(self._on_sem_batch_frame_done)
        self._sem_batch_worker.finished.connect(self._on_sem_batch_finished)
        self._sem_batch_worker.error.connect(self._on_sem_batch_error)
        self._sem_batch_worker.start()
        self._btn_run_sem_all.setText("Cancel Batch")
        self._lbl_sem_batch_progress.setText("Starting…")
        self._log(f"Batch semantic started (stride={stride})…")

    def _on_sem_batch_progress(self, done: int, total: int):
        pct = int(100 * done / total) if total else 0
        self._lbl_sem_batch_progress.setText(f"Frame {done}/{total}  ({pct}%)")

    def _on_sem_batch_frame_done(self, frame_idx: int, result):
        """Cache each frame result as it arrives; refresh if it's the current frame."""
        self._semantic_store[frame_idx] = result
        if frame_idx == self._current_frame_idx:
            self._current_semantic = result
            self._btn_save_semantic.setEnabled(True)
            if not self._view_semantic:
                self._btn_view_semantic.setChecked(True)
                self._view_semantic = True
            self._refresh_display()

    def _on_sem_batch_finished(self, n: int):
        self._btn_run_sem_all.setText("Run All Frames")
        self._lbl_sem_batch_progress.setText(f"Done — {n} frames processed.")
        if hasattr(self, "_spin_stride"):
            self._spin_stride.setEnabled(True)
        self._log(f"Batch semantic complete: {n} frames cached.")
        # Refresh display in case current frame result just arrived
        if self._current_frame_idx in self._semantic_store:
            self._current_semantic = self._semantic_store[self._current_frame_idx]
            self._btn_save_semantic.setEnabled(True)
            self._refresh_display()

    def _on_sem_batch_error(self, msg: str):
        self._btn_run_sem_all.setText("Run All Frames")
        self._lbl_sem_batch_progress.setText(f"Error: {msg}")
        if hasattr(self, "_spin_stride"):
            self._spin_stride.setEnabled(True)
        self._log(f"ERROR batch semantic: {msg}")

    def _on_save_semantic(self):
        """Save semantic result as PNG label map + colour PNG."""
        if not self._output_dir:
            self._log("Set an output directory first.")
            return
        if self._current_semantic is None:
            self._log("Run semantic segmentation first.")
            return
        if self._current_frame_bgr is None:
            return
        try:
            lp, cp = self._export.save_semantic_png(
                self._current_frame_bgr,
                self._current_semantic,
                self._current_frame_idx,
            )
            self._log(f"Semantic PNG saved: {os.path.basename(lp)}")
        except Exception as exc:
            self._log(f"ERROR saving semantic PNG: {exc}")

    def _on_view_mode_changed(self, mode_id: int):
        """Switch between instance overlay (0) and semantic view (1)."""
        self._view_semantic = (mode_id == 1)
        self._refresh_display()

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — Active Learning                                            #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_al_scan(self):
        """Scan frame store + disk labels and flag low-quality frames."""
        if self._al_worker and self._al_worker.isRunning():
            return
        labels_dir  = str(Path(self._output_dir) / "labels") if self._output_dir else ""
        total       = self._video.total_frames()
        conf_thr    = self._spin_al_conf.value()
        self._btn_al_scan.setEnabled(False)
        self._lbl_al_status.setText("Scanning…")
        self._al_worker = ActiveLearningWorker(
            self._frame_store, labels_dir, conf_thr, total
        )
        self._al_worker.finished.connect(self._on_al_scan_done)
        self._al_worker.start()
        self._log("Active learning scan started…")

    def _on_al_scan_done(self, flagged: list):
        self._flagged_frames = flagged
        self._btn_al_scan.setEnabled(True)
        n = len(flagged)
        if n == 0:
            self._lbl_al_status.setText("No frames flagged.")
            self._lbl_al_status.setStyleSheet("color:#44CC44;font-size:8pt;")
            self._btn_al_review.setEnabled(False)
        else:
            self._lbl_al_status.setText(f"{n} frame(s) flagged for review.")
            self._lbl_al_status.setStyleSheet("color:#DDAA00;font-size:8pt;")
            self._btn_al_review.setEnabled(True)
        self._log(f"Active learning scan complete — {n} frame(s) flagged.")

    def _on_al_review(self):
        """Open the ActiveLearningDialog with the flagged frame list."""
        if not self._flagged_frames:
            self._log("No flagged frames — run Scan first.")
            return
        dlg = ActiveLearningDialog(self._flagged_frames, self)
        dlg.seek_requested.connect(self._seek_to)
        dlg.exec()

    def _on_dataset_stats(self):
        """Open StatisticsDialog to show class distribution."""
        if not self._output_dir:
            self._log("Set an output directory first.")
            return
        labels_dir = str(Path(self._output_dir) / "labels")
        StatisticsDialog(labels_dir, self._known_classes, self).exec()

    # ══════════════════════════════════════════════════════════════════ #
    #  Frame navigation & display                                         #
    # ══════════════════════════════════════════════════════════════════ #

    def _seek_to(self, idx: int):
        """Seek to frame *idx* (clamped), persist current result, display, auto-infer."""
        total = self._video.total_frames()
        if total == 0:
            return
        idx = max(0, min(idx, total - 1))
        # Persist current annotations before leaving this frame
        if self._current_result is not None:
            self._frame_store[self._current_frame_idx] = \
                self._deep_copy_result(self._current_result)
        self._current_frame_idx = idx
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._display_frame(idx)
        self._maybe_auto_infer()

    def _on_slider_value_changed(self, value: int):
        """Handle QSlider valueChanged."""
        if self._video.total_frames() == 0:
            return
        if value != self._current_frame_idx:
            self._seek_to(value)

    def _display_frame(self, idx: int):
        """Read frame from VideoHandler, update FPS window and labels."""
        now = time.monotonic()
        if self._last_frame_time > 0.0:
            elapsed = now - self._last_frame_time
            if elapsed > 0.0:
                self._fps_window.append(1.0 / elapsed)
        self._last_frame_time = now

        frame = self._video.get_frame(idx)
        if frame is None:
            return
        self._current_frame_bgr  = frame
        self._selected_detection = -1
        self._frame_viewer.set_selection(None)
        self._lbl_sel_info.setText("Nothing selected")
        # Restore annotations from frame store if available
        stored = self._frame_store.get(idx)
        self._current_result   = self._deep_copy_result(stored) if stored else None
        # Restore semantic result from store if available, else clear
        self._current_semantic = self._semantic_store.get(idx, None)
        self._btn_save_semantic.setEnabled(self._current_semantic is not None)
        self._frame_viewer.reset_zoom()
        self._update_auto_segment_btn()
        self._update_propagation_buttons()
        self._refresh_display()

        total = self._video.total_frames()
        self._lbl_frame_pos.setText(f"Frame {idx + 1} / {total}")
        self._sb_frame.setText(f"Frame: {idx + 1}/{total}")

    def _refresh_display(self):
        """
        Re-render the frame, applying the active view mode:
        - Semantic mode: show SegFormer colour map when available.
        - Instance mode: show YOLO/G-SAM overlay when available.
        """
        if self._current_frame_bgr is None:
            return

        # ── Semantic view ────────────────────────────────────────────────
        if self._view_semantic:
            if self._current_semantic is not None:
                self._show_bgr(self._current_semantic.colormap)
            else:
                # Blend a placeholder tint so user knows semantic mode is active
                tint = self._current_frame_bgr.copy()
                tint[:, :, 1] = (tint[:, :, 1] * 0.4).astype(tint.dtype)
                self._show_bgr(tint)
            # Still draw selection highlight on top
            if self._selected_detection >= 0 and self._current_result is not None:
                idx = self._selected_detection
                if idx < len(self._current_result.boxes_xyxy):
                    self._frame_viewer.set_selection(
                        tuple(self._current_result.boxes_xyxy[idx])
                    )
            return

        # ── Instance view (default) ──────────────────────────────────────
        if (self._current_result is not None
                and self._chk_show_overlay.isChecked()):
            display_result = ExportHandler.filter_result(
                self._current_result, self._enabled_classes
            )
            annotated = self._engine.draw_overlay(
                self._current_frame_bgr, display_result
            )
            self._show_bgr(annotated)
        else:
            self._show_bgr(self._current_frame_bgr)

        # Restore selection highlight (may have been cleared by _show_bgr)
        if self._selected_detection >= 0 and self._current_result is not None:
            idx = self._selected_detection
            if idx < len(self._current_result.boxes_xyxy):
                self._frame_viewer.set_selection(
                    tuple(self._current_result.boxes_xyxy[idx])
                )

    def _show_bgr(self, frame_bgr: np.ndarray):
        """Convert BGR numpy frame to QPixmap and display it."""
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._frame_viewer.set_frame_pixmap(QPixmap.fromImage(qimg))

    def _advance_frame(self):
        """QTimer slot: advance one frame during playback."""
        total = self._video.total_frames()
        if total == 0:
            return
        # Fix 2: skip advance tick if inference is already running to avoid race
        if (self._infer_worker is not None and self._infer_worker.isRunning()) or \
           (self._gsam_infer_worker is not None and self._gsam_infer_worker.isRunning()):
            return
        next_idx = self._current_frame_idx + 1
        if next_idx >= total:
            self._on_play_pause()
            return
        self._seek_to(next_idx)

    def _on_play_pause(self):
        """Toggle playback state."""
        if self._is_playing:
            self._play_timer.stop()
            self._is_playing = False
            self._btn_play.setText("▶  Play")
        else:
            if self._video.total_frames() == 0:
                return
            self._play_timer.start()
            self._is_playing = True
            self._btn_play.setText("⏸  Pause")

    # ══════════════════════════════════════════════════════════════════ #
    #  Undo / redo                                                        #
    # ══════════════════════════════════════════════════════════════════ #

    @staticmethod
    def _deep_copy_result(result: "InferenceResult | None") -> "InferenceResult | None":
        """Return a deep copy of *result*, or ``None``."""
        if result is None:
            return None
        return InferenceResult(
            boxes_xyxy   = [list(b) for b in result.boxes_xyxy],
            class_ids    = list(result.class_ids),
            confidences  = list(result.confidences),
            masks_binary = [m.copy() for m in result.masks_binary],
            class_names  = list(result.class_names),
            orig_shape   = result.orig_shape,
        )

    def _push_undo(self):
        """Snapshot the current result onto this frame's undo stack (max 20)."""
        fid = self._current_frame_idx
        stack = self._undo_stacks.setdefault(fid, [])
        stack.append(self._deep_copy_result(self._current_result))
        if len(stack) > 20:
            stack.pop(0)
        # Any new action clears the redo stack
        self._redo_stacks.pop(fid, None)

    def _on_undo(self):
        """Restore the previous annotation state for the current frame."""
        fid   = self._current_frame_idx
        stack = self._undo_stacks.get(fid, [])
        if not stack:
            self._log("Nothing to undo.")
            return
        # Push current to redo stack
        redo = self._redo_stacks.setdefault(fid, [])
        redo.append(self._deep_copy_result(self._current_result))
        # Pop and restore
        self._current_result     = stack.pop()
        self._selected_detection = -1
        self._frame_viewer.set_selection(None)
        self._lbl_sel_info.setText("Nothing selected")
        self._refresh_display()
        self._log("Undo.")

    def _on_redo(self):
        """Re-apply the most recently undone change for the current frame."""
        fid   = self._current_frame_idx
        stack = self._redo_stacks.get(fid, [])
        if not stack:
            self._log("Nothing to redo.")
            return
        undo = self._undo_stacks.setdefault(fid, [])
        undo.append(self._deep_copy_result(self._current_result))
        self._current_result     = stack.pop()
        self._selected_detection = -1
        self._frame_viewer.set_selection(None)
        self._lbl_sel_info.setText("Nothing selected")
        self._refresh_display()
        self._log("Redo.")

    # ══════════════════════════════════════════════════════════════════ #
    #  Session save / load                                                #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_save_session(self):
        """Serialise the in-memory frame store to a JSON session file."""
        # First persist the current frame
        if self._current_result is not None:
            self._frame_store[self._current_frame_idx] = \
                self._deep_copy_result(self._current_result)
        if not self._frame_store:
            self._log("No annotations to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "FrameForge Session (*.ffses);;All Files (*)"
        )
        if not path:
            return
        try:
            session: dict = {
                "known_classes": {str(k): v for k, v in self._known_classes.items()},
                "frames": {},
            }
            for fid, result in self._frame_store.items():
                masks_b64 = []
                for m in result.masks_binary:
                    ok, buf = cv2.imencode(".png", m)
                    if ok:
                        masks_b64.append(
                            base64.b64encode(buf.tobytes()).decode("ascii")
                        )
                    else:
                        masks_b64.append("")
                session["frames"][str(fid)] = {
                    "boxes_xyxy":   result.boxes_xyxy,
                    "class_ids":    result.class_ids,
                    "confidences":  result.confidences,
                    "masks_binary": masks_b64,
                    "class_names":  result.class_names,
                    "orig_shape":   list(result.orig_shape),
                }
            Path(path).write_text(
                json.dumps(session, indent=2), encoding="utf-8"
            )
            self._log(f"Session saved: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR saving session: {exc}")

    def _on_load_session(self):
        """Load a previously saved session file into the frame store."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "FrameForge Session (*.ffses);;All Files (*)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            # Restore known classes
            for k, v in data.get("known_classes", {}).items():
                self._known_classes[int(k)] = v
            # Restore frame store
            self._frame_store.clear()
            for fid_str, entry in data.get("frames", {}).items():
                masks = []
                for b64 in entry.get("masks_binary", []):
                    if b64:
                        buf = np.frombuffer(
                            base64.b64decode(b64), dtype=np.uint8
                        )
                        m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            masks.append(m)
                            continue
                    h, w = entry["orig_shape"]
                    masks.append(np.zeros((h, w), dtype=np.uint8))
                self._frame_store[int(fid_str)] = InferenceResult(
                    boxes_xyxy  = entry["boxes_xyxy"],
                    class_ids   = entry["class_ids"],
                    confidences = entry["confidences"],
                    masks_binary= masks,
                    class_names = entry["class_names"],
                    orig_shape  = tuple(entry["orig_shape"]),
                )
            # Fix 4: warn if session resolution doesn't match current video
            if self._current_frame_bgr is not None and self._frame_store:
                vid_h, vid_w = self._current_frame_bgr.shape[:2]
                first_result = next(iter(self._frame_store.values()))
                sess_h, sess_w = first_result.orig_shape
                if (sess_h, sess_w) != (vid_h, vid_w):
                    QMessageBox.warning(
                        self, "Resolution Mismatch",
                        f"Session was created at {sess_w}×{sess_h} but the current video "
                        f"is {vid_w}×{vid_h}.\n\nAnnotation masks may not align correctly."
                    )
            # Re-display current frame if it now has stored annotations
            stored = self._frame_store.get(self._current_frame_idx)
            if stored and self._current_frame_bgr is not None:
                self._current_result = self._deep_copy_result(stored)
                self._refresh_display()
            n = len(self._frame_store)
            self._log(f"Session loaded: {n} annotated frame(s) from {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR loading session: {exc}")

    # ══════════════════════════════════════════════════════════════════ #
    #  Helpers                                                            #
    # ══════════════════════════════════════════════════════════════════ #

    def _update_auto_segment_btn(self):
        """Fix 11: enable Auto-Segment only when refiner loaded AND boxes exist."""
        has_boxes = (
            self._current_result is not None
            and len(self._current_result.boxes_xyxy) > 0
        )
        self._btn_auto_segment.setEnabled(
            self._sam_refiner.is_loaded() and has_boxes
        )

    def _get_display_indices(self) -> list[int]:
        """
        Return indices into ``_current_result`` that pass the current class
        filter (i.e. the detections currently visible on screen).
        """
        if self._current_result is None:
            return []
        if self._enabled_classes is None:
            return list(range(len(self._current_result.class_ids)))
        return [
            i for i, cid in enumerate(self._current_result.class_ids)
            if cid in self._enabled_classes
        ]

    # ══════════════════════════════════════════════════════════════════ #
    #  Status bar                                                         #
    # ══════════════════════════════════════════════════════════════════ #

    def _update_status_bar(self):
        """Refresh device, VRAM, and FPS labels every 2 seconds."""
        try:
            import torch
            if torch.cuda.is_available():
                self._sb_device.setText(
                    f"Device: CUDA ({torch.cuda.get_device_name(0)})"
                )
            else:
                self._sb_device.setText("Device: CPU")
        except Exception:
            self._sb_device.setText("Device: CPU")

        self._sb_vram.setText(f"VRAM: {self._engine.vram_used_mb():.0f} MB")

        if self._fps_window:
            avg = sum(self._fps_window) / len(self._fps_window)
            self._sb_fps.setText(f"FPS: {avg:.1f}")
        else:
            self._sb_fps.setText("FPS: --")

    # ══════════════════════════════════════════════════════════════════ #
    #  Logging                                                            #
    # ══════════════════════════════════════════════════════════════════ #

    def _log(self, message: str):
        """
        Prepend a timestamped entry to the log widget (max 200 entries).

        Parameters
        ----------
        message : str
            Human-readable log message.
        """
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{ts}] {message}")
        self._log_list.insertItem(0, item)
        while self._log_list.count() > 200:
            self._log_list.takeItem(self._log_list.count() - 1)

    # ══════════════════════════════════════════════════════════════════ #
    #  Window lifecycle                                                   #
    # ══════════════════════════════════════════════════════════════════ #

    def closeEvent(self, event):
        """Stop timers and release video capture on close."""
        self._play_timer.stop()
        self._status_timer.stop()
        self._video.release()
        event.accept()

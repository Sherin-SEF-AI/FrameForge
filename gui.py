"""
gui.py
──────
MainWindow and background-worker classes for FrameForge.

Components
----------
FrameViewer        — Custom QLabel that scales the video frame correctly at
                     any window size while preserving aspect ratio.
ModelLoaderWorker  — QThread for non-blocking model loading.
InferenceWorker    — QThread for non-blocking single-frame inference.
ExportAllWorker    — QThread for non-blocking full-video YOLO export.
MainWindow         — Primary application window: full UI layout, transport
                     controls, status bar, and all signal/slot wiring.
"""

import datetime
import logging
import os
import time
from collections import deque

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QScrollArea,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,
    QSlider, QListWidget, QListWidgetItem,
    QProgressDialog, QMessageBox, QFileDialog,
    QSizePolicy, QStatusBar, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter

from video_handler import VideoHandler
from inference_engine import InferenceEngine, InferenceResult
from export_handler import ExportHandler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Background worker threads                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

class ModelLoaderWorker(QThread):
    """
    Background thread that loads a YOLO / FastSAM model without blocking
    the GUI event loop.

    Signals
    -------
    finished : (bool, str)
        Emitted when loading completes.  ``bool`` is ``True`` on success;
        ``str`` is a human-readable status message.
    """

    finished = pyqtSignal(bool, str)

    def __init__(self, engine: InferenceEngine, model_name: str):
        """
        Parameters
        ----------
        engine : InferenceEngine
            Shared inference engine instance.
        model_name : str
            Filename of the weights to load (e.g. ``"yolo11n-seg.pt"``).
        """
        super().__init__()
        self._engine = engine
        self._model_name = model_name

    def run(self):
        """Load the model and emit ``finished``."""
        try:
            ok = self._engine.load_model(self._model_name)
            if ok:
                dev = self._engine.device().upper()
                self.finished.emit(True, f"Loaded on {dev}")
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
        Emitted with ``(InferenceResult, annotated_bgr_frame)`` on success.
    error : str
        Emitted with a description if an exception is raised.
    """

    finished = pyqtSignal(object, object)   # (InferenceResult, np.ndarray)
    error    = pyqtSignal(str)

    def __init__(
        self,
        engine: InferenceEngine,
        frame: np.ndarray,
        conf: float,
        iou: float,
    ):
        """
        Parameters
        ----------
        engine : InferenceEngine
            Loaded inference engine.
        frame : np.ndarray
            BGR frame to infer on (already copied by the caller).
        conf : float
            Confidence threshold.
        iou : float
            IoU / NMS threshold.
        """
        super().__init__()
        self._engine = engine
        self._frame  = frame
        self._conf   = conf
        self._iou    = iou

    def run(self):
        """Run inference and emit ``finished`` or ``error``."""
        try:
            result    = self._engine.infer(self._frame, self._conf, self._iou)
            annotated = self._engine.draw_overlay(self._frame, result)
            self.finished.emit(result, annotated)
        except Exception as exc:
            self.error.emit(str(exc))


# --------------------------------------------------------------------------- #

class ExportAllWorker(QThread):
    """
    Background thread that runs inference on every video frame and writes
    YOLO annotation files to disk.

    Signals
    -------
    progress : (int, int)
        ``(current_frame, total_frames)`` after each frame is processed.
    finished : str
        Completion message when all frames have been exported.
    error : str
        Error description if an unrecoverable exception occurs.
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(
        self,
        export_handler: ExportHandler,
        video_handler: VideoHandler,
        inference_engine: InferenceEngine,
        conf: float,
        iou: float,
    ):
        """
        Parameters
        ----------
        export_handler : ExportHandler
            Export handler with output directory already configured.
        video_handler : VideoHandler
            Open video source.
        inference_engine : InferenceEngine
            Loaded inference engine.
        conf : float
            Confidence threshold.
        iou : float
            IoU threshold.
        """
        super().__init__()
        self._export = export_handler
        self._video  = video_handler
        self._engine = inference_engine
        self._conf   = conf
        self._iou    = iou

    def run(self):
        """Export all frames and emit progress / finished / error signals."""
        try:
            self._export.export_all_frames(
                self._video,
                self._engine,
                self._conf,
                self._iou,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit("All frames exported successfully.")
        except Exception as exc:
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════ #
#  Custom widget                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class FrameViewer(QLabel):
    """
    A QLabel subclass that scales and centres the current video frame at
    any window size while preserving the original aspect ratio.

    Overrides ``paintEvent`` to draw a QPixmap using
    ``Qt.AspectRatioMode.KeepAspectRatio`` and
    ``Qt.TransformationMode.SmoothTransformation``.
    """

    def __init__(self, parent=None):
        """Initialise with a dark background and sensible minimum size."""
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1E1E1E;")
        self._pixmap: QPixmap | None = None

    def set_frame_pixmap(self, pixmap: QPixmap):
        """
        Store a new pixmap and schedule a repaint.

        Parameters
        ----------
        pixmap : QPixmap
            The pixmap to display.
        """
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        """
        Override to draw the stored pixmap scaled with aspect ratio
        preservation and smooth transformation, centred in the widget.
        """
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        scaled = self._pixmap.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width()  - scaled.width())  // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def _section_label(title: str) -> QLabel:
    """
    Create a bold QLabel section header with a 1 px bottom border divider.

    Parameters
    ----------
    title : str
        Section title text.

    Returns
    -------
    QLabel
        Styled section header label.
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
#  Main Window                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class MainWindow(QMainWindow):
    """
    Primary application window for FrameForge.

    Orchestrates VideoHandler, InferenceEngine, and ExportHandler through
    PyQt6 signals/slots and background QThread workers so that the GUI
    event loop is never blocked by long-running operations.
    """

    def __init__(self):
        """Initialise all components, build the UI, and wire signals."""
        super().__init__()
        self.setWindowTitle("FrameForge — Auto-Label & Pseudo-Label Studio")
        self.resize(1280, 720)

        # ── Core components ────────────────────────────────────────────
        self._video  = VideoHandler()
        self._engine = InferenceEngine()
        self._export = ExportHandler()

        # ── Application state ──────────────────────────────────────────
        self._current_frame_idx:  int                     = 0
        self._current_frame_bgr:  np.ndarray | None       = None
        self._current_result:     InferenceResult | None  = None
        self._output_dir:         str                     = ""
        self._is_playing:         bool                    = False
        self._last_infer_time:    float                   = 0.0
        self._last_frame_time:    float                   = 0.0
        self._fps_window:         deque                   = deque(maxlen=10)

        # Worker references (kept alive for the thread lifetime)
        self._model_worker:  ModelLoaderWorker  | None = None
        self._infer_worker:  InferenceWorker    | None = None
        self._export_worker: ExportAllWorker    | None = None

        # ── Build UI & wire signals ────────────────────────────────────
        self._build_ui()
        self._build_status_bar()
        self._wire_signals()

        # ── Timers ────────────────────────────────────────────────────
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)          # ≈ 30 fps
        self._play_timer.timeout.connect(self._advance_frame)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)      # refresh every 2 s
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start()

    # ══════════════════════════════════════════════════════════════════ #
    #  UI construction                                                    #
    # ══════════════════════════════════════════════════════════════════ #

    def _build_ui(self):
        """Construct and lay out all widgets inside the main window."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ── Left panel (scrollable, max 250 px) ───────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(250)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        left_widget = QWidget()
        left_widget.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(2)

        self._build_file_section(left_layout)
        self._build_model_section(left_layout)
        self._build_inference_section(left_layout)
        self._build_export_section(left_layout)
        self._build_log_section(left_layout)
        left_layout.addStretch(1)

        left_scroll.setWidget(left_widget)

        # ── Right panel ───────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self._frame_viewer = FrameViewer()
        right_layout.addWidget(self._frame_viewer, stretch=1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        right_layout.addWidget(self._slider)

        self._lbl_frame_pos = QLabel("Frame 0 / 0")
        self._lbl_frame_pos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self._lbl_frame_pos)

        right_layout.addLayout(self._build_transport_bar())

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Left-panel sections ──────────────────────────────────────────────

    def _build_file_section(self, layout: QVBoxLayout):
        """Add File section widgets to *layout*."""
        layout.addWidget(_section_label("File"))

        self._btn_open_video = QPushButton("Open Video")
        layout.addWidget(self._btn_open_video)

        self._lbl_filename = QLabel("No file loaded")
        self._lbl_filename.setWordWrap(True)
        self._lbl_filename.setStyleSheet("color: #888888; font-size: 8pt;")
        layout.addWidget(self._lbl_filename)

        self._btn_set_output = QPushButton("Set Output Dir")
        layout.addWidget(self._btn_set_output)

        self._lbl_output_dir = QLabel("Not set")
        self._lbl_output_dir.setWordWrap(True)
        self._lbl_output_dir.setStyleSheet("color: #888888; font-size: 8pt;")
        layout.addWidget(self._lbl_output_dir)

    def _build_model_section(self, layout: QVBoxLayout):
        """Add Model section widgets to *layout*."""
        layout.addWidget(_section_label("Model"))

        self._combo_model = QComboBox()
        self._combo_model.addItems(
            ["yolo11n-seg.pt", "FastSAM-s.pt", "yolo11s-seg.pt"]
        )
        layout.addWidget(self._combo_model)

        self._btn_load_model = QPushButton("Load Model")
        layout.addWidget(self._btn_load_model)

        self._lbl_model_status = QLabel("Not Loaded")
        self._lbl_model_status.setStyleSheet("color: #CC4444;")
        layout.addWidget(self._lbl_model_status)

    def _build_inference_section(self, layout: QVBoxLayout):
        """Add Inference section widgets to *layout*."""
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

    def _build_export_section(self, layout: QVBoxLayout):
        """Add Export section widgets to *layout*."""
        layout.addWidget(_section_label("Export"))

        self._btn_save_yolo = QPushButton("Save Pseudo-Label (YOLO)")
        layout.addWidget(self._btn_save_yolo)

        self._btn_save_coco = QPushButton("Save Pseudo-Label (COCO)")
        layout.addWidget(self._btn_save_coco)

        self._btn_export_all = QPushButton("Export All Frames")
        layout.addWidget(self._btn_export_all)

    def _build_log_section(self, layout: QVBoxLayout):
        """Add Log section widgets to *layout*."""
        layout.addWidget(_section_label("Log"))

        self._log_list = QListWidget()
        self._log_list.setMaximumHeight(200)
        self._log_list.setStyleSheet("font-size: 8pt;")
        layout.addWidget(self._log_list)

    # ── Transport bar ────────────────────────────────────────────────────

    def _build_transport_bar(self) -> QHBoxLayout:
        """
        Build and return the video transport control button row.

        Returns
        -------
        QHBoxLayout
            Layout containing all transport buttons in order.
        """
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

    # ── Status bar ───────────────────────────────────────────────────────

    def _build_status_bar(self):
        """Create the three permanent status-bar sections separated by dividers."""
        sb: QStatusBar = self.statusBar()
        sb.setStyleSheet("QStatusBar { border-top: 1px solid #464649; }")

        def _sb_lbl(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet("padding: 0 8px;")
            return lbl

        def _vline() -> QFrame:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet("color: #464649;")
            return sep

        self._sb_device = _sb_lbl("Device: CPU")
        self._sb_vram   = _sb_lbl("VRAM: 0 MB")
        self._sb_fps    = _sb_lbl("FPS: --")
        self._sb_frame  = _sb_lbl("Frame: 0 / 0")

        for widget in (
            self._sb_device, _vline(),
            self._sb_vram,   _vline(),
            self._sb_fps,    _vline(),
            self._sb_frame,
        ):
            sb.addPermanentWidget(widget)

    # ══════════════════════════════════════════════════════════════════ #
    #  Signal wiring                                                      #
    # ══════════════════════════════════════════════════════════════════ #

    def _wire_signals(self):
        """Connect all widget signals to their corresponding slot methods."""
        self._btn_open_video.clicked.connect(self._on_open_video)
        self._btn_set_output.clicked.connect(self._on_set_output_dir)
        self._btn_load_model.clicked.connect(self._on_load_model)
        self._btn_run_inference.clicked.connect(self._on_run_inference)
        self._btn_save_yolo.clicked.connect(self._on_save_yolo)
        self._btn_save_coco.clicked.connect(self._on_save_coco)
        self._btn_export_all.clicked.connect(self._on_export_all)

        # Use valueChanged only — seek_to blocks signals when updating value
        self._slider.valueChanged.connect(self._on_slider_value_changed)

        self._btn_first.clicked.connect(
            lambda: self._seek_to(0)
        )
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

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — file handling                                              #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_open_video(self):
        """Open a file dialog, load the selected video, and display frame 0."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI);;All Files (*)",
        )
        if not path:
            return

        if self._is_playing:
            self._on_play_pause()

        ok = self._video.open(path)
        if not ok:
            QMessageBox.critical(
                self, "Error", f"Could not open video file:\n{path}"
            )
            self._log(f"ERROR: Could not open video — {path}")
            return

        total = self._video.total_frames()
        self._slider.setMaximum(max(0, total - 1))
        self._slider.setValue(0)
        self._current_frame_idx = 0
        self._current_result    = None

        basename = os.path.basename(path)
        self._lbl_filename.setText(basename)
        w, h = self._video.resolution()
        self._log(
            f"Opened: {basename}  "
            f"[{total} frames @ {self._video.fps():.1f} fps  {w}×{h}]"
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
                self,
                "Permission Error",
                f"Cannot write to:\n{path}\n\nPlease choose a different directory.",
            )
            self._log(f"ERROR: Output directory not writable — {path}")

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — model loading                                              #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_load_model(self):
        """Spawn ModelLoaderWorker to load the selected model in the background."""
        model_name = self._combo_model.currentText()
        self._btn_load_model.setEnabled(False)
        self._lbl_model_status.setText("Loading…")
        self._lbl_model_status.setStyleSheet("color: #DDAA00;")
        self._log(f"Loading model: {model_name} …")

        self._model_worker = ModelLoaderWorker(self._engine, model_name)
        self._model_worker.finished.connect(self._on_model_loaded)
        self._model_worker.start()

    def _on_model_loaded(self, success: bool, message: str):
        """
        Handle ``ModelLoaderWorker.finished``.

        Parameters
        ----------
        success : bool
            Whether the model loaded successfully.
        message : str
            Human-readable status or error message.
        """
        self._btn_load_model.setEnabled(True)
        if success:
            self._lbl_model_status.setText(message)
            self._lbl_model_status.setStyleSheet("color: #44CC44;")
            self._log(f"Model ready — {message}")
        else:
            self._lbl_model_status.setText("Load Failed")
            self._lbl_model_status.setStyleSheet("color: #CC4444;")
            self._log(f"ERROR: {message}")
        self._update_status_bar()

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — inference                                                  #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_run_inference(self):
        """Spawn an InferenceWorker for the current frame."""
        if not self._engine.is_loaded():
            self._log("Please load a model first.")
            return
        if self._current_frame_bgr is None:
            self._log("No frame available — open a video first.")
            return
        if self._infer_worker is not None and self._infer_worker.isRunning():
            return  # already running; do not stack requests

        self._btn_run_inference.setEnabled(False)

        self._infer_worker = InferenceWorker(
            self._engine,
            self._current_frame_bgr.copy(),
            self._spin_conf.value(),
            self._spin_iou.value(),
        )
        self._infer_worker.finished.connect(self._on_inference_done)
        self._infer_worker.error.connect(self._on_inference_error)
        self._infer_worker.start()

    def _on_inference_done(self, result: InferenceResult, annotated: np.ndarray):
        """
        Handle ``InferenceWorker.finished``: display overlay and log result.

        Parameters
        ----------
        result : InferenceResult
            Structured inference output.
        annotated : np.ndarray
            BGR frame with mask/box overlay already drawn.
        """
        self._current_result  = result
        self._last_infer_time = time.monotonic()
        self._btn_run_inference.setEnabled(True)
        self._show_bgr(annotated)
        n = len(result.class_ids)
        self._log(
            f"Inference: {n} detection(s) on frame {self._current_frame_idx}"
        )

    def _on_inference_error(self, msg: str):
        """
        Handle ``InferenceWorker.error``.

        Parameters
        ----------
        msg : str
            Error description.
        """
        self._btn_run_inference.setEnabled(True)
        self._log(f"ERROR during inference: {msg}")

    def _maybe_auto_infer(self):
        """
        Trigger auto-inference if the checkbox is enabled, a model is loaded,
        no worker is already running, and at least 200 ms have elapsed since
        the last inference (throttle guard to avoid GPU overload).
        """
        if not self._chk_auto_infer.isChecked():
            return
        if not self._engine.is_loaded():
            return
        if self._infer_worker is not None and self._infer_worker.isRunning():
            return
        if time.monotonic() - self._last_infer_time < 0.2:
            return
        self._on_run_inference()

    # ══════════════════════════════════════════════════════════════════ #
    #  Slots — export                                                     #
    # ══════════════════════════════════════════════════════════════════ #

    def _on_save_yolo(self):
        """Save the current frame and inference result as a YOLO annotation."""
        if not self._output_dir:
            self._log("Set an output directory first.")
            return
        if self._current_frame_bgr is None:
            self._log("No frame loaded — open a video first.")
            return
        if self._current_result is None:
            self._log("Run inference on this frame before saving.")
            return
        try:
            path = self._export.save_yolo(
                self._current_frame_bgr,
                self._current_result,
                self._current_frame_idx,
            )
            self._log(f"Saved YOLO label: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR saving YOLO: {exc}")

    def _on_save_coco(self):
        """Save the current frame and inference result as a COCO annotation."""
        if not self._output_dir:
            self._log("Set an output directory first.")
            return
        if self._current_frame_bgr is None:
            self._log("No frame loaded — open a video first.")
            return
        if self._current_result is None:
            self._log("Run inference on this frame before saving.")
            return
        try:
            video_name = self._lbl_filename.text()
            path = self._export.save_coco(
                self._current_frame_bgr,
                self._current_result,
                self._current_frame_idx,
                video_name,
            )
            self._log(f"Saved COCO: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"ERROR saving COCO: {exc}")

    def _on_export_all(self):
        """Open a QProgressDialog and spawn ExportAllWorker."""
        if not self._engine.is_loaded():
            self._log("Please load a model first.")
            return
        if self._video.total_frames() == 0:
            self._log("Open a video first.")
            return
        if not self._output_dir:
            self._log("Set an output directory first.")
            return

        total = self._video.total_frames()
        self._progress_dlg = QProgressDialog(
            "Exporting frames…", "Cancel", 0, total, self
        )
        self._progress_dlg.setWindowTitle("Export All Frames")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setValue(0)
        self._progress_dlg.show()

        self._export_worker = ExportAllWorker(
            self._export,
            self._video,
            self._engine,
            self._spin_conf.value(),
            self._spin_iou.value(),
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._progress_dlg.canceled.connect(self._export_worker.terminate)
        self._export_worker.start()
        self._log("Export started…")

    def _on_export_progress(self, current: int, total: int):
        """
        Update the export progress dialog.

        Parameters
        ----------
        current : int
            Number of frames processed so far.
        total : int
            Total frame count.
        """
        if hasattr(self, "_progress_dlg") and self._progress_dlg.isVisible():
            self._progress_dlg.setValue(current)

    def _on_export_finished(self, message: str):
        """
        Handle export completion.

        Parameters
        ----------
        message : str
            Completion message from the worker.
        """
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._log(f"Export complete — {message}")

    def _on_export_error(self, msg: str):
        """
        Handle export error.

        Parameters
        ----------
        msg : str
            Error description from the worker.
        """
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._log(f"ERROR during export: {msg}")
        QMessageBox.critical(self, "Export Error", msg)

    # ══════════════════════════════════════════════════════════════════ #
    #  Frame navigation & display                                         #
    # ══════════════════════════════════════════════════════════════════ #

    def _seek_to(self, idx: int):
        """
        Seek to frame *idx* (clamped to the valid range), update the slider,
        display the frame, and optionally trigger auto-inference.

        Parameters
        ----------
        idx : int
            Target frame index.  Clamped to ``[0, total_frames - 1]``.
        """
        total = self._video.total_frames()
        if total == 0:
            return
        idx = max(0, min(idx, total - 1))
        self._current_frame_idx = idx

        # Block signals to prevent re-entrant value-change callbacks
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)

        self._display_frame(idx)
        self._maybe_auto_infer()

    def _on_slider_value_changed(self, value: int):
        """
        Handle QSlider value changes (both user interaction and keyboard).

        Parameters
        ----------
        value : int
            New slider position.
        """
        if self._video.total_frames() == 0:
            return
        if value != self._current_frame_idx:
            self._seek_to(value)

    def _display_frame(self, idx: int):
        """
        Read frame *idx* from the VideoHandler and update the display.

        Also updates the rolling FPS window and the frame-position labels.

        Parameters
        ----------
        idx : int
            Zero-based frame index to display.
        """
        now = time.monotonic()
        if self._last_frame_time > 0.0:
            elapsed = now - self._last_frame_time
            if elapsed > 0.0:
                self._fps_window.append(1.0 / elapsed)
        self._last_frame_time = now

        frame = self._video.get_frame(idx)
        if frame is None:
            return

        self._current_frame_bgr = frame
        self._current_result    = None   # invalidate stale overlay
        self._show_bgr(frame)

        total = self._video.total_frames()
        self._lbl_frame_pos.setText(f"Frame {idx + 1} / {total}")
        self._sb_frame.setText(f"Frame: {idx + 1}/{total}")

    def _show_bgr(self, frame_bgr: np.ndarray):
        """
        Convert a BGR numpy frame to QPixmap and hand it to the FrameViewer.

        Parameters
        ----------
        frame_bgr : np.ndarray
            HxWx3 BGR uint8 array.
        """
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)
        self._frame_viewer.set_frame_pixmap(pixmap)

    def _advance_frame(self):
        """
        Called by the playback QTimer every ~33 ms.

        Advances one frame; stops playback automatically at the last frame.
        Auto-inference throttling is enforced inside ``_maybe_auto_infer``.
        """
        total = self._video.total_frames()
        if total == 0:
            return
        next_idx = self._current_frame_idx + 1
        if next_idx >= total:
            self._on_play_pause()   # stop at end
            return
        self._seek_to(next_idx)

    def _on_play_pause(self):
        """Toggle play / pause state of the playback timer."""
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
    #  Status bar                                                         #
    # ══════════════════════════════════════════════════════════════════ #

    def _update_status_bar(self):
        """Refresh all permanent status-bar labels with current runtime values."""
        # Device
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self._sb_device.setText(f"Device: CUDA ({name})")
            else:
                self._sb_device.setText("Device: CPU")
        except Exception:
            self._sb_device.setText("Device: CPU")

        # VRAM
        vram = self._engine.vram_used_mb()
        self._sb_vram.setText(f"VRAM: {vram:.0f} MB")

        # FPS (rolling average of last 10 display durations)
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
        Prepend a timestamped entry to the action log QListWidget.

        New entries are inserted at index 0 (top).  The list is trimmed to
        a maximum of 200 entries to prevent unbounded memory growth.

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
        """
        Stop timers and release the VideoHandler capture on window close.

        Parameters
        ----------
        event : QCloseEvent
            The close event (always accepted).
        """
        self._play_timer.stop()
        self._status_timer.stop()
        self._video.release()
        event.accept()

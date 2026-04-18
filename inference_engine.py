"""
inference_engine.py
───────────────────
GPU-optimised inference wrapper around Ultralytics YOLO / FastSAM models.

All inference runs at imgsz=640 with FP16 (half-precision) when a CUDA
device is present, keeping VRAM usage well within the 4 GB budget of an
RTX 4050 Laptop GPU.  A complete set of GPU memory safeguards is
implemented: CUDNN benchmark mode, warm-up pass, empty-cache after every
call, and an automatic CPU fallback on OutOfMemoryError.
"""

import dataclasses
import logging
import warnings

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — 20 visually distinct BGR colours for overlay rendering
# ---------------------------------------------------------------------------
_PALETTE: list[tuple[int, int, int]] = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InferenceResult:
    """
    Plain data container for a single-frame inference output.

    Attributes
    ----------
    boxes_xyxy : list[list[float]]
        Bounding boxes in absolute pixel coordinates ``[[x1,y1,x2,y2], ...]``.
    class_ids : list[int]
        Integer class index for each detection.
    confidences : list[float]
        Detection confidence score in ``[0, 1]`` for each detection.
    masks_binary : list[np.ndarray]
        List of HxW uint8 binary masks (pixel values 0 or 255).  May be empty
        when the model does not produce segmentation masks.
    class_names : list[str]
        Human-readable class name corresponding to each detection.
    orig_shape : tuple[int, int]
        ``(height, width)`` of the frame that was inferred on.
    """

    boxes_xyxy:   list[list[float]]
    class_ids:    list[int]
    confidences:  list[float]
    masks_binary: list[np.ndarray]
    class_names:  list[str]
    orig_shape:   tuple[int, int]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    GPU-optimized inference wrapper for Ultralytics YOLO / FastSAM models.

    Designed to stay within 4 GB VRAM on an NVIDIA RTX 4050 Laptop GPU by
    enforcing FP16 precision, a fixed imgsz of 640, CUDNN benchmark mode,
    a warm-up pass, and per-call empty-cache calls.
    """

    def __init__(self):
        """Initialise engine in an unloaded state."""
        self.model = None
        self._model_name: str = ""
        self._device: str = "cpu"
        self._loaded: bool = False

        # Optimisation 3: enable CUDNN auto-tuner once at startup
        if TORCH_AVAILABLE:
            torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ #

    def load_model(self, model_name: str) -> bool:
        """
        Load a YOLO / FastSAM weights file, move it to the best available
        device, enable FP16 on CUDA, and perform a warm-up forward pass.

        Parameters
        ----------
        model_name : str
            Filename of the model weights, e.g. ``"yolo11n-seg.pt"``.
            Ultralytics will attempt an automatic download if the file is
            not found locally.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on ``FileNotFoundError`` or a
            GPU out-of-memory error during loading.
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.error("ultralytics package is not installed.")
            return False

        # Unload any previous model first
        self.model = None
        self._loaded = False

        try:
            self.model = YOLO(model_name)
        except FileNotFoundError:
            logger.error(
                "Model file not found: %s.  "
                "Download from ultralytics or place in working directory.",
                model_name,
            )
            return False
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            return False

        self._model_name = model_name

        # Optimisation 1 + 2: move to CUDA and enable FP16
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                self.model = self.model.to("cuda")
                self.model.model.half()  # FP16 — halves VRAM usage
                self._device = "cuda"
            except Exception as exc:
                logger.warning(
                    "Could not move model to CUDA (%s) — falling back to CPU.", exc
                )
                self._device = "cpu"
        else:
            self._device = "cpu"

        # Optimisation 4: warm-up pass to pre-allocate CUDA buffers
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model(
                    dummy,
                    imgsz=640,
                    half=(self._device == "cuda"),
                    verbose=False,
                )
            if TORCH_AVAILABLE and self._device == "cuda":
                torch.cuda.empty_cache()  # Optimisation 5
        except Exception as exc:
            logger.warning("Warm-up forward pass failed (non-fatal): %s", exc)

        self._loaded = True
        return True

    # ------------------------------------------------------------------ #

    def infer(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.35,
        iou: float = 0.45,
    ) -> "InferenceResult":
        """
        Run segmentation inference on a single BGR frame.

        The frame is letterboxed to 640×640 internally by Ultralytics; the
        returned masks and boxes are up-scaled back to the original resolution.

        Parameters
        ----------
        frame_bgr : np.ndarray
            HxWx3 BGR uint8 numpy array (any resolution).
        conf : float
            Minimum confidence threshold in ``[0, 1]``.
        iou : float
            NMS IoU threshold in ``[0, 1]``.

        Returns
        -------
        InferenceResult
            Extracted bounding boxes, masks, class IDs, confidences, and names.

        Raises
        ------
        RuntimeError
            If the model is not loaded, or if OOM persists after CPU fallback.
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("No model loaded — call load_model() first.")

        # Optimisation 6: OOM guard with automatic CPU fallback
        try:
            return self._run_inference(frame_bgr, conf, iou)
        except Exception as exc:
            is_oom = (
                TORCH_AVAILABLE
                and isinstance(exc, torch.cuda.OutOfMemoryError)
            )
            if is_oom:
                logger.warning(
                    "CUDA OutOfMemory on inference — falling back to CPU."
                )
                torch.cuda.empty_cache()
                try:
                    self.model = self.model.to("cpu")
                    self.model.model.float()
                    self._device = "cpu"
                    return self._run_inference(frame_bgr, conf, iou)
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"Inference failed on CPU after OOM fallback: {retry_exc}"
                    ) from retry_exc
            raise

    # ------------------------------------------------------------------ #

    def _run_inference(
        self,
        frame_bgr: np.ndarray,
        conf: float,
        iou: float,
    ) -> "InferenceResult":
        """
        Internal helper: call the Ultralytics model and unpack the result.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input BGR frame.
        conf : float
            Confidence threshold.
        iou : float
            IoU threshold.

        Returns
        -------
        InferenceResult
            Populated inference result.
        """
        use_half = self._device == "cuda"
        # Optimisation 2: fixed imgsz=640 — never process at full GoPro res
        results = self.model(
            frame_bgr,
            conf=conf,
            iou=iou,
            imgsz=640,
            half=use_half,
            verbose=False,
        )

        result = results[0]
        orig_h, orig_w = result.orig_shape

        boxes_xyxy:   list[list[float]] = []
        class_ids:    list[int]         = []
        confidences:  list[float]       = []
        masks_binary: list[np.ndarray]  = []
        class_names:  list[str]         = []

        names_map: dict[int, str] = result.names if result.names else {}

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                boxes_xyxy.append([float(v) for v in xyxy])
                cid = int(box.cls[0].item())
                class_ids.append(cid)
                confidences.append(float(box.conf[0].item()))
                class_names.append(names_map.get(cid, str(cid)))

        if result.masks is not None and len(result.masks) > 0:
            for mask_tensor in result.masks.data:
                # mask_tensor: float32 Tensor of shape (H', W')
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
                binary = (mask_resized > 0.5).astype(np.uint8) * 255
                masks_binary.append(binary)

        # Optimisation 5: release CUDA cache after every inference call
        if TORCH_AVAILABLE and self._device == "cuda":
            torch.cuda.empty_cache()

        return InferenceResult(
            boxes_xyxy=boxes_xyxy,
            class_ids=class_ids,
            confidences=confidences,
            masks_binary=masks_binary,
            class_names=class_names,
            orig_shape=(orig_h, orig_w),
        )

    # ------------------------------------------------------------------ #

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        result: "InferenceResult",
    ) -> np.ndarray:
        """
        Draw semi-transparent segmentation masks and labelled bounding boxes
        onto a copy of *frame_bgr*.

        Masks are alpha-blended at 0.45 opacity using a fixed 20-colour
        palette (cycled by class ID).  Bounding boxes are drawn at 2 px
        thickness with a filled label chip above each box.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Original BGR frame (not modified).
        result : InferenceResult
            Inference output to visualise.

        Returns
        -------
        np.ndarray
            Annotated BGR frame (a new array; the original is untouched).
        """
        annotated = frame_bgr.copy().astype(np.float32)

        # Pass 1: alpha-blend masks (only within mask region)
        for i, mask in enumerate(result.masks_binary):
            if i >= len(result.class_ids):
                break
            color = _PALETTE[result.class_ids[i] % len(_PALETTE)]
            color_array = np.array(color, dtype=np.float32)
            mask_bool = mask > 0
            # Blend: out = src * (1 - alpha) + color * alpha
            annotated[mask_bool] = (
                annotated[mask_bool] * 0.55 + color_array * 0.45
            )

        annotated = annotated.clip(0, 255).astype(np.uint8)

        # Pass 2: draw bounding boxes and labels on top
        for i, box in enumerate(result.boxes_xyxy):
            x1, y1, x2, y2 = (int(v) for v in box)
            color = _PALETTE[result.class_ids[i] % len(_PALETTE)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{result.class_names[i]} {result.confidences[i]:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            chip_top    = max(y1 - th - baseline - 4, 0)
            chip_bottom = chip_top + th + baseline + 4
            cv2.rectangle(
                annotated,
                (x1, chip_top),
                (x1 + tw + 4, chip_bottom),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, chip_bottom - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    # ------------------------------------------------------------------ #

    def vram_used_mb(self) -> float:
        """
        Return the current CUDA memory allocated by PyTorch in megabytes.

        Optimisation 7: used by the status-bar timer to display live VRAM
        consumption.

        Returns
        -------
        float
            Allocated VRAM in MB, or ``0.0`` when not running on CUDA or
            when PyTorch is unavailable.
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1e6

    # ------------------------------------------------------------------ #

    def is_loaded(self) -> bool:
        """Return ``True`` if a model has been loaded successfully."""
        return self._loaded

    def device(self) -> str:
        """Return the current inference device string (``'cuda'`` or ``'cpu'``)."""
        return self._device

    def model_name(self) -> str:
        """Return the filename of the currently loaded model weights."""
        return self._model_name

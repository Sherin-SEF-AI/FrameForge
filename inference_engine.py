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

import contextlib
import dataclasses
import logging
import os
import warnings

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sam2.build_sam import build_sam2_video_predictor as _build_sam2_video_predictor
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False

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
        imgsz: int = 640,
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
            return self._run_inference(frame_bgr, conf, iou, imgsz)
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
                    return self._run_inference(frame_bgr, conf, iou, imgsz)
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
        imgsz: int = 640,
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
        results = self.model(
            frame_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
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
                mask_np = mask_tensor.cpu().numpy()
                # Skip resize if already at target resolution
                if mask_np.shape == (orig_h, orig_w):
                    mask_resized = mask_np
                else:
                    mask_resized = cv2.resize(
                        mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                binary = (mask_resized > 0.5).astype(np.uint8) * 255
                masks_binary.append(binary)

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
        annotated = frame_bgr.copy()

        # Pass 1: alpha-blend masks — stay in uint8 to avoid float32 copy
        for i, mask in enumerate(result.masks_binary):
            if i >= len(result.class_ids):
                break
            color     = _PALETTE[result.class_ids[i] % len(_PALETTE)]
            mask_bool = mask > 0
            if not mask_bool.any():
                continue
            region               = annotated[mask_bool]
            color_u16            = np.array(color, dtype=np.uint16)
            annotated[mask_bool] = (
                (region.astype(np.uint16) * 55 + color_u16 * 45) // 100
            ).astype(np.uint8)

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


# ═══════════════════════════════════════════════════════════════════════════ #
#  SemanticResult dataclass                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclasses.dataclass
class SemanticResult:
    """
    Pixel-level semantic segmentation output.

    Attributes
    ----------
    label_map : np.ndarray
        H × W uint8 array of class IDs (0-based).
    colormap : np.ndarray
        H × W × 3 BGR array — colourised visualisation of *label_map*.
    class_names : list[str]
        Human-readable name for each label ID.
    orig_shape : tuple[int, int]
        ``(height, width)`` of the source frame.
    """

    label_map:   np.ndarray
    colormap:    np.ndarray
    class_names: list[str]
    orig_shape:  tuple[int, int]


# ═══════════════════════════════════════════════════════════════════════════ #
#  Optional-dependency guards                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

_GROUNDING_DINO_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import PIL.Image as _PIL_Image
    _GROUNDING_DINO_AVAILABLE = True
except ImportError:
    pass

_SAM_AVAILABLE = False
try:
    from segment_anything import SamPredictor, sam_model_registry
    _SAM_AVAILABLE = True
except ImportError:
    pass

_MOBILE_SAM_AVAILABLE = False
try:
    from mobile_sam import build_sam_vit_t
    from mobile_sam import SamPredictor as _MobileSamPredictor
    _MOBILE_SAM_AVAILABLE = True
except ImportError:
    pass

_SEGFORMER_AVAILABLE = False
try:
    from transformers import (
        AutoImageProcessor,
        SegformerForSemanticSegmentation,
    )
    _SEGFORMER_AVAILABLE = True
except ImportError:
    pass


def _torch_no_grad():
    """Return a no-grad context manager, or a null one if torch is absent."""
    if TORCH_AVAILABLE:
        return torch.no_grad()
    return contextlib.nullcontext()


# ═══════════════════════════════════════════════════════════════════════════ #
#  GroundedSAMEngine                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class GroundedSAMEngine:
    """
    Zero-shot text-prompted instance segmentation.

    Grounding DINO converts a comma/period-separated text prompt into
    bounding boxes.  SAM then generates precise binary masks for each box.

    Both components are optional:
    - Without SAM: returns boxes and class labels but no masks.
    - Without transformers: the engine reports itself as unavailable.

    Dependencies
    ------------
    ``pip install transformers>=4.38 timm pillow``
    ``pip install segment-anything``   # optional, for masks
    """

    DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

    # SAM checkpoint filenames and download URLs
    SAM_CHECKPOINTS: dict[str, tuple[str, str]] = {
        "vit_b": (
            "sam_vit_b_01ec64.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        ),
        "vit_l": (
            "sam_vit_l_0b3195.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        ),
        "vit_h": (
            "sam_vit_h_4b8939.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        ),
    }

    def __init__(self):
        self._dino_processor = None
        self._dino_model     = None
        self._sam_predictor  = None
        self._device         = "cpu"
        self._loaded         = False
        self._status         = "Not loaded"
        # Cache: avoid re-encoding the same frame in SAM when thresholds change
        self._sam_frame_fingerprint: tuple | None = None

    # ── Class-level queries ──────────────────────────────────────────────

    @staticmethod
    def dino_available() -> bool:
        """Return True when Grounding DINO dependencies are installed."""
        return _GROUNDING_DINO_AVAILABLE

    @staticmethod
    def sam_available() -> bool:
        """Return True when segment-anything is installed."""
        return _SAM_AVAILABLE

    # ── Loading ──────────────────────────────────────────────────────────

    def load(
        self,
        sam_checkpoint: str = "",
        sam_variant:    str = "vit_b",
    ) -> tuple[bool, str]:
        """
        Load Grounding DINO and (optionally) a SAM predictor.

        Parameters
        ----------
        sam_checkpoint : str
            Path to a SAM ``.pth`` checkpoint.  Empty string skips SAM.
        sam_variant : str
            SAM model variant: ``"vit_b"`` / ``"vit_l"`` / ``"vit_h"``.

        Returns
        -------
        (success, message) : tuple[bool, str]
        """
        if not _GROUNDING_DINO_AVAILABLE:
            msg = (
                "Grounding DINO not installed.\n"
                "Run: pip install transformers>=4.38 timm pillow"
            )
            self._status = msg
            return False, msg

        try:
            dev = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
            self._device = dev

            logger.info("Loading Grounding DINO (%s) on %s …", self.DINO_MODEL_ID, dev)
            self._dino_processor = AutoProcessor.from_pretrained(self.DINO_MODEL_ID)
            self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.DINO_MODEL_ID
            ).to(dev)
            self._dino_model.eval()

            # SAM (optional)
            if sam_checkpoint and _SAM_AVAILABLE:
                logger.info("Loading SAM %s from %s …", sam_variant, sam_checkpoint)
                sam = sam_model_registry[sam_variant](checkpoint=sam_checkpoint)
                sam.to(dev)
                self._sam_predictor = SamPredictor(sam)
            else:
                self._sam_predictor = None

            self._loaded = True
            sam_note = (
                f" + SAM-{sam_variant}" if self._sam_predictor else " (boxes only — no SAM)"
            )
            self._status = f"Grounded SAM on {dev.upper()}{sam_note}"
            return True, self._status

        except Exception as exc:
            self._loaded = False
            self._status = str(exc)
            logger.error("GroundedSAMEngine.load failed: %s", exc)
            return False, str(exc)

    def is_loaded(self) -> bool:
        return self._loaded

    def status(self) -> str:
        return self._status

    # ── Inference ────────────────────────────────────────────────────────

    def infer(
        self,
        frame_bgr:      np.ndarray,
        text_prompt:    str,
        box_threshold:  float = 0.35,
        text_threshold: float = 0.25,
    ) -> "InferenceResult":
        """
        Run Grounded SAM on *frame_bgr* and return an ``InferenceResult``.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Source frame in BGR format.
        text_prompt : str
            Comma- or period-separated class names, e.g.
            ``"pedestrian, car, pothole"`` or the full taxonomy prompt.
        box_threshold : float
            DINO box confidence threshold.
        text_threshold : float
            DINO text similarity threshold.
        """
        if not self._loaded or self._dino_model is None:
            raise RuntimeError("GroundedSAMEngine not loaded — call load() first.")

        from taxonomy import id_for

        orig_h, orig_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = _PIL_Image.fromarray(frame_rgb)

        prompt = self._format_prompt(text_prompt)

        inputs = self._dino_processor(
            images=pil_image, text=prompt, return_tensors="pt"
        ).to(self._device)

        with _torch_no_grad():
            outputs = self._dino_model(**inputs)

        detections = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(orig_h, orig_w)],
        )[0]

        boxes_np   = detections["boxes"].cpu().numpy()    # (N, 4) xyxy
        scores_np  = detections["scores"].cpu().numpy()   # (N,)
        raw_labels = detections["labels"]                 # list[str]

        boxes_xyxy:   list[list[float]] = []
        class_ids:    list[int]         = []
        confidences:  list[float]       = []
        masks_binary: list[np.ndarray]  = []
        class_names:  list[str]         = []

        if self._sam_predictor is not None and len(boxes_np) > 0:
            # Only re-encode if the frame has changed (fingerprint: shape + 2 pixel samples)
            mid_r = orig_h // 2
            mid_c = orig_w // 2
            fp = (frame_rgb.shape,
                  int(frame_rgb[0, 0, 0]),
                  int(frame_rgb[mid_r, mid_c, 0]))
            if fp != self._sam_frame_fingerprint:
                self._sam_predictor.set_image(frame_rgb)
                self._sam_frame_fingerprint = fp

        for box, score, raw_label in zip(boxes_np, scores_np, raw_labels):
            x1, y1, x2, y2 = box.tolist()
            clean = raw_label.strip().lower()
            cid   = id_for(clean)
            if cid < 0:
                cid = len(class_ids)    # assign a new temporary ID

            boxes_xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            class_ids.append(cid)
            confidences.append(float(score))
            class_names.append(clean)

            # SAM mask for this box
            if self._sam_predictor is not None:
                try:
                    sam_masks, _, _ = self._sam_predictor.predict(
                        box=np.array([x1, y1, x2, y2]),
                        multimask_output=False,
                    )
                    masks_binary.append((sam_masks[0] * 255).astype(np.uint8))
                except Exception as exc:
                    logger.warning("SAM predict failed: %s", exc)
                    masks_binary.append(np.zeros((orig_h, orig_w), dtype=np.uint8))

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return InferenceResult(
            boxes_xyxy=boxes_xyxy,
            class_ids=class_ids,
            confidences=confidences,
            masks_binary=masks_binary,
            class_names=class_names,
            orig_shape=(orig_h, orig_w),
        )

    def draw_overlay(self, frame_bgr: np.ndarray, result: "InferenceResult") -> np.ndarray:
        """Render overlay using the standard InferenceEngine palette."""
        return InferenceEngine().draw_overlay(frame_bgr, result)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        """
        Normalise a user-supplied prompt to Grounding DINO format.

        Input  : ``"pedestrian, car, pothole"``
        Output : ``"pedestrian . car . pothole ."``
        """
        # Split on commas or periods, strip whitespace, replace underscores
        parts = [
            p.strip().replace("_", " ")
            for p in prompt.replace(",", ".").split(".")
            if p.strip()
        ]
        return " . ".join(parts) + " ."


# ═══════════════════════════════════════════════════════════════════════════ #
#  SAMBoxRefiner                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class SAMBoxRefiner:
    """
    Lightweight mask refiner: takes any InferenceResult with bounding boxes
    and uses SAM or MobileSAM to generate tight pixel-accurate masks.

    Supports two backends
    ---------------------
    - Original SAM  (segment_anything): vit_b / vit_l / vit_h
    - MobileSAM     (mobile_sam):       TinyViT — ~10× faster, similar quality

    Typical workflow
    ----------------
    1. Run YOLO to get bounding boxes (fast).
    2. Call ``refine(frame_bgr, yolo_result)`` to replace rectangular masks
       with tight SAM masks in one call.
    """

    MOBILE_SAM_URL  = "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt"
    MOBILE_SAM_CKPT = "mobile_sam.pt"

    def __init__(self):
        self._predictor  = None
        self._device     = "cpu"
        self._loaded     = False
        self._is_mobile  = False
        self._status     = "Not loaded"

    @staticmethod
    def sam_available() -> bool:
        """True when segment-anything is installed."""
        return _SAM_AVAILABLE

    @staticmethod
    def mobile_sam_available() -> bool:
        """True when mobile-sam is installed."""
        return _MOBILE_SAM_AVAILABLE

    def is_loaded(self) -> bool:
        return self._loaded

    def status(self) -> str:
        return self._status

    def load(
        self,
        checkpoint: str = "",
        variant:    str = "vit_b",
        use_mobile: bool = False,
    ) -> tuple[bool, str]:
        """
        Load the SAM or MobileSAM predictor.

        Parameters
        ----------
        checkpoint : str
            Path to a SAM .pth checkpoint.  For MobileSAM, auto-downloaded
            if left empty.
        variant : str
            SAM variant (``'vit_b'``/``'vit_l'``/``'vit_h'``).
            Ignored when *use_mobile* is True.
        use_mobile : bool
            When True, load MobileSAM (TinyViT) instead of original SAM.
        """
        dev = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self._device = dev
        try:
            if use_mobile:
                if not _MOBILE_SAM_AVAILABLE:
                    return False, "MobileSAM not installed.\nRun: pip install mobile-sam"
                ckpt = checkpoint or self.MOBILE_SAM_CKPT
                if not os.path.exists(ckpt):
                    import urllib.request
                    logger.info("Downloading MobileSAM checkpoint → %s …", ckpt)
                    urllib.request.urlretrieve(self.MOBILE_SAM_URL, ckpt)
                sam = build_sam_vit_t(checkpoint=ckpt)
                sam.to(dev)
                self._predictor = _MobileSamPredictor(sam)
                self._is_mobile = True
            else:
                if not _SAM_AVAILABLE:
                    return False, "SAM not installed.\nRun: pip install segment-anything"
                if not checkpoint:
                    return False, "SAM checkpoint path required (e.g. sam_vit_b_01ec64.pth)."
                if not os.path.exists(checkpoint):
                    return False, f"Checkpoint not found: {checkpoint}"
                sam = sam_model_registry[variant](checkpoint=checkpoint)
                sam.to(dev)
                self._predictor = SamPredictor(sam)
                self._is_mobile = False

            self._loaded = True
            name = "MobileSAM" if self._is_mobile else f"SAM-{variant}"
            self._status = f"{name} on {dev.upper()}"
            return True, self._status

        except Exception as exc:
            self._loaded = False
            self._status = str(exc)
            logger.error("SAMBoxRefiner.load failed: %s", exc)
            return False, str(exc)

    def refine(
        self,
        frame_bgr: np.ndarray,
        result:    "InferenceResult",
    ) -> "InferenceResult":
        """
        Generate a tight SAM mask for every bounding box in *result*.

        Returns a new ``InferenceResult`` with ``masks_binary`` replaced by
        SAM predictions.  All other fields (boxes, class IDs, confidences,
        names) are preserved unchanged.

        Falls back to a filled rectangular mask if SAM fails on a specific box.
        """
        if not self._loaded or self._predictor is None:
            raise RuntimeError("SAMBoxRefiner not loaded — call load() first.")
        if not result.boxes_xyxy:
            return result

        orig_h, orig_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(frame_rgb)

        new_masks: list[np.ndarray] = []
        for box in result.boxes_xyxy:
            x1, y1, x2, y2 = [float(v) for v in box]
            try:
                masks, _, _ = self._predictor.predict(
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    multimask_output=False,
                )
                new_masks.append((masks[0] * 255).astype(np.uint8))
            except Exception as exc:
                logger.warning("SAM predict failed for box %s: %s", box, exc)
                fallback = np.zeros((orig_h, orig_w), dtype=np.uint8)
                fallback[int(y1):int(y2), int(x1):int(x2)] = 255
                new_masks.append(fallback)

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return InferenceResult(
            boxes_xyxy=list(result.boxes_xyxy),
            class_ids=list(result.class_ids),
            confidences=list(result.confidences),
            masks_binary=new_masks,
            class_names=list(result.class_names),
            orig_shape=result.orig_shape,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  SemanticSegmentationEngine                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class SemanticSegmentationEngine:
    """
    Full-scene pixel-level semantic segmentation using SegFormer.

    Default model: ``nvidia/segformer-b2-finetuned-cityscapes-1024-1024``
    (19 Cityscapes classes, downloaded automatically via HuggingFace Hub on
    first use).

    Dependencies
    ------------
    ``pip install transformers>=4.38 pillow``
    """

    DEFAULT_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

    # Cityscapes BGR palette (19 classes)
    CITYSCAPES_BGR: list[tuple[int, int, int]] = [
        (128,  64, 128),   # road
        (232,  35, 244),   # sidewalk
        ( 70,  70,  70),   # building
        (156, 102, 102),   # wall
        (153, 153, 190),   # fence
        (153, 153, 153),   # pole
        ( 30, 170, 250),   # traffic light
        (  0, 220, 220),   # traffic sign
        ( 35, 142, 107),   # vegetation
        (152, 251, 152),   # terrain
        (180, 130,  70),   # sky
        ( 60,  20, 220),   # person
        (  0,   0, 255),   # rider
        (142,   0,   0),   # car
        ( 70,   0,   0),   # truck
        (100,  60,   0),   # bus
        (100,  80,   0),   # train
        (230,   0,   0),   # motorcycle
        ( 32,  11, 119),   # bicycle
    ]

    def __init__(self):
        self._processor   = None
        self._model       = None
        self._loaded      = False
        self._device      = "cpu"
        self._class_names: list[str] = []
        self._status      = "Not loaded"

    @staticmethod
    def is_available() -> bool:
        """Return True when SegFormer dependencies are installed."""
        return _SEGFORMER_AVAILABLE

    # ── Loading ──────────────────────────────────────────────────────────

    def load(self, model_id: str = DEFAULT_MODEL) -> tuple[bool, str]:
        """
        Download (if needed) and load the SegFormer model.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier.
        """
        if not _SEGFORMER_AVAILABLE:
            msg = "SegFormer not installed.\nRun: pip install transformers>=4.38 pillow"
            self._status = msg
            return False, msg

        try:
            dev = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
            self._device = dev

            logger.info("Loading SegFormer (%s) on %s …", model_id, dev)
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                model_id
            ).to(dev)
            self._model.eval()
            # FP16 on GPU — halves VRAM and roughly doubles throughput
            if dev == "cuda":
                self._model = self._model.half()
            self._fp16 = (dev == "cuda")

            # Extract class names from model config
            cfg = self._model.config
            if hasattr(cfg, "id2label"):
                n = len(cfg.id2label)
                self._class_names = [cfg.id2label[i] for i in range(n)]
            else:
                self._class_names = [f"class_{i}" for i in range(256)]

            self._loaded = True
            self._status = f"SegFormer on {dev.upper()} ({len(self._class_names)} classes)"
            return True, self._status

        except Exception as exc:
            self._loaded = False
            self._status = str(exc)
            logger.error("SemanticSegmentationEngine.load failed: %s", exc)
            return False, str(exc)

    def is_loaded(self) -> bool:
        return self._loaded

    def status(self) -> str:
        return self._status

    # ── Inference ────────────────────────────────────────────────────────

    def infer(self, frame_bgr: np.ndarray) -> SemanticResult:
        """
        Run semantic segmentation on *frame_bgr*.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Source BGR frame (any resolution).

        Returns
        -------
        SemanticResult
            Pixel-level class IDs and a colourised visualisation.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "SemanticSegmentationEngine not loaded — call load() first."
            )

        import torch.nn.functional as F

        orig_h, orig_w = frame_bgr.shape[:2]

        # Pre-resize to SegFormer's native resolution with OpenCV (faster than PIL resize)
        _SEGFORMER_INPUT_SIZE = 512
        if orig_h > _SEGFORMER_INPUT_SIZE or orig_w > _SEGFORMER_INPUT_SIZE:
            scale = _SEGFORMER_INPUT_SIZE / max(orig_h, orig_w)
            small = cv2.resize(
                frame_bgr,
                (int(orig_w * scale), int(orig_h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small = frame_bgr

        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil_image = _PIL_Image.fromarray(frame_rgb)

        inputs = self._processor(images=pil_image, return_tensors="pt")
        # Cast pixel_values to FP16 when model is in half precision
        if getattr(self, "_fp16", False):
            inputs = {
                k: v.to(self._device, dtype=torch.float16) if v.is_floating_point() else v.to(self._device)
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with _torch_no_grad():
            outputs = self._model(**inputs)

        # Upsample logits back to original frame resolution
        logits = outputs.logits.float()   # (1, C, H/4, W/4) — back to fp32 for interpolate
        upsampled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        label_map = (
            upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        )

        # Vectorized BGR colormap via numpy LUT — much faster than a Python loop
        palette  = self.CITYSCAPES_BGR
        lut      = np.zeros((256, 3), dtype=np.uint8)
        for i, c in enumerate(palette):
            lut[i] = c
        colormap = lut[label_map]   # shape (H, W, 3)

        return SemanticResult(
            label_map=label_map,
            colormap=colormap,
            class_names=self._class_names,
            orig_shape=(orig_h, orig_w),
        )


# ---------------------------------------------------------------------------
# SAM 2 Temporal Propagation Engine
# ---------------------------------------------------------------------------

class SAM2PropagationEngine:
    """
    Wraps Meta's SAM 2 video predictor for temporal mask propagation.

    Workflow
    --------
    1. ``load(checkpoint, variant)``  — load model weights once.
    2. ``propagate(...)``             — extract frames to a temp dir, initialise
       the video state with keyframe masks/boxes, propagate forward or backward,
       return a dict mapping global_frame_idx → InferenceResult.

    The caller owns the temp-dir lifetime; this class always cleans it up in a
    finally block inside ``propagate``.
    """

    # Mapping from short variant name to Hydra config filename
    CONFIGS = {
        "tiny":   "sam2_hiera_tiny.yaml",
        "small":  "sam2_hiera_small.yaml",
        "base+":  "sam2_hiera_base_plus.yaml",
        "large":  "sam2_hiera_large.yaml",
    }
    # Default checkpoint filenames for auto-download hint in UI
    CHECKPOINTS = {
        "tiny":   "sam2_hiera_tiny.pt",
        "small":  "sam2_hiera_small.pt",
        "base+":  "sam2_hiera_base_plus.pt",
        "large":  "sam2_hiera_large.pt",
    }

    def __init__(self):
        self._predictor = None
        self._device    = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._predictor is not None

    def status(self) -> str:
        if self._predictor is None:
            return "Not loaded"
        return f"Ready ({self._device.upper()})"

    def load(self, checkpoint: str, variant: str = "tiny") -> "tuple[bool, str]":
        """Load SAM 2 video predictor from *checkpoint*."""
        if not _SAM2_AVAILABLE:
            return False, "sam2 not installed.  Run: pip install sam2"
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available."
        cfg = self.CONFIGS.get(variant, "sam2_hiera_tiny.yaml")
        try:
            self._predictor = _build_sam2_video_predictor(
                cfg, checkpoint, device=self._device
            )
            return True, f"SAM 2 ({variant}) loaded on {self._device.upper()}"
        except Exception as exc:
            self._predictor = None
            return False, str(exc)

    # ------------------------------------------------------------------

    def propagate(
        self,
        video_path:       str,
        keyframe_idx:     int,
        keyframe_result:  "InferenceResult",
        start_idx:        int,
        end_idx:          int,
        reverse:          bool = False,
        progress_cb:      "callable | None" = None,
        cancel_flag:      "list[bool] | None" = None,
    ) -> "dict[int, InferenceResult]":
        """
        Propagate *keyframe_result* masks through frames [start_idx, end_idx].

        Parameters
        ----------
        video_path      : absolute path to the source video file.
        keyframe_idx    : global frame index of the annotated keyframe.
        keyframe_result : InferenceResult holding the masks to propagate from.
        start_idx       : first global frame index to include (inclusive).
        end_idx         : last global frame index to include (inclusive).
        reverse         : if True, propagate from keyframe *backward* to start_idx.
        progress_cb     : optional callable(done: int, total: int).
        cancel_flag     : mutable list[bool]; propagation stops if flag[0] is True.

        Returns
        -------
        dict mapping global_frame_idx → InferenceResult for every propagated frame.
        """
        import shutil
        import tempfile

        if not self.is_loaded():
            raise RuntimeError("SAM 2 predictor not loaded.")
        if not keyframe_result.masks_binary and not keyframe_result.boxes_xyxy:
            raise ValueError("Keyframe has no masks or boxes to propagate from.")

        frame_indices = list(range(start_idx, end_idx + 1))
        if keyframe_idx not in frame_indices:
            raise ValueError(
                f"keyframe_idx {keyframe_idx} is outside range "
                f"[{start_idx}, {end_idx}]."
            )
        keyframe_local = frame_indices.index(keyframe_idx)

        tmp_dir = tempfile.mkdtemp(prefix="ff_sam2_")
        try:
            # ── Step 1: extract frames to numbered JPEGs ──────────────────
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            try:
                for local_i, global_i in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(global_i))
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cv2.imwrite(
                            os.path.join(tmp_dir, f"{local_i:05d}.jpg"), frame
                        )
            finally:
                cap.release()

            orig_h, orig_w = keyframe_result.orig_shape

            # ── Step 2: initialise SAM 2 video state ─────────────────────
            with torch.inference_mode():
                state = self._predictor.init_state(video_path=tmp_dir)
                self._predictor.reset_state(state)

                # Add one SAM 2 object per detection in the keyframe
                n_objects = len(keyframe_result.class_ids)
                for obj_id in range(n_objects):
                    if obj_id < len(keyframe_result.masks_binary):
                        # Prefer mask prompt (pixel-accurate)
                        m = keyframe_result.masks_binary[obj_id]
                        bool_mask = (m > 0)
                        self._predictor.add_new_mask(
                            state,
                            frame_idx=keyframe_local,
                            obj_id=obj_id,
                            mask=bool_mask,
                        )
                    elif obj_id < len(keyframe_result.boxes_xyxy):
                        # Fall back to box prompt
                        box = np.array(
                            keyframe_result.boxes_xyxy[obj_id], dtype=np.float32
                        )
                        self._predictor.add_new_points_or_box(
                            state,
                            frame_idx=keyframe_local,
                            obj_id=obj_id,
                            box=box,
                        )

                # ── Step 3: propagate ─────────────────────────────────────
                results: "dict[int, InferenceResult]" = {}
                total = len(frame_indices)

                for local_idx, obj_ids, mask_logits in \
                        self._predictor.propagate_in_video(state, reverse=reverse):

                    if cancel_flag and cancel_flag[0]:
                        break

                    global_idx = frame_indices[local_idx]

                    boxes, class_ids, confs, masks, cnames = [], [], [], [], []
                    for slot, obj_id in enumerate(obj_ids):
                        if obj_id >= n_objects:
                            continue
                        # mask_logits shape: (N_objects, 1, H, W)
                        m = (mask_logits[slot, 0] > 0.0).cpu().numpy()
                        m_u8 = m.astype(np.uint8) * 255
                        ys, xs = np.where(m)
                        if len(xs) == 0:
                            continue
                        x1, y1 = int(xs.min()), int(ys.min())
                        x2, y2 = int(xs.max()), int(ys.max())
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        class_ids.append(keyframe_result.class_ids[obj_id])
                        confs.append(keyframe_result.confidences[obj_id])
                        masks.append(m_u8)
                        cnames.append(keyframe_result.class_names[obj_id])

                    results[global_idx] = InferenceResult(
                        boxes_xyxy   = boxes,
                        class_ids    = class_ids,
                        confidences  = confs,
                        masks_binary = masks,
                        class_names  = cnames,
                        orig_shape   = (orig_h, orig_w),
                    )

                    if progress_cb:
                        progress_cb(local_idx + 1, total)

            return results

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

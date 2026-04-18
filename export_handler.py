"""
export_handler.py
─────────────────
Saves video frames and their inference annotations to disk in YOLO
segmentation format and standard COCO JSON format.

Output directory layout::

    <output_dir>/
        images/            ← raw JPEG frames  (frame_000042.jpg)
        labels/            ← YOLO .txt files  (frame_000042.txt)
        coco/
            annotations.json  ← COCO annotation file (accumulated across saves)

All export methods are designed to be called from a QThread worker; no Qt
GUI methods are invoked here.
"""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from inference_engine import InferenceResult

logger = logging.getLogger(__name__)


class ExportHandler:
    """
    Saves frames and annotations to disk in YOLO and COCO formats.

    Thread safety note: the ``export_all_frames`` method is intended to run
    inside a ``QThread``; it must not call any Qt GUI methods.  The
    ``progress_callback`` it accepts is expected to be a thread-safe signal
    emit.
    """

    def __init__(self):
        """Initialise with no output directory configured."""
        self._output_dir: Path | None = None
        # Running counter for COCO annotation IDs across multiple save calls
        self._coco_annotation_id: int = 1

    # ------------------------------------------------------------------ #

    def set_output_dir(self, path: str):
        """
        Configure the output root and create the required sub-directories.

        Parameters
        ----------
        path : str
            Root output directory path.  The sub-directories ``images/``,
            ``labels/``, and ``coco/`` are created automatically.

        Raises
        ------
        PermissionError
            Re-raised if any directory cannot be created due to permissions.
        """
        p = Path(path)
        try:
            for sub in ("images", "labels", "coco"):
                (p / sub).mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            logger.error(
                "Output directory not writable: %s — suggest a different path.", path
            )
            raise
        self._output_dir = p

    # ------------------------------------------------------------------ #

    def _require_output_dir(self):
        """
        Raise ``RuntimeError`` if no output directory has been configured.

        Raises
        ------
        RuntimeError
            When ``set_output_dir`` has not been called.
        """
        if self._output_dir is None:
            raise RuntimeError(
                "Output directory not configured — call set_output_dir() first."
            )

    # ------------------------------------------------------------------ #

    def save_yolo(
        self,
        frame_bgr: np.ndarray,
        result: InferenceResult,
        frame_idx: int,
    ) -> str:
        """
        Save the frame as JPEG and its detections as a YOLO annotation file.

        YOLO segmentation format (one detection per line)::

            class_id  x1_norm y1_norm x2_norm y2_norm ...

        Where each ``x_norm / y_norm`` pair is a polygon vertex normalised
        to ``[0, 1]`` by the frame width / height respectively.  Polygon
        points are down-sampled to a maximum of 50 vertices to keep files
        compact.

        Falls back to the standard YOLO detection (bounding-box) format when
        no mask is available for a detection::

            class_id  cx_norm cy_norm w_norm h_norm

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame to save as JPEG at quality 95.
        result : InferenceResult
            Inference output for this frame.
        frame_idx : int
            Zero-based frame index used to build the output filename.

        Returns
        -------
        str
            Absolute path to the saved ``.txt`` annotation file.
        """
        self._require_output_dir()

        frame_name = f"frame_{frame_idx:06d}"
        img_path   = self._output_dir / "images" / f"{frame_name}.jpg"
        lbl_path   = self._output_dir / "labels" / f"{frame_name}.txt"

        # Save JPEG (quality 95 as specified)
        success = cv2.imwrite(
            str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not success:
            logger.error("Failed to write image: %s", img_path)

        h, w = frame_bgr.shape[:2]
        lines: list[str] = []

        for i, class_id in enumerate(result.class_ids):
            wrote_polygon = False

            if i < len(result.masks_binary) and result.masks_binary[i] is not None:
                mask = result.masks_binary[i]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    # Use the largest contour by area
                    contour = max(contours, key=cv2.contourArea)
                    # Down-sample to at most 50 vertices for compact output
                    step = max(1, len(contour) // 50)
                    pts  = contour[::step].reshape(-1, 2)
                    coords = " ".join(
                        f"{pt[0] / w:.6f} {pt[1] / h:.6f}" for pt in pts
                    )
                    lines.append(f"{class_id} {coords}")
                    wrote_polygon = True

            if not wrote_polygon and i < len(result.boxes_xyxy):
                # Fallback: YOLO detection (bbox) format
                x1, y1, x2, y2 = result.boxes_xyxy[i]
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        return str(lbl_path)

    # ------------------------------------------------------------------ #

    def save_coco(
        self,
        frame_bgr: np.ndarray,
        result: InferenceResult,
        frame_idx: int,
        video_name: str,
    ) -> str:
        """
        Append the current frame's detections to ``coco/annotations.json``.

        The file accumulates entries across multiple ``save_coco`` calls,
        deduplicating by ``image_id``.  The schema follows the standard
        COCO object detection / segmentation format::

            {
                "images":      [...],
                "annotations": [...],
                "categories":  [...]
            }

        Segmentation is stored as a list of polygon coordinate lists.
        Bounding boxes use COCO format ``[x, y, w, h]`` (absolute pixels,
        not normalised).  Polygon area is computed via
        ``cv2.contourArea``.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame (dimensions used for width/height metadata).
        result : InferenceResult
            Inference output for this frame.
        frame_idx : int
            Used as the COCO ``image_id``.
        video_name : str
            Source video filename stored in the ``images`` entry.

        Returns
        -------
        str
            Absolute path to ``coco/annotations.json``.
        """
        self._require_output_dir()

        coco_path = self._output_dir / "coco" / "annotations.json"
        h, w = frame_bgr.shape[:2]

        # Load existing data or initialise a fresh structure
        if coco_path.exists():
            try:
                with coco_path.open("r", encoding="utf-8") as fh:
                    coco_data: dict = json.load(fh)
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "annotations.json appears corrupted — starting fresh."
                )
                coco_data = {"images": [], "annotations": [], "categories": []}
        else:
            coco_data = {"images": [], "annotations": [], "categories": []}

        # Keep annotation ID counter ahead of any existing entries
        if coco_data["annotations"]:
            max_existing = max(ann["id"] for ann in coco_data["annotations"])
            self._coco_annotation_id = max(
                self._coco_annotation_id, max_existing + 1
            )

        # Add image entry if not already present
        existing_image_ids = {img["id"] for img in coco_data["images"]}
        if frame_idx not in existing_image_ids:
            coco_data["images"].append(
                {
                    "id":         frame_idx,
                    "file_name":  f"frame_{frame_idx:06d}.jpg",
                    "width":      w,
                    "height":     h,
                    "video_name": video_name,
                }
            )

        # Merge new categories
        existing_cats: dict[int, str] = {
            cat["id"]: cat["name"] for cat in coco_data["categories"]
        }
        for cid, cname in zip(result.class_ids, result.class_names):
            if cid not in existing_cats:
                existing_cats[cid] = cname
        coco_data["categories"] = [
            {"id": cid, "name": cname, "supercategory": "object"}
            for cid, cname in sorted(existing_cats.items())
        ]

        # Remove any stale annotations for this frame (idempotent re-saves)
        coco_data["annotations"] = [
            ann
            for ann in coco_data["annotations"]
            if ann["image_id"] != frame_idx
        ]

        # Add new annotations
        for i, class_id in enumerate(result.class_ids):
            segmentation: list[list[float]] = []
            area: float = 0.0

            if i < len(result.masks_binary) and result.masks_binary[i] is not None:
                mask = result.masks_binary[i]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    area += float(cv2.contourArea(cnt))
                    pts  = cnt.reshape(-1, 2).tolist()
                    flat = [coord for pt in pts for coord in pt]
                    if len(flat) >= 6:          # COCO requires ≥ 3 points
                        segmentation.append([float(v) for v in flat])

            # COCO bbox: [x, y, w, h] in absolute pixels
            if i < len(result.boxes_xyxy):
                x1, y1, x2, y2 = result.boxes_xyxy[i]
                bbox = [
                    float(x1), float(y1),
                    float(x2 - x1), float(y2 - y1),
                ]
                if area == 0.0:
                    area = float((x2 - x1) * (y2 - y1))
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]

            coco_data["annotations"].append(
                {
                    "id":           self._coco_annotation_id,
                    "image_id":     frame_idx,
                    "category_id":  class_id,
                    "segmentation": segmentation,
                    "area":         area,
                    "bbox":         bbox,
                    "iscrowd":      0,
                    "score":        float(result.confidences[i])
                                    if i < len(result.confidences) else 1.0,
                }
            )
            self._coco_annotation_id += 1

        with coco_path.open("w", encoding="utf-8") as fh:
            json.dump(coco_data, fh, indent=2)

        return str(coco_path)

    # ------------------------------------------------------------------ #

    def export_all_frames(
        self,
        video_handler,
        inference_engine,
        conf: float,
        iou: float,
        progress_callback=None,
    ):
        """
        Iterate every frame in *video_handler*, run inference, and save YOLO
        annotations for each frame.

        This method is designed to be called from a ``QThread`` worker.  It
        does **not** call any Qt GUI methods.  All GUI updates must be
        performed via the ``progress_callback`` signal emit.

        Parameters
        ----------
        video_handler : VideoHandler
            An open video source.
        inference_engine : InferenceEngine
            A loaded inference engine.
        conf : float
            Confidence threshold passed to each ``infer()`` call.
        iou : float
            IoU threshold passed to each ``infer()`` call.
        progress_callback : callable or None
            Optional ``callback(current: int, total: int)`` invoked after
            each frame is processed.  Must be thread-safe (typically a
            Qt signal ``emit``).
        """
        total = video_handler.total_frames()
        for idx in range(total):
            frame = video_handler.get_frame(idx)
            if frame is None:
                logger.warning("Could not read frame %d — skipping.", idx)
            else:
                try:
                    result = inference_engine.infer(frame, conf=conf, iou=iou)
                    self.save_yolo(frame, result, idx)
                except Exception as exc:
                    logger.error("Export failed at frame %d: %s", idx, exc)

            if progress_callback is not None:
                progress_callback(idx + 1, total)

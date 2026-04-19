"""
export_handler.py
─────────────────
Saves video frames and their inference annotations to disk in YOLO
segmentation, COCO JSON, CVAT XML 1.1, and LabelStudio JSON formats.

Output directory layout::

    <output_dir>/
        images/            ← raw JPEG frames
        labels/            ← YOLO .txt files
        coco/
            annotations.json
        cvat/
            annotations.xml
        labelstudio/
            tasks.json

All export methods are safe to call from a QThread worker; no Qt GUI
methods are invoked here.
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from inference_engine import InferenceResult

logger = logging.getLogger(__name__)


class ExportHandler:
    """
    Saves frames and annotations to disk in YOLO, COCO, CVAT, and
    LabelStudio formats.

    Thread safety: ``export_all_frames`` runs inside a QThread and must
    not call any Qt GUI methods.  Use the ``progress_callback`` for GUI
    updates via a thread-safe signal emit.
    """

    def __init__(self):
        """Initialise with no output directory configured."""
        self._output_dir: Path | None = None
        self._coco_annotation_id: int = 1

    # ------------------------------------------------------------------ #

    def set_output_dir(self, path: str):
        """
        Configure the output root and create all required sub-directories.

        Parameters
        ----------
        path : str
            Root output directory.  Sub-directories ``images/``, ``labels/``,
            ``coco/``, ``cvat/``, and ``labelstudio/`` are created automatically.

        Raises
        ------
        PermissionError
            Re-raised if any directory cannot be created due to permissions.
        """
        p = Path(path)
        try:
            for sub in ("images", "labels", "coco", "cvat", "labelstudio",
                        "semantic"):
                (p / sub).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(
                "Output directory not writable: %s — suggest a different path.", path
            )
            raise
        self._output_dir = p

    # ------------------------------------------------------------------ #

    def _require_output_dir(self):
        """Raise RuntimeError when no output directory has been configured."""
        if self._output_dir is None:
            raise RuntimeError(
                "Output directory not configured — call set_output_dir() first."
            )

    # ------------------------------------------------------------------ #

    def save_semantic_png(
        self,
        frame_bgr: np.ndarray,
        semantic_result: "object",   # SemanticResult (avoid circular import)
        frame_idx: int,
    ) -> tuple[str, str]:
        """
        Save a semantic segmentation result as two PNG files:

        * ``semantic/frame_XXXXXX_labels.png`` — raw label-ID map (grayscale,
          one pixel value per class ID).  This is the format consumed by
          mmsegmentation, SegFormer training pipelines, etc.
        * ``semantic/frame_XXXXXX_color.png``  — BGR colour visualisation for
          human review.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Original frame (used for image-side metadata only — not saved here;
            use ``save_yolo`` / ``save_coco`` to save the JPEG image).
        semantic_result : SemanticResult
            Output of ``SemanticSegmentationEngine.infer()``.
        frame_idx : int
            Zero-based frame index (sets the filename).

        Returns
        -------
        (label_path, color_path) : tuple[str, str]
            Absolute paths to the two saved PNG files.
        """
        self._require_output_dir()
        stem = f"frame_{frame_idx:06d}"
        sem_dir = self._output_dir / "semantic"

        label_path = sem_dir / f"{stem}_labels.png"
        color_path = sem_dir / f"{stem}_color.png"

        cv2.imwrite(str(label_path), semantic_result.label_map)
        cv2.imwrite(str(color_path), semantic_result.colormap)

        logger.debug("Saved semantic PNGs: %s / %s", label_path, color_path)
        return str(label_path), str(color_path)

    # ------------------------------------------------------------------ #

    def generate_data_yaml(self, class_names: list[str]) -> str:
        """
        Write a YOLO-format ``data.yaml`` in the output directory, ready for
        ``yolo train data=<path>/data.yaml``.

        The same ``images/`` folder is used for both train and val — suitable
        for pseudo-label training.  Users can split into a proper val set later.

        Parameters
        ----------
        class_names : list[str]
            Ordered list of class name strings matching the label IDs.

        Returns
        -------
        str
            Absolute path to the written ``data.yaml``.
        """
        self._require_output_dir()
        lines = [
            "path: .",   # relative — portable across machines and directories
            "train: images",
            "val: images",
            f"nc: {len(class_names)}",
            "names:",
        ]
        for i, name in enumerate(class_names):
            lines.append(f"  {i}: {name}")

        yaml_path = self._output_dir / "data.yaml"
        yaml_path.write_text("\n".join(lines) + "\n")
        logger.info("data.yaml written → %s", yaml_path)
        return str(yaml_path)

    # ------------------------------------------------------------------ #

    @staticmethod
    def filter_result(
        result: InferenceResult,
        enabled_classes: "set[int] | None",
    ) -> InferenceResult:
        """
        Return a copy of *result* containing only detections whose class ID
        is present in *enabled_classes*.

        Parameters
        ----------
        result : InferenceResult
            Source inference result.
        enabled_classes : set[int] or None
            Set of class IDs to keep.  ``None`` means keep all.

        Returns
        -------
        InferenceResult
            Filtered result (new object; original is unmodified).
        """
        if enabled_classes is None:
            return result
        indices = [
            i for i, cid in enumerate(result.class_ids)
            if cid in enabled_classes
        ]
        return InferenceResult(
            boxes_xyxy   = [result.boxes_xyxy[i]   for i in indices],
            class_ids    = [result.class_ids[i]    for i in indices],
            confidences  = [result.confidences[i]  for i in indices],
            masks_binary = [result.masks_binary[i] for i in indices
                            if i < len(result.masks_binary)],
            class_names  = [result.class_names[i]  for i in indices],
            orig_shape   = result.orig_shape,
        )

    # ------------------------------------------------------------------ #

    def save_yolo(
        self,
        frame_bgr: np.ndarray,
        result: InferenceResult,
        frame_idx: int,
    ) -> str:
        """
        Save the frame as JPEG and write a YOLO-format annotation file.

        YOLO segmentation format (one detection per line)::

            class_id  x1 y1 x2 y2 ...  (polygon vertices, normalised 0-1)

        Falls back to bounding-box format when no mask is available::

            class_id  cx cy width height  (normalised 0-1)

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame (saved at JPEG quality 95).
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
                    contour = max(contours, key=cv2.contourArea)
                    step    = max(1, len(contour) // 50)
                    pts     = contour[::step].reshape(-1, 2)
                    coords  = " ".join(
                        f"{pt[0] / w:.6f} {pt[1] / h:.6f}" for pt in pts
                    )
                    lines.append(f"{class_id} {coords}")
                    wrote_polygon = True

            if not wrote_polygon and i < len(result.boxes_xyxy):
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
        Append the frame's detections to ``coco/annotations.json``.

        Uses standard COCO object detection / segmentation schema.
        Bounding boxes are in COCO ``[x, y, w, h]`` (absolute pixels).
        Polygon area is computed via ``cv2.contourArea``.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame (dimensions used for image metadata).
        result : InferenceResult
            Inference output for this frame.
        frame_idx : int
            Used as the COCO ``image_id``.
        video_name : str
            Source video filename stored in the images entry.

        Returns
        -------
        str
            Absolute path to ``coco/annotations.json``.
        """
        self._require_output_dir()

        coco_path = self._output_dir / "coco" / "annotations.json"
        h, w = frame_bgr.shape[:2]

        if coco_path.exists():
            try:
                with coco_path.open("r", encoding="utf-8") as fh:
                    coco_data: dict = json.load(fh)
            except (json.JSONDecodeError, OSError):
                logger.warning("annotations.json corrupted — starting fresh.")
                coco_data = {"images": [], "annotations": [], "categories": []}
        else:
            coco_data = {"images": [], "annotations": [], "categories": []}

        if coco_data["annotations"]:
            max_existing = max(a["id"] for a in coco_data["annotations"])
            self._coco_annotation_id = max(self._coco_annotation_id, max_existing + 1)

        existing_ids = {img["id"] for img in coco_data["images"]}
        if frame_idx not in existing_ids:
            coco_data["images"].append({
                "id": frame_idx, "file_name": f"frame_{frame_idx:06d}.jpg",
                "width": w, "height": h, "video_name": video_name,
            })

        existing_cats: dict[int, str] = {
            c["id"]: c["name"] for c in coco_data["categories"]
        }
        for cid, cname in zip(result.class_ids, result.class_names):
            if cid not in existing_cats:
                existing_cats[cid] = cname
        coco_data["categories"] = [
            {"id": cid, "name": cname, "supercategory": "object"}
            for cid, cname in sorted(existing_cats.items())
        ]

        coco_data["annotations"] = [
            a for a in coco_data["annotations"] if a["image_id"] != frame_idx
        ]

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
                    flat = [float(v) for pt in cnt.reshape(-1, 2).tolist()
                            for v in pt]
                    if len(flat) >= 6:
                        segmentation.append(flat)

            if i < len(result.boxes_xyxy):
                x1, y1, x2, y2 = result.boxes_xyxy[i]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                if area == 0.0:
                    area = float((x2 - x1) * (y2 - y1))
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]

            coco_data["annotations"].append({
                "id": self._coco_annotation_id,
                "image_id": frame_idx,
                "category_id": class_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": float(result.confidences[i])
                          if i < len(result.confidences) else 1.0,
            })
            self._coco_annotation_id += 1

        with coco_path.open("w", encoding="utf-8") as fh:
            json.dump(coco_data, fh, indent=2)
        return str(coco_path)

    # ------------------------------------------------------------------ #

    def save_cvat(
        self,
        frame_bgr: np.ndarray,
        result: InferenceResult,
        frame_idx: int,
        video_name: str,
    ) -> str:
        """
        Append the frame's detections to ``cvat/annotations.xml`` in
        CVAT XML 1.1 format.

        Each detection with a mask is written as a ``<polygon>`` element;
        detections with only a bounding box are written as ``<box>``
        elements.  The file is idempotent: re-saving the same frame_idx
        replaces the previous entry.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame (dimensions used for image metadata).
        result : InferenceResult
            Inference output for this frame.
        frame_idx : int
            Zero-based frame index.
        video_name : str
            Source video filename.

        Returns
        -------
        str
            Absolute path to ``cvat/annotations.xml``.
        """
        self._require_output_dir()

        cvat_path = self._output_dir / "cvat" / "annotations.xml"
        h, w = frame_bgr.shape[:2]
        fname = f"frame_{frame_idx:06d}.jpg"

        # Load or create XML tree
        if cvat_path.exists():
            try:
                tree = ET.parse(str(cvat_path))
                root = tree.getroot()
            except ET.ParseError:
                root = self._cvat_root()
        else:
            root = self._cvat_root()

        # Merge label names into <meta><task><labels>
        labels_el = root.find(".//labels")
        if labels_el is None:
            labels_el = ET.SubElement(root.find(".//task"), "labels")
        existing_label_names = {
            el.findtext("name") for el in labels_el.findall("label")
        }
        for cname in result.class_names:
            if cname and cname not in existing_label_names:
                lbl_el = ET.SubElement(labels_el, "label")
                ET.SubElement(lbl_el, "name").text = cname
                ET.SubElement(lbl_el, "color").text = "#ff6600"
                existing_label_names.add(cname)

        # Remove stale entry for this frame (idempotent re-saves)
        for old in root.findall(f"image[@name='{fname}']"):
            root.remove(old)

        img_el = ET.SubElement(root, "image")
        img_el.set("id",     str(frame_idx))
        img_el.set("name",   fname)
        img_el.set("width",  str(w))
        img_el.set("height", str(h))
        img_el.set("source", video_name)

        for i, class_id in enumerate(result.class_ids):
            cname = (result.class_names[i]
                     if i < len(result.class_names) else str(class_id))
            conf  = (result.confidences[i]
                     if i < len(result.confidences) else 1.0)

            wrote_polygon = False
            if i < len(result.masks_binary) and result.masks_binary[i] is not None:
                mask = result.masks_binary[i]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    contour  = max(contours, key=cv2.contourArea)
                    step     = max(1, len(contour) // 50)
                    pts      = contour[::step].reshape(-1, 2)
                    pts_str  = ";".join(
                        f"{pt[0]:.2f},{pt[1]:.2f}" for pt in pts
                    )
                    poly_el  = ET.SubElement(img_el, "polygon")
                    poly_el.set("label",    cname)
                    poly_el.set("points",   pts_str)
                    poly_el.set("occluded", "0")
                    poly_el.set("source",   "auto")
                    poly_el.set("z_order",  "0")
                    poly_el.set("conf",     f"{conf:.4f}")
                    wrote_polygon = True

            if not wrote_polygon and i < len(result.boxes_xyxy):
                x1, y1, x2, y2 = result.boxes_xyxy[i]
                box_el = ET.SubElement(img_el, "box")
                box_el.set("label",    cname)
                box_el.set("xtl",      f"{x1:.2f}")
                box_el.set("ytl",      f"{y1:.2f}")
                box_el.set("xbr",      f"{x2:.2f}")
                box_el.set("ybr",      f"{y2:.2f}")
                box_el.set("occluded", "0")
                box_el.set("source",   "auto")
                box_el.set("z_order",  "0")
                box_el.set("conf",     f"{conf:.4f}")

        ET.indent(root, space="  ")
        ET.ElementTree(root).write(
            str(cvat_path), encoding="utf-8", xml_declaration=True
        )
        return str(cvat_path)

    @staticmethod
    def _cvat_root() -> ET.Element:
        """
        Create a minimal CVAT XML 1.1 root element.

        Returns
        -------
        ET.Element
            ``<annotations>`` root with ``<version>`` and ``<meta>`` stubs.
        """
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "name").text = "FrameForge Export"
        ET.SubElement(task, "mode").text = "annotation"
        ET.SubElement(task, "labels")
        return root

    # ------------------------------------------------------------------ #

    def save_labelstudio(
        self,
        frame_bgr: np.ndarray,
        result: InferenceResult,
        frame_idx: int,
        video_name: str,
    ) -> str:
        """
        Append the frame's detections to ``labelstudio/tasks.json``.

        Outputs standard LabelStudio task JSON.  Bounding boxes are stored
        as ``rectanglelabels``; masks as ``polygonlabels``.  All coordinates
        are in percentage units (0-100) as required by LabelStudio.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame.
        result : InferenceResult
            Inference output for this frame.
        frame_idx : int
            Zero-based frame index.
        video_name : str
            Source video filename.

        Returns
        -------
        str
            Absolute path to ``labelstudio/tasks.json``.
        """
        self._require_output_dir()

        ls_path = self._output_dir / "labelstudio" / "tasks.json"
        h, w    = frame_bgr.shape[:2]
        fname   = f"frame_{frame_idx:06d}.jpg"

        if ls_path.exists():
            try:
                with ls_path.open("r", encoding="utf-8") as fh:
                    tasks: list = json.load(fh)
            except (json.JSONDecodeError, OSError):
                tasks = []
        else:
            tasks = []

        # Remove stale entry (idempotent re-saves)
        tasks = [
            t for t in tasks
            if t.get("data", {}).get("image", "").split("/")[-1] != fname
        ]

        annotation_results: list[dict] = []
        for i, class_id in enumerate(result.class_ids):
            cname = (result.class_names[i]
                     if i < len(result.class_names) else str(class_id))
            conf  = (result.confidences[i]
                     if i < len(result.confidences) else 1.0)
            uid   = f"ann_{frame_idx}_{i}"

            wrote_polygon = False
            if i < len(result.masks_binary) and result.masks_binary[i] is not None:
                mask = result.masks_binary[i]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    step    = max(1, len(contour) // 50)
                    pts     = contour[::step].reshape(-1, 2)
                    # LabelStudio polygon uses percentage coordinates
                    points  = [
                        [float(pt[0]) / w * 100.0, float(pt[1]) / h * 100.0]
                        for pt in pts
                    ]
                    annotation_results.append({
                        "id": uid, "type": "polygonlabels",
                        "from_name": "label", "to_name": "image",
                        "original_width": w, "original_height": h,
                        "value": {
                            "points": points,
                            "polygonlabels": [cname],
                            "confidence": conf,
                        },
                    })
                    wrote_polygon = True

            if not wrote_polygon and i < len(result.boxes_xyxy):
                x1, y1, x2, y2 = result.boxes_xyxy[i]
                annotation_results.append({
                    "id": uid, "type": "rectanglelabels",
                    "from_name": "label", "to_name": "image",
                    "original_width": w, "original_height": h,
                    "value": {
                        "x":     float(x1) / w * 100.0,
                        "y":     float(y1) / h * 100.0,
                        "width": float(x2 - x1) / w * 100.0,
                        "height": float(y2 - y1) / h * 100.0,
                        "rotation": 0,
                        "rectanglelabels": [cname],
                        "confidence": conf,
                    },
                })

        tasks.append({
            "data":        {"image": f"images/{fname}", "video_name": video_name},
            "annotations": [{"result": annotation_results}],
            "meta":        {"frame_idx": frame_idx},
        })

        with ls_path.open("w", encoding="utf-8") as fh:
            json.dump(tasks, fh, indent=2)
        return str(ls_path)

    # ------------------------------------------------------------------ #

    def export_all_frames(
        self,
        video_handler,
        inference_engine,
        conf: float,
        iou: float,
        progress_callback=None,
        stride: int = 1,
        enabled_classes: "set[int] | None" = None,
    ):
        """
        Iterate frames in *video_handler* at *stride* intervals, run
        inference on each, apply an optional class filter, and save YOLO
        annotations.

        Designed to be called from a QThread worker; no Qt GUI methods are
        invoked here.

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
            Optional ``callback(current: int, total: int)`` — must be
            thread-safe (e.g. a Qt signal emit).
        stride : int
            Process every *stride*-th frame (default 1 = every frame).
        enabled_classes : set[int] or None
            Only export detections whose class ID is in this set.
            ``None`` exports all classes.
        """
        total            = video_handler.total_frames()
        stride           = max(1, stride)
        frames_to_export = list(range(0, total, stride))
        n_export         = len(frames_to_export)

        for pos, idx in enumerate(frames_to_export):
            frame = video_handler.get_frame(idx)
            if frame is None:
                logger.warning("Could not read frame %d — skipping.", idx)
            else:
                try:
                    result = inference_engine.infer(frame, conf=conf, iou=iou)
                    if enabled_classes is not None:
                        result = self.filter_result(result, enabled_classes)
                    self.save_yolo(frame, result, idx)
                except Exception as exc:
                    logger.error("Export failed at frame %d: %s", idx, exc)

            if progress_callback is not None:
                progress_callback(pos + 1, n_export)

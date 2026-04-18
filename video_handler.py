"""
video_handler.py
────────────────
Provides the VideoHandler class — a thin OpenCV wrapper for seek-based,
single-frame random access to MP4 (and other OpenCV-supported) video files.

Design constraint: never loads more than one decoded frame into memory at a
time.  Seeking is done via ``cap.set(cv2.CAP_PROP_POS_FRAMES, idx)`` so that
arbitrary frames can be retrieved without sequential decoding and without
holding the entire video in RAM.
"""

import cv2
import numpy as np


class VideoHandler:
    """
    Wraps cv2.VideoCapture for seek-based random access.

    Never loads more than one frame into memory at a time.  All metadata
    (total frames, FPS, resolution) is cached at ``open()`` time so that
    subsequent queries are O(1) without touching the capture object.
    """

    def __init__(self):
        """Initialise with no open capture and zeroed metadata."""
        self._cap: cv2.VideoCapture | None = None
        self._path: str = ""
        self._total_frames: int = 0
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------ #

    def open(self, path: str) -> bool:
        """
        Open a video file, releasing any previously open capture first.

        Parameters
        ----------
        path : str
            Absolute or relative path to an MP4 (or any OpenCV-supported) file.

        Returns
        -------
        bool
            True if the file was opened and metadata was read successfully,
            False otherwise.
        """
        self.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False

        self._cap = cap
        self._path = path
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = raw_fps if raw_fps > 0 else 30.0
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    # ------------------------------------------------------------------ #

    def get_frame(self, idx: int) -> np.ndarray | None:
        """
        Seek to frame *idx* and return it as a BGR numpy array.

        Uses ``CAP_PROP_POS_FRAMES`` for true random access; no sequential
        decode buffer is maintained between calls.

        Parameters
        ----------
        idx : int
            Zero-based frame index.  Clamped to ``[0, total_frames - 1]``
            internally.

        Returns
        -------
        np.ndarray or None
            HxWx3 BGR uint8 array on success, or ``None`` if the seek or
            read operation failed.
        """
        if self._cap is None:
            return None
        idx = max(0, min(idx, self._total_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame

    # ------------------------------------------------------------------ #

    def total_frames(self) -> int:
        """
        Return the total number of frames reported by the video container.

        Returns
        -------
        int
            Total frame count, or 0 if no video is open.
        """
        return self._total_frames

    # ------------------------------------------------------------------ #

    def fps(self) -> float:
        """
        Return the video's native frame rate.

        Returns
        -------
        float
            Frames per second (defaults to 30.0 if the container reports 0).
        """
        return self._fps

    # ------------------------------------------------------------------ #

    def resolution(self) -> tuple[int, int]:
        """
        Return the video's pixel dimensions.

        Returns
        -------
        tuple[int, int]
            ``(width, height)`` in pixels, or ``(0, 0)`` if no video is open.
        """
        return (self._width, self._height)

    # ------------------------------------------------------------------ #

    def release(self):
        """
        Release the underlying cv2.VideoCapture and reset all cached metadata.

        Safe to call even if no video is currently open.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._path = ""
        self._total_frames = 0
        self._fps = 0.0
        self._width = 0
        self._height = 0

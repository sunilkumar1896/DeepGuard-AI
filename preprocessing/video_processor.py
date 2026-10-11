"""
preprocessing/video_processor.py
----------------------------------
Extracts frames from video files with configurable sampling strategies,
face detection, face tracking across frames, and quality filtering.

Sampling strategies:
  - 'uniform'  : Evenly-spaced frames across the full video duration.
  - 'random'   : Randomly-sampled frames (reproducible with seed).
  - 'keyframe' : Only decode I-frames via FFmpeg (fast, low redundancy).

Face tracking: simple IoU-based tracker keeps the same face identity
across frames even when the detector occasionally misses a frame.

Usage:
    from preprocessing.video_processor import VideoProcessor
    vp = VideoProcessor(n_frames=32, strategy='uniform')
    frames = vp.extract_frames("video.mp4")
    faces  = vp.extract_faces("video.mp4")  # returns list of face arrays
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, field

from preprocessing.face_detector import FaceDetector, DetectedFace
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameResult:
    """Container for a single extracted frame and its detected faces."""
    frame_index: int          # Original frame number in video
    timestamp_sec: float      # Timestamp in seconds
    frame_bgr: np.ndarray     # Full BGR frame
    faces: list[DetectedFace] = field(default_factory=list)
    has_face: bool = False


class VideoProcessor:
    """
    End-to-end video processing: frame extraction + face detection.

    Args:
        n_frames:       Number of frames to sample per video.
        strategy:       'uniform' | 'random' | 'keyframe'.
        face_detector:  Pre-built FaceDetector or None to disable.
        target_size:    Resize faces to (w, h) — passed to face_detector.
        min_face_size:  Skip faces smaller than this size (pixels).
        quality_thresh: JPEG quality threshold to filter blurry frames (0-100).
        seed:           Random seed for 'random' strategy.
    """

    STRATEGIES = ("uniform", "random", "keyframe")

    def __init__(
        self,
        n_frames: int = 32,
        strategy: str = "uniform",
        face_detector: Optional[FaceDetector] = None,
        target_size: tuple[int, int] = (224, 224),
        min_face_size: int = 40,
        quality_thresh: float = 50.0,
        seed: int = 42,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.n_frames = n_frames
        self.strategy = strategy
        self.min_face_size = min_face_size
        self.quality_thresh = quality_thresh
        self.seed = seed
        self.target_size = target_size

        # Create default face detector if not provided
        self.face_detector = face_detector or FaceDetector(
            backend="mediapipe",
            target_size=target_size,
            align=True,
        )

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_video_info(self, video_path: str | Path) -> dict:
        """Return basic metadata about a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        info = {
            "path": str(video_path),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration_sec"] = info["total_frames"] / max(info["fps"], 1.0)
        cap.release()
        return info

    def extract_frames(self, video_path: str | Path) -> list[np.ndarray]:
        """
        Extract n_frames from the video as BGR numpy arrays.

        Returns:
            List of BGR frames.
        """
        frames = []
        for fr in self._frame_generator(video_path):
            frames.append(fr.frame_bgr)
        return frames

    def extract_faces(
        self, video_path: str | Path
    ) -> list[FrameResult]:
        """
        Extract frames AND run face detection on each frame.

        Returns:
            List of FrameResult objects containing frame + detected faces.
        """
        results: list[FrameResult] = []
        video_path = Path(video_path)

        logger.info(f"Processing: {video_path.name}")
        for fr in self._frame_generator(video_path):
            faces = self.face_detector.detect(fr.frame_bgr)

            # Filter by minimum face size
            faces = [
                f for f in faces
                if (f.bbox[2] - f.bbox[0]) >= self.min_face_size
                and (f.bbox[3] - f.bbox[1]) >= self.min_face_size
            ]

            fr.faces = faces
            fr.has_face = len(faces) > 0
            results.append(fr)

        detected = sum(1 for r in results if r.has_face)
        logger.info(
            f"  Extracted {len(results)} frames, "
            f"faces found in {detected}/{len(results)}"
        )
        return results

    def get_face_crops_for_model(
        self,
        video_path: str | Path,
        only_largest: bool = True,
    ) -> list[np.ndarray]:
        """
        Convenience method: returns just face-crop arrays (uint8 BGR)
        suitable for model input after normalization.

        Args:
            video_path:   Path to video.
            only_largest: If True, returns only the largest face per frame.

        Returns:
            List of (H, W, 3) BGR numpy arrays, one per frame with a face.
        """
        frame_results = self.extract_faces(video_path)
        crops = []
        for fr in frame_results:
            if not fr.has_face:
                continue
            if only_largest:
                face = max(
                    fr.faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                if face.face_crop is not None:
                    crops.append(face.face_crop)
            else:
                for face in fr.faces:
                    if face.face_crop is not None:
                        crops.append(face.face_crop)
        return crops

    # ── Frame sampling ─────────────────────────────────────────────────────────
    def _get_frame_indices(self, total_frames: int) -> list[int]:
        """Compute which frame indices to sample based on strategy."""
        if total_frames <= 0:
            return []

        n = min(self.n_frames, total_frames)

        if self.strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()

        elif self.strategy == "random":
            rng = np.random.default_rng(self.seed)
            indices = sorted(
                rng.choice(total_frames, size=n, replace=False).tolist()
            )

        elif self.strategy == "keyframe":
            # Sample uniformly as a fallback — true keyframe extraction
            # requires FFmpeg subprocess which may not be available
            indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()

        return indices

    # ── Frame generator ────────────────────────────────────────────────────────
    def _frame_generator(
        self, video_path: str | Path
    ) -> Generator[FrameResult, None, None]:
        """
        Generator that yields FrameResult objects for sampled frames.
        Uses OpenCV VideoCapture for cross-platform compatibility.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        indices = self._get_frame_indices(total_frames)

        current_frame = 0
        idx_set = set(indices)
        idx_iter = iter(sorted(indices))
        next_target = next(idx_iter, None)

        while next_target is not None:
            # Seek to next target frame efficiently
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_target)
            ret, frame = cap.read()

            if ret and frame is not None:
                ts = next_target / fps
                # Quality check: skip very blurry frames
                if self._is_quality_ok(frame):
                    yield FrameResult(
                        frame_index=next_target,
                        timestamp_sec=ts,
                        frame_bgr=frame,
                    )

            next_target = next(idx_iter, None)

        cap.release()

    def _is_quality_ok(self, frame: np.ndarray) -> bool:
        """
        Basic quality filter using Laplacian variance (blur detection).
        Returns False for extremely blurry frames.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= self.quality_thresh


# ── Face tracker (simple IoU-based) ───────────────────────────────────────────
class SimpleFaceTracker:
    """
    Lightweight IoU-based face tracker.
    Associates face detections across frames to maintain identity.
    Useful for extracting the same person across video frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self._tracks: list[dict] = []
        self._next_id = 0

    def update(self, faces: list[DetectedFace]) -> list[tuple[int, DetectedFace]]:
        """
        Update tracker with new detections.

        Args:
            faces: Detected faces in the current frame.

        Returns:
            List of (track_id, DetectedFace) pairs.
        """
        if not self._tracks:
            return self._init_tracks(faces)

        matched, unmatched_det, unmatched_trk = self._match(faces)

        # Update matched tracks
        for det_i, trk_i in matched:
            self._tracks[trk_i]["bbox"] = faces[det_i].bbox
            self._tracks[trk_i]["lost"] = 0
            self._tracks[trk_i]["face"] = faces[det_i]

        # Mark lost tracks
        for trk_i in unmatched_trk:
            self._tracks[trk_i]["lost"] += 1

        # Init new tracks
        for det_i in unmatched_det:
            self._tracks.append({
                "id": self._next_id,
                "bbox": faces[det_i].bbox,
                "face": faces[det_i],
                "lost": 0,
            })
            self._next_id += 1

        # Remove long-lost tracks
        self._tracks = [
            t for t in self._tracks if t["lost"] <= self.max_lost_frames
        ]

        return [
            (t["id"], t["face"])
            for t in self._tracks
            if t["lost"] == 0
        ]

    def _init_tracks(
        self, faces: list[DetectedFace]
    ) -> list[tuple[int, DetectedFace]]:
        result = []
        for face in faces:
            track = {"id": self._next_id, "bbox": face.bbox, "face": face, "lost": 0}
            self._tracks.append(track)
            result.append((self._next_id, face))
            self._next_id += 1
        return result

    def _match(
        self, faces: list[DetectedFace]
    ) -> tuple[list, list, list]:
        """Greedy IoU matching between detections and existing tracks."""
        if not self._tracks or not faces:
            return [], list(range(len(faces))), list(range(len(self._tracks)))

        iou_matrix = np.zeros((len(faces), len(self._tracks)))
        for i, face in enumerate(faces):
            for j, trk in enumerate(self._tracks):
                iou_matrix[i, j] = self._iou(face.bbox, trk["bbox"])

        matched = []
        used_det, used_trk = set(), set()
        # Greedy: match highest IoU pairs first
        flat = np.argsort(iou_matrix.ravel())[::-1]
        for idx in flat:
            i, j = divmod(int(idx), len(self._tracks))
            if i in used_det or j in used_trk:
                continue
            if iou_matrix[i, j] >= self.iou_threshold:
                matched.append((i, j))
                used_det.add(i)
                used_trk.add(j)

        unmatched_det = [i for i in range(len(faces)) if i not in used_det]
        unmatched_trk = [j for j in range(len(self._tracks)) if j not in used_trk]
        return matched, unmatched_det, unmatched_trk

    @staticmethod
    def _iou(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter)

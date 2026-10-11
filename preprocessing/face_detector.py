"""
preprocessing/face_detector.py
-------------------------------
Multi-backend face detector supporting:
  - MediaPipe (fast, good for real-time)
  - RetinaFace via InsightFace (accurate, heavier)
  - OpenCV DNN Caffe model (fallback, no extra deps)

Returns detected faces with bounding boxes, landmarks,
confidence scores, and cropped/aligned face images.

Usage:
    from preprocessing.face_detector import FaceDetector
    detector = FaceDetector(backend='mediapipe')
    faces = detector.detect(frame_bgr)
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Data class for a single detected face ─────────────────────────────────────
@dataclass
class DetectedFace:
    """Represents one detected face in a frame."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) absolute pixels
    confidence: float
    landmarks: Optional[np.ndarray] = None   # (5, 2) key-points if available
    face_crop: Optional[np.ndarray] = None   # cropped & aligned BGR face
    face_index: int = 0


# ── Base class ────────────────────────────────────────────────────────────────
class FaceDetector:
    """
    Unified face detector with pluggable backends.

    Args:
        backend:       'mediapipe' | 'retinaface' | 'opencv_dnn'
        min_confidence: Minimum detection confidence threshold.
        margin:        Fractional margin to add around each face bbox.
        target_size:   (w, h) to resize cropped faces, or None to skip.
        align:         Whether to perform landmark-based alignment.
    """

    SUPPORTED_BACKENDS = ("mediapipe", "retinaface", "opencv_dnn")

    def __init__(
        self,
        backend: str = "mediapipe",
        min_confidence: float = 0.5,
        margin: float = 0.3,
        target_size: Optional[tuple[int, int]] = (224, 224),
        align: bool = True,
    ) -> None:
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from {self.SUPPORTED_BACKENDS}"
            )
        self.backend = backend
        self.min_confidence = min_confidence
        self.margin = margin
        self.target_size = target_size
        self.align = align
        self._model = None
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        if self.backend == "mediapipe":
            self._load_mediapipe()
        elif self.backend == "retinaface":
            self._load_retinaface()
        elif self.backend == "opencv_dnn":
            self._load_opencv_dnn()

    def _load_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection
            self._detector = self._mp_face.FaceDetection(
                model_selection=1,               # 1 = full-range model
                min_detection_confidence=self.min_confidence,
            )
            logger.info("MediaPipe face detector loaded.")
        except ImportError:
            logger.warning("MediaPipe not installed — falling back to OpenCV DNN.")
            self.backend = "opencv_dnn"
            self._load_opencv_dnn()

    def _load_retinaface(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self._detector.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace RetinaFace detector loaded.")
        except ImportError:
            logger.warning("InsightFace not installed — falling back to MediaPipe.")
            self.backend = "mediapipe"
            self._load_mediapipe()

    def _load_opencv_dnn(self) -> None:
        """Load OpenCV's res10 SSD face detector (ships with opencv-contrib)."""
        # Model files — download if missing
        model_dir = Path(__file__).parent.parent / "models" / "opencv_dnn"
        model_dir.mkdir(parents=True, exist_ok=True)
        proto = model_dir / "deploy.prototxt"
        weights = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if not proto.exists() or not weights.exists():
            self._download_opencv_dnn_model(model_dir)

        self._detector = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        logger.info("OpenCV DNN face detector loaded.")

    @staticmethod
    def _download_opencv_dnn_model(save_dir: Path) -> None:
        """Download OpenCV SSD face detector weights."""
        import urllib.request
        base = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector"
        files = {
            "deploy.prototxt": f"{base}/deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": (
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
                "res10_300x300_ssd_iter_140000.caffemodel"
            ),
        }
        for fname, url in files.items():
            dest = save_dir / fname
            logger.info(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)
        logger.info("OpenCV DNN model downloaded.")

    # ── Public API ─────────────────────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image as numpy array.

        Returns:
            List of DetectedFace objects sorted by confidence (descending).
        """
        if self.backend == "mediapipe":
            faces = self._detect_mediapipe(frame_bgr)
        elif self.backend == "retinaface":
            faces = self._detect_retinaface(frame_bgr)
        else:
            faces = self._detect_opencv_dnn(frame_bgr)

        # Crop and optionally align each detection
        h, w = frame_bgr.shape[:2]
        result = []
        for i, face in enumerate(faces):
            crop = self._crop_face(frame_bgr, face.bbox, h, w)
            if crop is None:
                continue
            if self.align and face.landmarks is not None:
                crop = self._align_face(crop, face.landmarks, face.bbox)
            if self.target_size:
                crop = cv2.resize(crop, self.target_size)
            face.face_crop = crop
            face.face_index = i
            result.append(face)

        return sorted(result, key=lambda f: f.confidence, reverse=True)

    def detect_largest(self, frame_bgr: np.ndarray) -> Optional[DetectedFace]:
        """Return only the largest (by area) detected face."""
        faces = self.detect(frame_bgr)
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    # ── Backend-specific detection ─────────────────────────────────────────────
    def _detect_mediapipe(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)
        faces = []
        if not results.detections:
            return faces
        h, w = frame_bgr.shape[:2]
        for det in results.detections:
            score = det.score[0] if det.score else 0.0
            if score < self.min_confidence:
                continue
            box = det.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))

            # Extract landmarks if available
            kps = det.location_data.relative_keypoints
            if kps:
                landmarks = np.array(
                    [[kp.x * w, kp.y * h] for kp in kps], dtype=np.float32
                )
            else:
                landmarks = None

            faces.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                landmarks=landmarks,
            ))
        return faces

    def _detect_retinaface(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        results = self._detector.get(frame_bgr)
        faces = []
        for face in results:
            score = float(face.det_score)
            if score < self.min_confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            kps = face.kps.astype(np.float32) if hasattr(face, "kps") else None
            faces.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                confidence=score,
                landmarks=kps,
            ))
        return faces

    def _detect_opencv_dnn(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0),
        )
        self._detector.setInput(blob)
        detections = self._detector.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.min_confidence:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
            ))
        return sorted(faces, key=lambda f: f.confidence, reverse=True)

    # ── Helper: crop with margin ───────────────────────────────────────────────
    def _crop_face(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        h: int,
        w: int,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None
        # Add margin
        mx = int(bw * self.margin)
        my = int(bh * self.margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)
        return frame[y1:y2, x1:x2].copy()

    # ── Helper: face alignment ─────────────────────────────────────────────────
    @staticmethod
    def _align_face(
        face_crop: np.ndarray,
        landmarks: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Align face using eye landmarks (indices 0=left-eye, 1=right-eye).
        Rotates so eyes are horizontal.
        """
        if landmarks.shape[0] < 2:
            return face_crop

        # Adjust landmarks to be relative to the crop
        x1, y1 = bbox[0], bbox[1]
        left_eye = landmarks[0] - np.array([x1, y1])
        right_eye = landmarks[1] - np.array([x1, y1])

        # Compute angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        h, w = face_crop.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(face_crop, M, (w, h), flags=cv2.INTER_LINEAR)
        return aligned

    # ── Visualization ─────────────────────────────────────────────────────────
    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        faces: list[DetectedFace],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw bounding boxes and confidence scores on frame (for debugging)."""
        vis = frame_bgr.copy()
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{face.confidence:.2f}"
            cv2.putText(
                vis, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
        return vis

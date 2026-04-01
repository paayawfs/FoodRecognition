"""
Camera wrapper.
Uses Picamera2 on Raspberry Pi; falls back to OpenCV webcam on other platforms.
"""
import threading
import cv2

try:
    from picamera2 import Picamera2
    _PICAMERA_AVAILABLE = True
except ImportError:
    _PICAMERA_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PREVIEW_RESOLUTION, CAPTURE_RESOLUTION, STREAM_JPEG_QUALITY


class Camera:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._streaming = True
        if _PICAMERA_AVAILABLE:
            self.cam = Picamera2()
            self.cam.configure(
                self.cam.create_preview_configuration(
                    main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
                )
            )
            self._capture_config = self.cam.create_still_configuration(
                main={"size": CAPTURE_RESOLUTION, "format": "RGB888"}
            )
            self.cam.start()
            self._backend = "picamera2"
        else:
            self.cap = cv2.VideoCapture(0)
            self._backend = "opencv"
        print(f"Camera: using {self._backend}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Camera()
        return cls._instance

    def generate_frames(self):
        """Yield MJPEG frames (bytes) continuously."""
        while self._streaming:
            frame_bgr = self._read_frame()
            if frame_bgr is None:
                continue
            ok, buf = cv2.imencode(
                '.jpg', frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
            )
            if not ok:
                continue
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.tobytes()
                + b'\r\n'
            )

    def capture_still(self):
        """
        Capture a full-resolution still image.
        Returns BGR numpy array, or None on failure.
        """
        if self._backend == "picamera2":
            self.cam.switch_mode(self._capture_config)
            import numpy as np
            frame_rgb = self.cam.capture_array()
            self.cam.switch_mode(
                self.cam.create_preview_configuration(
                    main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
                )
            )
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

    def _read_frame(self):
        if self._backend == "picamera2":
            try:
                frame_rgb = self.cam.capture_array()
                return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

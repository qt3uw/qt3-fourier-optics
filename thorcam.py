import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

DLL_PATH = 'C:\\Users\\QT3\\repos\\qt3-fourier-optics-instructors\\dlls\\64_lib'


class ThorCam:
    """
    Minimal wrapper for a Thorlabs camera using thorlabs_tsi_sdk.

    Features:
    - open / close camera
    - get / set exposure in microseconds or milliseconds
    - snap a single image with software triggering
    - configurable timeout for slow links

    Notes:
    - Assumes the official thorlabs_tsi_sdk is installed in the active environment.
    - DLL path must point to the Thorlabs Native_64_lib directory.
    - This class is intended for use in notebooks or simple scripts, not high-speed acquisition.
    """

    def __init__(
        self,
        camera_id: str,
        dll_path: str = DLL_PATH,
        default_timeout_s: float = 15.0,
    ) -> None:
        """
        Parameters
        ----------
        dll_path
            Path to the Thorlabs Native_64_lib folder.
        camera_id
            Optional camera serial number as a string. If None, uses the first detected camera.
        default_timeout_s
            Default timeout for snap() in seconds.
        """
        self.dll_path = Path(dll_path)
        self.camera_id = camera_id
        self.default_timeout_s = float(default_timeout_s)

        self._sdk = None
        self._cam = None
        self._is_open = False

    def open(self) -> None:
        """Open the SDK and camera."""
        if self._is_open:
            return

        if not self.dll_path.exists():
            raise FileNotFoundError(f"DLL path does not exist: {self.dll_path}")

        os.add_dll_directory(str(self.dll_path))

        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

        self._sdk = TLCameraSDK()

        cameras = self._sdk.discover_available_cameras()
        if not cameras:
            self._sdk.dispose()
            self._sdk = None
            raise RuntimeError("No Thorlabs cameras detected.")

        chosen_id = self.camera_id if self.camera_id is not None else cameras[0]
        if chosen_id not in cameras:
            self._sdk.dispose()
            self._sdk = None
            raise RuntimeError(
                f"Requested camera {chosen_id!r} not found. Detected cameras: {cameras}"
            )

        self._cam = self._sdk.open_camera(chosen_id)
        self._is_open = True

        # Conservative defaults for single-frame software-triggered capture.
        self._cam.frames_per_trigger_zero_for_unlimited = 1

        # If supported, explicitly set software-triggered operation mode.
        try:
            self._cam.operation_mode = 0
        except Exception:
            pass

    def close(self) -> None:
        """Close camera and SDK cleanly."""

        if self._cam is not None:
            try:
                self._cam.disarm()
            except Exception:
                pass

            try:
                self._cam.dispose()
            except Exception:
                pass

            self._cam = None

        if self._sdk is not None:
            try:
                self._sdk.dispose()
            except Exception:
                pass

            self._sdk = None

        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def camera_id_string(self) -> str:
        self._require_open()
        return str(self._cam.serial_number)

    @property
    def sensor_shape(self) -> tuple[int, int]:
        """
        Returns
        -------
        (height, width)
        """
        self._require_open()
        return (int(self._cam.image_height_pixels), int(self._cam.image_width_pixels))

    def get_exposure_us(self) -> int:
        """Get exposure time in microseconds."""
        self._require_open()
        return int(self._cam.exposure_time_us)

    def set_exposure_us(self, exposure_us: int) -> None:
        """Set exposure time in microseconds."""
        self._require_open()
        if exposure_us <= 0:
            raise ValueError("Exposure must be positive.")
        self._cam.exposure_time_us = int(exposure_us)

    def get_exposure_ms(self) -> float:
        """Get exposure time in milliseconds."""
        return self.get_exposure_us() / 1000.0

    def set_exposure_ms(self, exposure_ms: float) -> None:
        """Set exposure time in milliseconds."""
        if exposure_ms <= 0:
            raise ValueError("Exposure must be positive.")
        self.set_exposure_us(int(round(exposure_ms * 1000.0)))


    def set_gain(self, gain:int) -> None:
        """Set gain"""
        if gain < 0:
            raise ValueError("Gain must be positive.")
        self._cam.gain = int(np.round(gain))

    def get_gain(self) -> int:
        """Get gain"""
        self._require_open()
        return self._cam.gain

    def snap(self, timeout_s: Optional[float] = None, copy: bool = True) -> np.ndarray:
        """
        Take a single software-triggered image.

        Parameters
        ----------
        timeout_s
            Maximum time to wait for a frame, in seconds.
            If None, uses self.default_timeout_s.
        copy
            If True, returns a copied NumPy array. Recommended.

        Returns
        -------
        image : np.ndarray
            2D uint16 image.

        Raises
        ------
        TimeoutError
            If no frame arrives within timeout_s.
        """
        self._require_open()

        if timeout_s is None:
            timeout_s = self.default_timeout_s
        timeout_s = float(timeout_s)

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive.")

        # This SDK call appears to block internally, so set it from timeout_s.
        self._cam.image_poll_timeout_ms = int(round(timeout_s * 1000))

        # Ensure single-frame-per-trigger behavior.
        self._cam.frames_per_trigger_zero_for_unlimited = 1
        try:
            self._cam.operation_mode = 0
        except Exception:
            pass

        self._cam.arm(2)
        try:
            self._cam.issue_software_trigger()
            frame = self._cam.get_pending_frame_or_null()

            if frame is None:
                raise TimeoutError(
                    f"No frame received within {timeout_s:.3f} s."
                )

            image = frame.image_buffer
            return np.copy(image) if copy else image

        finally:
            self._cam.disarm()

    def info(self) -> dict:
        """Return a small info dictionary useful in notebooks."""
        self._require_open()
        out = {
            "model": self._safe_get("model"),
            "name": self._safe_get("name"),
            "serial_number": self._safe_get("serial_number"),
            "firmware_version": self._safe_get("firmware_version"),
            "sensor_shape": self.sensor_shape,
            "bit_depth": self._safe_get("bit_depth"),
            "exposure_us": self.get_exposure_us(),
        }
        return out

    def _safe_get(self, attr: str):
        try:
            return getattr(self._cam, attr)
        except Exception:
            return None

    def _require_open(self) -> None:
        if not self._is_open or self._cam is None:
            raise RuntimeError("Camera is not open. Call open() first.")
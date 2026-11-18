import atexit
import threading
import time
import cv2
import numpy as np
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
from src.core.runtime.hot_reload import hot


class CaptureWindow:
    """基于 WindowsCapture 的轻量级高帧率后台窗口捕获器（零内存泄漏优化版）"""

    REGION_SCOPE = "regions"

    def __init__(self, fps: int = 30, draw_border: bool = True):
        self._lock = threading.Lock()
        self.frame_interval: float = 1.0 / max(1, fps)
        self.last_frame: np.ndarray | None = None
        self.last_capture_time: float = 0.0
        self.capture_control: InternalCaptureControl | None = None

        self.capture = WindowsCapture(
            cursor_capture=False,
            draw_border=draw_border,
            window_name=hot.get("name", scope="game_window"),
        )
        self._register_events()

        self._thread = threading.Thread(
            target=self._start_capture, name="CaptureThread", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)

    def _register_events(self) -> None:
        @self.capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            self.capture_control = capture_control
            now = time.monotonic()
            if now - self.last_capture_time < self.frame_interval:
                return

            buf = frame.frame_buffer
            if buf is None or buf.size == 0:
                return

            self.last_capture_time = now
            # 直接通过切片 [:, :, :3] 丢弃透明通道拿到 BGR 视图，不调用耗内存的 cvtColor
            bgr_view = buf[:, :, :3]

            with self._lock:
                # 关键优化：复用已分配的数组内存，避免每帧分配 6.2MB 新内存导致 _ArrayMemoryError
                if self.last_frame is None or self.last_frame.shape != bgr_view.shape:
                    self.last_frame = np.empty_like(bgr_view)
                np.copyto(self.last_frame, bgr_view)

        @self.capture.event
        def on_closed():
            print("🛑 [ScreenCapturer] 捕获会话已关闭")

    def _start_capture(self) -> None:
        try:
            self.capture.start()
        except Exception as e:
            print(f"❌ [ScreenCapturer] 捕获启动失败: {e}")

    def get_full_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.last_frame.copy() if self.last_frame is not None else None

    def get_region_frame(
        self,
        region: str | list[int | float] | tuple[int | float, ...],
        enable_offset: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, tuple[int, int]] | None:
        """获取指定区域 BGR 画面数据"""
        with self._lock:
            frame = self.last_frame

        if frame is None:
            return None

        region = (
            region
            if isinstance(region, (list, tuple))
            else hot.get(region, scope=self.REGION_SCOPE)
        )

        if not isinstance(region, (list, tuple)) or len(region) != 4:
            print(f"⚠️ [ScreenCapturer] 未找到有效的区域配置: {region}")
            return None

        h, w = frame.shape[:2]
        x1 = max(0, min(w, int(region[0])))
        y1 = max(0, min(h, int(region[1])))
        x2 = max(0, min(w, int(region[2])))
        y2 = max(0, min(h, int(region[3])))

        if x2 <= x1 or y2 <= y1:
            return None

        with self._lock:
            if self.last_frame is None:
                return None
            crop_bgr = self.last_frame[y1:y2, x1:x2].copy()
        return (crop_bgr, (x1, y1)) if enable_offset else crop_bgr

    def stop(self) -> None:
        ctrl, self.capture_control = self.capture_control, None
        if ctrl is not None:
            try:
                ctrl.stop()
            except Exception:
                pass
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    on_closed = stop

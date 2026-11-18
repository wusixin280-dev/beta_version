import logging
import threading
from typing import Any, Callable, Mapping
import numpy as np
from src.core.runtime.shared_container import SharedContainer, TaskType
from src.core.visual.capture_window import CaptureWindow
from src.core.visual.sift_matcher import SiftMatcher

logger = logging.getLogger(__name__)


class MonitorThread:
    """后台视觉监控线程:负责按需截图并分发执行模板匹配、过滤、OCR与特征识别任务"""

    def __init__(
        self,
        shared_container: SharedContainer,
        capture: CaptureWindow,
        interval: float = 0.05,
    ):
        self.shared_container = shared_container
        self.capture = capture
        self.sift_matcher = SiftMatcher()
        self.interval = interval

        self._stop_event = threading.Event()
        # 为什么这样写：必须与 SharedContainer 中的 TaskType 枚举名严格一致
        self._handlers: dict[
            TaskType, Callable[[str, np.ndarray, Mapping[str, Any]], None]
        ] = {
            TaskType.FILTER: self.execute_filter,
            TaskType.READER: self.execute_reader,
            TaskType.TEMPLATE: self.execute_template,
            TaskType.FEATURE: self.execute_sift,
        }

        self._thread = threading.Thread(
            target=self.run, name="MonitorThread", daemon=True
        )
        self._thread.start()

    def run(self) -> None:
        """后台线程主循环:持续轮询 SharedContainer 中的任务指令并执行对应视觉感知"""
        while not self._stop_event.is_set():
            try:
                task = self.shared_container.get_task()
                if task is not None:
                    # 为什么这样写：TaskObject 定义的字段是 task_type 和 task_params
                    handler = self._handlers.get(task.task_type)
                    if handler is not None:
                        name = str(task.region)
                        params = task.task_params
                        frame = self.capture.get_region_frame(
                            task.region,
                            params.get("frame_offset", False),
                        )
                        if frame is not None:
                            handler(name, frame, params)
            except Exception:
                logger.exception("❌ [MonitorThread] 执行异常")

            self._stop_event.wait(self.interval)

    def execute_filter(
        self, name: str, frame: np.ndarray, params: Mapping[str, Any]
    ) -> None:
        print(4)

    def execute_reader(
        self, name: str, frame: np.ndarray, params: Mapping[str, Any]
    ) -> None:
        print(3)

    def execute_template(
        self, name: str, frame: np.ndarray, params: Mapping[str, Any]
    ) -> None:
        print(2)

    def execute_sift(
        self, name: str, frame: np.ndarray, params: Mapping[str, Any]
    ) -> None:
        # 1. 计算 SIFT 特征匹配结果
        sift_result = self.sift_matcher.match(frame)
        # 2. 将最新结果写入共享容器，key 为区域名称（如 "small_map"）
        self.shared_container.save_sift(name, sift_result)

    def stop(self) -> None:
        """安全停止监控线程"""
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

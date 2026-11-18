import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from src.core.visual.sift_matcher import SiftMatchResult

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""

    FILTER = 1  # 图像过滤
    READER = 2  # 文字读取
    TEMPLATE = 3  # 模板匹配
    FEATURE = 4  # 特征匹配


@dataclass(frozen=True)
class TaskObject:
    """不可变任务对象，保证多线程读取时真正的只读安全"""

    region: str | tuple[float, ...]
    task_type: TaskType
    task_params: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        region: str | list[float] | tuple[float, ...],
        task_type: TaskType,
        task_params: dict[str, Any] | None = None,
    ):
        # 将 list 转为不可变的 tuple，并拷贝字典封装为只读视图
        normalized_region = tuple(region) if isinstance(region, list) else region
        copied_params = MappingProxyType(dict(task_params) if task_params else {})

        object.__setattr__(self, "region", normalized_region)
        object.__setattr__(self, "task_type", task_type)
        object.__setattr__(self, "task_params", copied_params)


class SharedContainer:
    """模块数据共享容器"""

    def __init__(self):
        self._lock = threading.Lock()

        self._current_task: TaskObject | None = None
        self._sift_data: dict[str, SiftMatchResult] = {}

    def submit_task(
        self,
        region: str | list[float] | tuple[float, ...],
        task_type: TaskType,
        task_params: dict[str, Any] | None = None,
    ) -> bool:
        """发布任务，如果任务一致直接跳过

        Args:
            region (str | list[float] | tuple[float, ...]): 截图区域
            task_type (TaskType): 任务类型
            task_params (dict[str, Any] | None, optional): 任务参数

        Returns:
            bool: 是否执行任务：重复false, 新任务true
        """
        try:
            new_task = TaskObject(
                region=region, task_type=task_type, task_params=task_params
            )
        except Exception:
            logger.exception("[共享容器模块]: 任务对象构建失败")
            return False

        with self._lock:
            # dataclass 会自动对比 region、task_type 和 task_params
            if self._current_task == new_task:
                return False
            self._sift_data.clear()
            self._current_task = new_task
            return True

    def get_task(self) -> TaskObject | None:
        """获取当前视觉任务对象（线程安全，返回不可变只读对象）"""
        with self._lock:
            return self._current_task

    def stop_task(self) -> None:
        """停止当前后台视觉任务"""
        with self._lock:
            self._current_task = None
            self._sift_data.clear()  # 可选：停止任务时同步清理残留数据

    def save_sift(self, key: str, sift_data: SiftMatchResult):
        with self._lock:
            self._sift_data[key] = sift_data
            return True

    def get_sift(self, key: str) -> SiftMatchResult | None:
        with self._lock:
            return self._sift_data.get(key)

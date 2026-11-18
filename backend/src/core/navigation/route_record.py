import math
from pathlib import Path
from enum import Enum, auto
import yaml
import logging
from src.core.runtime.hot_reload import hot
from src.core.visual.sift_matcher import SiftMatchResult

logger = logging.getLogger(__name__)
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

RESPAWN_AREA: float = 35.0
RECORD_STEP: float = 1.5  # 常规移动最小记录步长


def resolve_route_path(map_name: str) -> Path:
    """解析路线 YAML 保存路径"""
    root = Path(hot.find_project_root())
    route_rel = (
        hot.get(f"{map_name}.route", scope="maps")
        or hot.get("map_route", scope="map_configuration")
        or f"config/routes/{map_name}.yaml"
    )
    return root / str(route_rel)


class RecordState(Enum):
    IDLE = auto()  # 空闲状态
    RECORD = auto()  # 录制中


class RecordRoute:

    def __init__(self):
        self._route_file = resolve_route_path("zero_dam")
        self._record_state: RecordState = RecordState.IDLE
        self._current_points: list[dict[str, int | float | bool]] = []

    @property
    def is_valid_record(self) -> bool:
        return self._record_state != RecordState.IDLE

    @staticmethod
    def is_valid_sift(sift_result: SiftMatchResult | None) -> bool:
        return (
            sift_result is not None
            and sift_result.x is not None
            and sift_result.y is not None
            and sift_result.angle is not None
        )

    def toggle_record(self) -> bool:
        if self.is_valid_record:
            self.stop_record()
        else:
            self.start_record()
        return self.is_valid_record

    def start_record(self):
        self._current_points.clear()
        self._record_state = RecordState.RECORD
        logger.info("[RouteRecorder] 开始录制路线: %s", self._route_file)

    def stop_record(self):
        self._record_state = RecordState.IDLE
        self.save_route()
        logger.info(
            "[RouteRecorder] 停止录制，当前总点数: %d", len(self._current_points)
        )

    def save_route(self):
        """保存路线"""

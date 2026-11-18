from pathlib import Path
from typing import Any, Optional, Tuple, Union
import math
import cv2
import numpy as np


class RouteDrawer:
    """将实时定位坐标绘制在大地图上并保存为 debug_route.png"""

    def __init__(
        self,
        map_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        line_color: Tuple[int, int, int] = (0, 0, 255),  # 轨迹线颜色 (BGR: 红色)
        start_color: Tuple[int, int, int] = (0, 255, 0),  # 起点颜色 (BGR: 绿色)
        current_color: Tuple[int, int, int] = (255, 255, 0),  # 当前位置颜色 (BGR: 青色)
        line_thickness: int = 2,
        min_distance: float = 1.0,
    ):
        project_root = Path(__file__).resolve().parents[2]
        self.map_path = (
            Path(map_path)
            if map_path
            else project_root / "assets" / "maps" / "zero_dam.png"
        )
        self.output_path = (
            Path(output_path)
            if output_path
            else project_root / "assets" / "maps" / "debug_route.png"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.line_color = line_color
        self.start_color = start_color
        self.current_color = current_color
        self.line_thickness = line_thickness
        self.min_distance = min_distance

        self.base_map = self._imread(self.map_path)
        if self.base_map is None:
            raise FileNotFoundError(f"无法读取大地图文件: {self.map_path}")

        self.canvas = self.base_map.copy()
        self.points: list[Tuple[int, int]] = []
        self.last_drawn_pt: Optional[Tuple[int, int]] = None

    @staticmethod
    def _imread(path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def _imwrite(path: Path, image: np.ndarray) -> bool:
        ext = path.suffix or ".png"
        success, encoded = cv2.imencode(ext, image)
        if success:
            encoded.tofile(str(path))
        return bool(success)

    @classmethod
    def _parse_sift_result(cls, result: Any) -> Optional[Tuple[int, int]]:
        """解析 get_sift 返回的坐标结构"""
        if result is None or isinstance(result, bool):
            return None

        # 字典结构: {'x': ..., 'y': ...} 或 {'pos': (x, y)} / {'center': (x, y)}
        if isinstance(result, dict):
            if "x" in result and "y" in result:
                return cls._to_int_point(result["x"], result["y"])
            if "pos_x" in result and "pos_y" in result:
                return cls._to_int_point(result["pos_x"], result["pos_y"])
            for key in ("pos", "position", "center", "point", "coord", "location"):
                if key in result:
                    return cls._parse_sift_result(result[key])
            return None

        # 序列或数组结构: (x, y), [x, y, ...], ((x, y), score) 等
        if isinstance(result, (tuple, list, np.ndarray)):
            if len(result) == 0:
                return None
            if isinstance(result[0], (tuple, list, np.ndarray, dict)):
                return cls._parse_sift_result(result[0])
            if len(result) >= 2:
                return cls._to_int_point(result[0], result[1])
            return None

        # 具备 .x 和 .y 属性的对象
        if hasattr(result, "x") and hasattr(result, "y"):
            return cls._to_int_point(getattr(result, "x"), getattr(result, "y"))

        return None

    @staticmethod
    def _to_int_point(x: Any, y: Any) -> Optional[Tuple[int, int]]:
        if x is None or y is None:
            return None
        try:
            fx, fy = float(x), float(y)
            if math.isnan(fx) or math.isnan(fy) or math.isinf(fx) or math.isinf(fy):
                return None
            return int(round(fx)), int(round(fy))
        except (TypeError, ValueError):
            return None

    def reset(self) -> None:
        """清空轨迹并重置为初始大地图"""
        self.canvas = self.base_map.copy()
        self.points.clear()
        self.last_drawn_pt = None

    def update(self, sift_result: Any) -> bool:
        """
        根据 get_sift 的返回值实时更新轨迹并保存至 debug_route.png。
        返回 True 表示本次更新并保存了图片，False 表示无有效新坐标或坐标未变化。
        """
        pt = self._parse_sift_result(sift_result)
        if pt is None:
            return False

        h, w = self.canvas.shape[:2]
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            return False

        # 如果坐标与上一帧完全一致，跳过重绘与磁盘写入，防止死循环高频写盘
        if self.last_drawn_pt == pt:
            return False

        if not self.points:
            self.points.append(pt)
            cv2.circle(self.canvas, pt, 5, self.start_color, -1, cv2.LINE_AA)
        else:
            prev_pt = self.points[-1]
            dist = math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1])
            if dist < self.min_distance:
                return False
            cv2.line(
                self.canvas,
                prev_pt,
                pt,
                self.line_color,
                self.line_thickness,
                cv2.LINE_AA,
            )
            cv2.circle(self.canvas, pt, 2, self.line_color, -1, cv2.LINE_AA)
            self.points.append(pt)

        self.last_drawn_pt = pt

        # 在副本上标记当前最新点位置，并写入 debug_route.png
        frame = self.canvas.copy()
        cv2.circle(frame, pt, 5, self.current_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)

        return self._imwrite(self.output_path, frame)

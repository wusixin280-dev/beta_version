import cv2
import numpy as np

# 玩家箭头 HSV 颜色分割阈值
DEFAULT_LOWER = np.array([52, 110, 220], dtype=np.uint8)
DEFAULT_UPPER = np.array([52, 120, 235], dtype=np.uint8)
UPSCALE = 4  # 4倍超分插值：消除低分辨率像素网格导致的 1~2° 角度量化误差


# ==================== 4. 玩家朝向角计算器 ====================
class AngleGetter:
    """
    玩家朝向角计算器 (0°~360°, 正北为 0°, 顺时针递增)
    核心流程：4x 超分二值化 -> 距离变换剥离中心圆 + IRLS 鲁棒圆拟合 -> 箭头脊线距离变换加权求方向
    """

    def __init__(
        self,
        lower: np.ndarray = DEFAULT_LOWER,
        upper: np.ndarray = DEFAULT_UPPER,
        scale: int = UPSCALE,
    ):
        self.lower = np.asarray(lower, dtype=np.uint8)
        self.upper = np.asarray(upper, dtype=np.uint8)
        self.scale = max(1, int(scale))

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """先裁剪中心小区域，再放大并提取 HSV 二值掩码"""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        crop_r = 36  # 覆盖中心圆(pad=28)及外围箭头区域
        roi_bgr = img[
            max(0, cy - crop_r) : min(h, cy + crop_r),
            max(0, cx - crop_r) : min(w, cx + crop_r),
        ]
        if self.scale > 1:
            roi_bgr = cv2.resize(
                roi_bgr,
                (0, 0),
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_CUBIC,
            )
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower, self.upper)

    # def preprocess_image(self, img: np.ndarray) -> np.ndarray:
    #     """双三次插值放大图像并提取玩家图标 HSV 颜色掩膜"""
    #     if self.scale > 1:
    #         img = cv2.resize(
    #             img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC
    #         )
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     return cv2.inRange(hsv, self.lower, self.upper)

    @staticmethod
    def _get_mask_centroid(
        binary_mask: np.ndarray, fallback: tuple[float, float]
    ) -> tuple[float, float]:
        """利用图像矩计算二值掩膜质心，若面积为 0 则返回备用坐标"""
        m = cv2.moments(binary_mask, binaryImage=True)
        if m["m00"] > 0:
            return m["m10"] / m["m00"], m["m01"] / m["m00"]
        return fallback

    @staticmethod
    def _fit_circle_irls(
        pts: np.ndarray, init_cx: float, init_cy: float, init_r: float
    ) -> tuple[float, float, float]:
        """基于 Huber 权重的迭代重加权最小二乘 (IRLS) 圆拟合，抑制边缘局部凸起干扰"""
        if len(pts) < 12:
            return init_cx, init_cy, init_r

        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = x * x + y * y

        cx, cy, r = float(init_cx), float(init_cy), float(init_r)
        delta = max(0.5, r * 0.06)

        for _ in range(2):
            res = np.abs(np.hypot(x - cx, y - cy) - r)
            w = np.where(res <= delta, 1.0, delta / (res + 1e-6))
            try:
                sol, _, _, _ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
                nx, ny, c_val = float(sol[0]), float(sol[1]), float(sol[2])
                nr_sq = nx * nx + ny * ny + c_val
                if nr_sq > 1.0:
                    cx, cy, r = nx, ny, float(np.sqrt(nr_sq))
            except np.linalg.LinAlgError:
                break

        return cx, cy, r

    def find_central_circle(
        self, binary_img: np.ndarray
    ) -> tuple[float, float, float] | None:
        """定位中心圆并求取亚像素圆心 (cx, cy, radius)"""
        h, w = binary_img.shape[:2]
        map_cx, map_cy = w // 2, h // 2
        pad = 28 * self.scale
        x1, y1 = max(0, map_cx - pad), max(0, map_cy - pad)
        x2, y2 = min(w, map_cx + pad), min(h, map_cy + pad)

        roi = binary_img[y1:y2, x1:x2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            roi, connectivity=4
        )
        if num_labels <= 1:
            return None

        # 寻找靠近小地图中心的最大连通域作为中心圆候选
        roi_cx, roi_cy = (x2 - x1) * 0.5, (y2 - y1) * 0.5
        best_label = -1
        best_area = 0
        max_center_dist_sq = (12 * self.scale) ** 2

        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            c_x, c_y = centroids[idx]
            if (c_x - roi_cx) ** 2 + (
                c_y - roi_cy
            ) ** 2 <= max_center_dist_sq and area > best_area:
                best_area = area
                best_label = idx

        if best_label == -1 or best_area < (15 * self.scale**2):
            return None

        circle_comp = np.where(labels == best_label, 255, 0).astype(np.uint8)

        # 步骤 1：填补圆内数字 '1' 的空洞
        init_cx, init_cy = centroids[best_label]
        cv2.circle(
            circle_comp,
            (int(round(init_cx)), int(round(init_cy))),
            max(1, int(round(2.0 * self.scale))),
            255,
            -1,
        )
        contours, _ = cv2.findContours(
            circle_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            cv2.drawContours(circle_comp, contours, -1, 255, cv2.FILLED)

        # 步骤 2：距离变换找内切圆峰值，剥离外部可能粘连的方向箭头
        dist_map = cv2.distanceTransform(circle_comp, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
        if max_val <= 1.5 * self.scale:
            return None

        pk_x, pk_y = max_loc
        core_mask = (dist_map >= max_val * 0.55).astype(np.uint8) * 255
        _, core_labels, _, _ = cv2.connectedComponentsWithStats(
            core_mask, connectivity=4
        )
        target_core = core_labels[pk_y, pk_x]

        if target_core > 0:
            core_only = np.where(core_labels == target_core, 255, 0).astype(np.uint8)
            ccx, ccy = self._get_mask_centroid(core_only, (float(pk_x), float(pk_y)))
        else:
            ccx, ccy = float(pk_x), float(pk_y)

        # 用 1.08 倍内切半径截断外部粘连箭头
        yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
        peel_mask = (xx - ccx) ** 2 + (yy - ccy) ** 2 <= (max_val * 1.08) ** 2
        circle_peeled = np.where(peel_mask & (circle_comp > 0), 255, 0).astype(np.uint8)

        # 步骤 3：计算凸包修复边缘凹口，并通过 IRLS 拟合高精度亚像素圆心
        cnts, _ = cv2.findContours(
            circle_peeled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not cnts:
            return None

        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        hull_img = np.zeros_like(circle_peeled)
        cv2.drawContours(hull_img, [hull], -1, 255, cv2.FILLED)

        m_hull = cv2.moments(hull_img, binaryImage=True)
        if m_hull["m00"] <= 0:
            return None

        cx_local = m_hull["m10"] / m_hull["m00"]
        cy_local = m_hull["m01"] / m_hull["m00"]
        radius = float(np.sqrt(m_hull["m00"] / np.pi))

        hull_cnts, _ = cv2.findContours(
            hull_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if hull_cnts:
            pts = max(hull_cnts, key=len).reshape(-1, 2)
            cx_local, cy_local, radius = self._fit_circle_irls(
                pts, cx_local, cy_local, radius
            )

        return cx_local + x1, cy_local + y1, radius

    def find_arrow_direction(
        self, binary_img: np.ndarray, center: tuple[float, float], radius: float
    ) -> tuple[float, float] | None:
        """提取外围箭头连通域，通过距离变换脊线加权与锥角修剪计算方向向量 (dx, dy)"""
        h, w = binary_img.shape[:2]
        cx, cy = center

        roi_size = int(np.ceil(radius * 3.2))
        ix, iy = int(round(cx)), int(round(cy))
        x1, y1 = max(0, ix - roi_size), max(0, iy - roi_size)
        x2, y2 = min(w, ix + roi_size), min(h, iy + roi_size)

        roi = binary_img[y1:y2, x1:x2]
        local_cx, local_cy = cx - x1, cy - y1

        yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
        dist_map_r = np.hypot(xx - local_cx, yy - local_cy)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            roi, connectivity=4
        )
        center_ix = int(np.clip(round(local_cx), 0, roi.shape[1] - 1))
        center_iy = int(np.clip(round(local_cy), 0, roi.shape[0] - 1))
        circle_label = labels[center_iy, center_ix]

        best_arrow_label = -1
        best_arrow_area = 0
        min_area = max(3, int(0.8 * self.scale**2))
        max_valid_area = np.pi * (radius * 0.95) ** 2

        # 分支 A：优先寻找与中心圆自然分离的箭头连通域
        for idx in range(1, num_labels):
            if idx == circle_label:
                continue
            area = stats[idx, cv2.CC_STAT_AREA]
            if not (min_area <= area <= max_valid_area):
                continue
            dist = np.hypot(centroids[idx][0] - local_cx, centroids[idx][1] - local_cy)
            if (radius * 0.95 <= dist <= radius * 2.60) and area > best_arrow_area:
                best_arrow_area = area
                best_arrow_label = idx

        if best_arrow_label != -1:
            arrow_mask = np.where(
                (labels == best_arrow_label) & (dist_map_r <= radius * 2.60), 255, 0
            ).astype(np.uint8)
        else:
            # 分支 B (兜底)：若箭头与中心圆粘连，使用 [1.10R, 2.55R] 圆环切出箭头并取最大连通域
            ring_bin = np.where(
                (roi > 0)
                & (dist_map_r >= radius * 1.10)
                & (dist_map_r <= radius * 2.55),
                255,
                0,
            ).astype(np.uint8)
            n_ring, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(
                ring_bin, connectivity=8
            )
            if n_ring <= 1:
                return None
            best_r_label = 1 + int(np.argmax(r_stats[1:, cv2.CC_STAT_AREA]))
            if r_stats[best_r_label, cv2.CC_STAT_AREA] < min_area:
                return None
            arrow_mask = np.where(r_labels == best_r_label, 255, 0).astype(np.uint8)

        ys, xs = np.nonzero(arrow_mask)
        if len(xs) == 0:
            return None

        # 径向力臂权重 * 箭头内部距离变换脊线权重 (聚焦于最厚的箭头尖端中轴线)
        arrow_dt = cv2.distanceTransform(arrow_mask, cv2.DIST_L2, 5)
        dxs = xs.astype(np.float64) - local_cx
        dys = ys.astype(np.float64) - local_cy
        rs = np.hypot(dxs, dys)

        radial_w = np.clip((rs - radius * 0.92) / max(1.0, radius), 0.05, None) ** 2.0
        spine_w = arrow_dt[ys, xs].astype(np.float64) ** 1.5
        weights = radial_w * spine_w

        w_sum = float(np.sum(weights))
        if w_sum <= 1e-6:
            return float(np.mean(dxs)), float(np.mean(dys))

        dx0 = float(np.sum(dxs * weights)) / w_sum
        dy0 = float(np.sum(dys * weights)) / w_sum
        norm0 = np.hypot(dx0, dy0)
        if norm0 <= 1e-6:
            return None

        # 前向 38° 锥角修剪：剔除单侧翼尾粘连噪点
        ux, uy = dx0 / norm0, dy0 / norm0
        cos_theta = (dxs * ux + dys * uy) / np.maximum(rs, 1e-6)
        final_w = weights * np.clip(cos_theta - 0.788, 0.0, None)  # cos(38°) ≈ 0.788

        final_w_sum = float(np.sum(final_w))
        if final_w_sum > 1e-6:
            return (
                float(np.sum(dxs * final_w)) / final_w_sum,
                float(np.sum(dys * final_w)) / final_w_sum,
            )
        return dx0, dy0

    def get_angle(
        self, img: np.ndarray, return_binary: bool = False
    ) -> float | None | tuple[float | None, np.ndarray]:
        """计算玩家朝向角 (0 ~ 360°，正北为 0°，顺时针递增)"""
        binary_img = self.preprocess_image(img)
        circle = self.find_central_circle(binary_img)
        if circle is None:
            return (None, binary_img) if return_binary else None

        cx, cy, radius = circle
        arrow_dir = self.find_arrow_direction(binary_img, (cx, cy), radius)
        if arrow_dir is None:
            return (None, binary_img) if return_binary else None

        dx, dy = arrow_dir
        if dx * dx + dy * dy < 1e-4:
            return (None, binary_img) if return_binary else None

        angle_deg = round(float(np.degrees(np.arctan2(dx, -dy))) % 360.0, 2)
        return (angle_deg, binary_img) if return_binary else angle_deg

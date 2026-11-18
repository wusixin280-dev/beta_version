import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np
from src.core.runtime.hot_reload import hot
from src.core.visual.angle_getter import AngleGetter

# ==================== 常量与算子默认配置 ====================
SMALL_MAP_SIZE = (275, 275)  # 小地图标准裁剪尺寸 (宽, 高)

# SIFT 参数:更低对比度阈值(0.012) + 宽边缘阈值(20) + 5层金字塔,兼顾微弱等高线与小地图放大后的拉伸区域
SIFT_DETECTOR = cv2.SIFT_create(
    nfeatures=0,
    nOctaveLayers=5,
    contrastThreshold=0.012,
    edgeThreshold=20,
    sigma=1.2,
)
BF_MATCHER = cv2.BFMatcher(cv2.NORM_L2)


_CLAHE = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
_CLAHE_FINE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))


def enhance_gray(bgr_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gray)


def enhance_gray_zoomed(bgr_img: np.ndarray) -> np.ndarray:
    """针对小地图放大/特征稀疏区域的增强:细粒度 CLAHE + Unsharp Mask 边缘锐化"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe_img = _CLAHE_FINE.apply(gray)
    blurred = cv2.GaussianBlur(clahe_img, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(clahe_img, 1.6, blurred, -0.6, 0)


def get_map_paths(map_name: str = "zero_dam") -> tuple[Path, Path]:
    """从热重载配置中读取指定地图的 PNG 原图路径与 NPZ 特征库路径"""
    root = Path(hot.find_project_root())
    img_rel = hot.get(f"{map_name}.image", scope="maps") or hot.get(
        "map_png", scope="map_configuration"
    )
    npz_rel = hot.get(f"{map_name}.npz_data", scope="maps") or hot.get(
        "map_npz", scope="map_configuration"
    )

    if not img_rel or not npz_rel:
        raise ValueError(
            f"❌ 未在配置文件中找到地图 [{map_name}] 的 image 或 npz_data 路径配置"
        )

    return root / img_rel, root / npz_rel


# ==================== 1. 匹配结果数据类 ====================
@dataclass
class SiftMatchResult:
    """SIFT 定位坐标与朝向角结果封装(同时支持 .属性 与 ['键名'] 访问)"""

    x: int
    y: int
    angle: float | None

    @property
    def pos(self) -> tuple[int, int]:
        """返回 (x, y) 坐标元组"""
        return (self.x, self.y)

    def __getitem__(self, key: str):
        """兼容字典下标访问: result['x']"""
        return getattr(self, key)

    def to_dict(self) -> dict:
        """转换为普通字典"""
        return asdict(self)


# ==================== 2. 顶层匹配调度器 ====================
class SiftMatcher:
    """小地图 SIFT 定位与玩家朝向角并行识别模块"""

    def __init__(self, map_name: str = "zero_dam"):
        self.map_png, self.map_npz = get_map_paths(map_name)

        # 修复初始化顺序:必须先确保大地图特征库 .npz 已生成,再初始化 Locator 加载它
        self.preprocess_full_map()

        self.locator = Locator(self.map_npz)
        self.angle_getter = AngleGetter()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def match(
        self, small_map: np.ndarray, print_time: bool = False
    ) -> SiftMatchResult | None:
        """并行计算小地图在大地图中的 (x, y) 坐标与玩家朝向角 angle"""
        if small_map is None or small_map.size == 0:
            return None

        # 利用 OpenCV 底层释放 GIL 的特性,后台线程算朝向角,主线程算 SIFT 定位
        angle_future = self._executor.submit(self.angle_getter.get_angle, small_map)
        pos = self.locator.locate(small_map, print_time=print_time)
        if pos is None:
            return None

        angle = angle_future.result()
        return SiftMatchResult(x=pos[0], y=pos[1], angle=angle)

    def preprocess_full_map(self) -> None:
        """预处理大地图:提取全图 SIFT 特征点坐标与描述子并压缩保存为 .npz"""
        if self.map_npz.exists():
            return

        if not self.map_png.exists():
            raise FileNotFoundError(f"|PREPROCESS| ❌ 大地图文件不存在: {self.map_png}")

        print(f"|PREPROCESS| 正在读取大地图: {self.map_png}")
        big_map = cv2.imread(str(self.map_png))
        if big_map is None:
            raise FileNotFoundError(
                f"|PREPROCESS| ❌ 无法解码大地图图像: {self.map_png}"
            )

        print("|PREPROCESS| 🔍 正在进行 CLAHE 增强与特征点提取...")
        enhanced_map = enhance_gray(big_map)
        kp_big, des_big = SIFT_DETECTOR.detectAndCompute(enhanced_map, None)
        print(f"|PREPROCESS| 📊 提取特征点数量: {len(kp_big)}")

        # 仅保存定位真正需要的坐标 (kp_pt) 与描述子 (des_big),去除冗余属性以减小体积
        kp_pt = np.array([kp.pt for kp in kp_big], dtype=np.float32)

        self.map_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.map_npz,
            big_map_shape=big_map.shape[:2],
            kp_pt=kp_pt,
            des_big=des_big,
        )
        print(f"|PREPROCESS| ✔ 预处理完成,已保存至: {self.map_npz}")


# ==================== 3. 大地图坐标定位器 ====================
class Locator:
    """基于局部 ROI 追踪 + 全局回退策略的小地图定位器"""

    def __init__(self, map_npz_path: Path):
        self.map_npz_path = map_npz_path
        self.map_shape: tuple[int, int] = (0, 0)
        self.kp_big: np.ndarray | None = None
        self.des_big: np.ndarray | None = None

        # 构建环形掩膜:外圈 0.45R 过滤小地图边缘 UI 刻度,内圈 24px 遮挡中心玩家箭头
        w, h = SMALL_MAP_SIZE
        self.mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.mask, (w // 2, h // 2), int(min(w, h) * 0.45), 255, -1)
        cv2.circle(self.mask, (w // 2, h // 2), 24, 0, -1)

        # 追踪状态变量
        self.expected_scale: float | None = None
        self.last_pos: tuple[int, int] | None = None
        self.lost_count: int = 0

        self.load_preprocess_result()

    def load_preprocess_result(self) -> None:
        """加载预处理的 .npz 特征库"""
        if not self.map_npz_path.exists():
            raise FileNotFoundError(
                f"|LOCATOR| ❌ 预处理文件不存在: {self.map_npz_path}"
            )

        with np.load(self.map_npz_path, allow_pickle=False) as data:
            self.map_shape = tuple(data["big_map_shape"][:2])
            self.kp_big = data["kp_pt"].astype(np.float32)
            self.des_big = data["des_big"]

        self.expected_scale = None
        self.last_pos = None
        self.lost_count = 0
        print(f"|LOCATOR| ✔ 特征库加载成功,特征点总数: {len(self.kp_big)}")

    @staticmethod
    def _filter_knn_matches(matches, ratio: float) -> list:
        """Lowe's Ratio Test:筛选最优匹配距离显著小于次优匹配的特征点对"""
        return [
            pair[0]
            for pair in matches
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ]

    def _match_and_estimate(
        self,
        kp_query,
        des_query: np.ndarray,
        target_kp: np.ndarray,
        target_des: np.ndarray,
        is_local: bool = False,
    ) -> tuple[tuple[int, int] | None, int, float]:
        """执行 KNN 特征匹配 + RANSAC 仿射矩阵估计 + 尺度/旋转角几何校验"""
        w, h = SMALL_MAP_SIZE

        # 针对小地图放大后特征点变少的情况,在局部追踪时动态放宽匹配与内点门槛
        few_features = len(kp_query) < 25
        if is_local and few_features:
            ratio_thresh = 0.80
            min_matches = 4
            min_inliers = 3
        else:
            ratio_thresh = 0.78 if is_local else 0.74
            min_matches = 5 if is_local else 7
            min_inliers = 4 if (is_local and self.expected_scale is not None) else 6

        raw_matches = BF_MATCHER.knnMatch(des_query, target_des, k=2)
        good_matches = self._filter_knn_matches(raw_matches, ratio_thresh)

        # 局部追踪模式下,若匹配点略微不足,适度放宽一次比例阈值至 0.84
        if len(good_matches) < min_matches and is_local:
            good_matches = self._filter_knn_matches(raw_matches, 0.84)

        if len(good_matches) < min_matches:
            return None, 0, 0.0

        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = target_kp[[m.trainIdx for m in good_matches]].reshape(-1, 1, 2)

        # 使用 RANSAC 估计 4 自由度部分仿射变换 (平移 + 旋转 + 均匀缩放)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.5,
            maxIters=2000,
            confidence=0.99,
        )
        if M is None or inliers is None:
            return None, 0, 0.0

        inlier_count = int(np.sum(inliers))
        if inlier_count < min_inliers:
            return None, inlier_count, 0.0

        scale = float(np.hypot(M[0, 0], M[1, 0]))
        rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

        # 校验 1:小地图视角固定朝北,旋转角不应超过阈值 (内点较少时容许 8.5° 抖动)
        max_rot = 8.5 if inlier_count < 8 else 6.0
        if abs(rot_deg) > max_rot:
            return None, inlier_count, scale

        # 校验 2:缩放尺度合理性校验与多级缩放自适应切换
        # 小地图放大意味着 1px 小地图对应更少的大地图 px,因此下探最小 scale 至 0.15
        if not (0.15 <= scale <= 4.0):
            return None, inlier_count, scale

        use_current_scale = False
        if self.expected_scale is None:
            if inlier_count >= 8:
                self.expected_scale = scale
            use_current_scale = True
        else:
            tol = 0.16 if inlier_count < 8 else 0.12
            scale_diff = abs(scale - self.expected_scale) / self.expected_scale
            if scale_diff <= tol:
                # 尺度处于当前层级正常波动范围内,平滑更新参考尺度
                if inlier_count >= 10:
                    self.expected_scale = 0.9 * self.expected_scale + 0.1 * scale
            elif inlier_count >= 7 or (
                is_local and self.lost_count >= 3 and inlier_count >= 4
            ):
                # 【核心修复】小地图进入/离开放大区域导致尺度跳变:
                # 若当前内点数足够可靠(>=7) 或 局部已连续丢帧(>=3)且内点达标,则切换至新尺度
                self.expected_scale = scale
                use_current_scale = True
            else:
                return None, inlier_count, scale

        # 用参考尺度将所有内点反推回小地图中心,取中位数消除残余旋转误差
        inlier_mask = inliers.ravel() == 1
        inlier_src = src_pts[inlier_mask].reshape(-1, 2)
        inlier_dst = dst_pts[inlier_mask].reshape(-1, 2)
        ref_scale = (
            scale
            if (use_current_scale or self.expected_scale is None)
            else self.expected_scale
        )

        center_query = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        pred_centers = inlier_dst + (center_query - inlier_src) * ref_scale
        cx, cy = np.median(pred_centers, axis=0)
        x, y = int(round(float(cx))), int(round(float(cy)))

        map_h, map_w = self.map_shape
        if 0 <= x < map_w and 0 <= y < map_h:
            return (x, y), inlier_count, scale

        return None, inlier_count, scale

    def locate(
        self, small_map: np.ndarray, print_time: bool = False
    ) -> tuple[int, int] | None:
        """定位当前小地图在大地图中的像素坐标 (x, y)"""
        start_time = time.perf_counter()
        w, h = SMALL_MAP_SIZE

        if small_map.shape[:2] != (h, w):
            small_map = cv2.resize(small_map, (w, h))

        enhanced_gray = enhance_gray(small_map)
        kp_query, des_query = SIFT_DETECTOR.detectAndCompute(enhanced_gray, self.mask)

        # 【新增兜底】若小地图进入放大区域导致提取出的特征点过少 (<20),启用细粒度锐化增强再次提取
        if des_query is None or len(kp_query) < 20:
            zoomed_gray = enhance_gray_zoomed(small_map)
            kp_zoom, des_zoom = SIFT_DETECTOR.detectAndCompute(zoomed_gray, self.mask)
            if des_zoom is not None and len(kp_zoom) > (
                0 if kp_query is None else len(kp_query)
            ):
                kp_query, des_query = kp_zoom, des_zoom

        if des_query is None or len(kp_query) < 4:
            self.lost_count += 1
            return None

        pos = None
        inlier_count = 0
        scale = 0.0
        mode_str = "GLOBAL"

        # 策略 1 (局部极速追踪):若存在上一帧位置且连续丢失 <= 15 帧,仅在附近窗口搜索
        if self.last_pos is not None and self.lost_count <= 15:
            lx, ly = self.last_pos
            roi_radius = 320.0 + self.lost_count * 30.0
            roi_mask = (np.abs(self.kp_big[:, 0] - lx) <= roi_radius) & (
                np.abs(self.kp_big[:, 1] - ly) <= roi_radius
            )
            if np.sum(roi_mask) >= 8:
                pos, inlier_count, scale = self._match_and_estimate(
                    kp_query,
                    des_query,
                    self.kp_big[roi_mask],
                    self.des_big[roi_mask],
                    is_local=True,
                )
                if pos is not None:
                    mode_str = "LOCAL"

        # 策略 2 (全局回退搜索):局部追踪未命中或首帧定位时,扫描全图特征库
        if pos is None:
            pos, inlier_count, scale = self._match_and_estimate(
                kp_query, des_query, self.kp_big, self.des_big, is_local=False
            )
            mode_str = "GLOBAL"

        if pos is not None:
            self.last_pos = pos
            self.lost_count = 0
            if print_time:
                cost = time.perf_counter() - start_time
                print(
                    f"|LOCATOR| [{mode_str}] 耗时: {cost:.4f}s "
                    f"(内点:{inlier_count}/{len(kp_query)}, 尺度:{scale:.3f}) -> {pos}"
                )
            return pos

        self.lost_count += 1
        return None

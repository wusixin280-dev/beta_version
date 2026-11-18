# ==================== 多文件 YAML 配置热重载模块 ====================
import os
import time
import logging
import threading
from collections import deque
from typing import Any
import yaml

logger = logging.getLogger("ConfigLoader")
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_AMBIGUOUS = object()  # 用于标记存在同名冲突的后缀简写


class HotReload:
    """
    支持多目录 YAML 自动扫描、无锁原子快照读取与一级父级作用域(Scope)隔离的热更新配置器
    (BFS 预建索引,运行时 O(1) 零分配快路径读取)
    """

    __slots__ = (
        "_config_dir",
        "_exclude_dirs",
        "_lock",
        "_file_sigs",
        "_file_data",
        "_last_check_time",
        "_check_interval",
        "_snapshot",
    )

    def __init__(
        self,
        config_dir: str = "config",
        check_interval: float = 2.0,
        exclude_dirs: set[str] | None = None,
    ):
        """
        初始化多文件 YAML 热重载配置器
        :param config_dir: 相对于项目根目录的配置文件夹路径,或绝对路径(默认 'config')
        :param check_interval: 磁盘文件修改时间检查的节流间隔(秒,默认 2.0 秒)
        :param exclude_dirs: 扫描时忽略的子目录名称集合(默认忽略 maps/routes 等纯数据目录)
        """
        if os.path.isabs(config_dir):
            self._config_dir = config_dir
        else:
            self._config_dir = os.path.join(self.find_project_root(), config_dir)

        # 默认忽略存放坐标路线等非全局配置的子目录
        self._exclude_dirs: set[str] = (
            exclude_dirs
            if exclude_dirs is not None
            else {"__pycache__", "maps", "routes"}
        )

        self._lock = threading.Lock()
        self._file_sigs: dict[str, tuple[int, int]] = (
            {}
        )  # {full_path: (mtime_ns, size)}
        self._file_data: dict[str, tuple[int, str, dict[str, Any]]] = (
            {}
        )  # {full_path: (depth, rel_path, data)}
        self._last_check_time: float = 0.0
        self._check_interval: float = check_interval

        # 原子快照: (direct_cache, scope_cache)
        self._snapshot: tuple[dict[str, Any], dict[str, dict[str, Any]]] = ({}, {})

        self.reload_if_needed(force=True)

    @staticmethod
    def find_project_root() -> str:
        """向上递归定位包含 main_controller.py、config 或 .git 的项目根目录"""
        curr = os.path.abspath(os.path.dirname(__file__))
        markers = ("main_controller.py", "pyproject.toml", ".git")
        for _ in range(6):
            if any(
                os.path.exists(os.path.join(curr, m)) for m in markers
            ) or os.path.isdir(os.path.join(curr, "config")):
                return curr
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @classmethod
    def freeze(cls, obj: Any) -> Any:
        """将 list 递归转换为不可变的 tuple,防止业务层原地修改污染全局缓存快照"""
        if isinstance(obj, dict):
            return {k: cls.freeze(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return tuple(cls.freeze(item) for item in obj)
        return obj

    def reload(self) -> bool:
        """手动强制立即刷新所有配置"""
        return self.reload_if_needed(force=True)

    def reload_if_needed(self, force: bool = False) -> bool:
        """
        带节流控制与非阻塞锁的多配置文件增量热更新检测
        :param force: 是否忽略节流间隔并强制同步刷新
        :return: 本次是否触发了配置更新
        """
        now = time.monotonic()
        if not force and (now - self._last_check_time < self._check_interval):
            return False

        if not self._lock.acquire(blocking=force):
            return False

        try:
            now = time.monotonic()
            if not force and (now - self._last_check_time < self._check_interval):
                return False
            self._last_check_time = now

            if not os.path.isdir(self._config_dir):
                logger.error("❌ [ConfigLoader] 找不到配置目录: %s", self._config_dir)
                return False

            changed = False
            current_files: set[str] = set()

            for root, dirs, files in os.walk(self._config_dir):
                # 过滤隐藏目录与排除的数据目录
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".") and d not in self._exclude_dirs
                ]
                for file in files:
                    if not file.endswith((".yaml", ".yml")):
                        continue
                    full_path = os.path.join(root, file)
                    current_files.add(full_path)

                    try:
                        st = os.stat(full_path)
                        # 防御编辑器 Ctrl+S 保存瞬间文件被截断为 0 字节的竞态
                        if st.st_size == 0 and full_path in self._file_data:
                            time.sleep(0.02)
                            st = os.stat(full_path)
                        sig = (st.st_mtime_ns, st.st_size)
                    except OSError:
                        continue

                    if self._file_sigs.get(full_path) == sig:
                        continue

                    rel_path = os.path.relpath(full_path, self._config_dir).replace(
                        "\\", "/"
                    )
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            raw_data = yaml.load(f, Loader=_YAML_LOADER)

                        data = (
                            self.freeze(raw_data) if isinstance(raw_data, dict) else {}
                        )
                        depth = rel_path.count("/")
                        self._file_data[full_path] = (depth, rel_path, data)
                        self._file_sigs[full_path] = sig
                        changed = True
                        logger.info("🔄 [ConfigLoader] 已热加载配置: %s", rel_path)
                    except (yaml.YAMLError, OSError) as e:
                        self._file_sigs[full_path] = sig
                        logger.warning(
                            "⚠️ [ConfigLoader] 解析失败,已保留旧配置 (%s): %s",
                            rel_path,
                            e,
                        )

            # 清理已被删除的配置文件
            deleted_files = self._file_sigs.keys() - current_files
            if deleted_files:
                for del_path in deleted_files:
                    self._file_sigs.pop(del_path, None)
                    self._file_data.pop(del_path, None)
                changed = True

            if changed:
                self.build_index()
                return True
            return False
        finally:
            self._lock.release()

    @staticmethod
    def register_suffix(suffix_map: dict[str, Any], key: str, val: Any) -> None:
        """注册后缀简写:若同一个简写对应多个不同路径的值,则标记为歧义(_AMBIGUOUS)"""
        if key not in suffix_map:
            suffix_map[key] = val
        elif suffix_map[key] is not _AMBIGUOUS and suffix_map[key] != val:
            suffix_map[key] = _AMBIGUOUS

    def build_index(self) -> None:
        """
        分两阶段构建索引:
        阶段 1: 注册所有精确完整路径(确保完整路径永远不会被浅层后缀简写覆盖)
        阶段 2: 注册无歧义的后缀简写(若存在同名冲突如多个地图都有 'image',则禁用该简写强制使用带父级路径)
        """
        direct_cache: dict[str, Any] = {}
        scope_cache: dict[str, dict[str, Any]] = {}

        direct_suffixes: dict[str, Any] = {}
        scope_suffixes: dict[str, dict[str, Any]] = {}

        sorted_files = sorted(
            self._file_data.values(), key=lambda item: (item[0], item[1])
        )

        for _, _, data in sorted_files:
            for raw_scope, top_val in data.items():
                scope_name = str(raw_scope)
                direct_cache.setdefault(scope_name, top_val)

                if not isinstance(top_val, dict):
                    continue

                sub_cache = scope_cache.setdefault(scope_name, {})
                sub_suffixes = scope_suffixes.setdefault(scope_name, {})
                queue: deque[tuple[tuple[str, ...], dict[str, Any]]] = deque(
                    [((), top_val)]
                )

                while queue:
                    parent_parts, curr_dict = queue.popleft()
                    for raw_k, val in curr_dict.items():
                        str_k = str(raw_k)
                        rel_parts = parent_parts + (str_k,)
                        rel_key = ".".join(rel_parts)

                        # 1. 精确路径直接写入主缓存
                        sub_cache.setdefault(rel_key, val)
                        direct_cache.setdefault(f"{scope_name}.{rel_key}", val)
                        direct_cache.setdefault(rel_key, val)

                        # 2. 后缀简写先收集并检测冲突
                        for i in range(1, len(rel_parts)):
                            suffix = ".".join(rel_parts[i:])
                            self.register_suffix(sub_suffixes, suffix, val)
                            self.register_suffix(direct_suffixes, suffix, val)

                        if isinstance(val, dict):
                            queue.append((rel_parts, val))

        # 阶段 2: 仅合并非歧义且未占用精确路径的后缀简写
        for scope_name, sub_suffixes in scope_suffixes.items():
            sub_cache = scope_cache[scope_name]
            for suffix, val in sub_suffixes.items():
                if val is not _AMBIGUOUS:
                    sub_cache.setdefault(suffix, val)

        for suffix, val in direct_suffixes.items():
            if val is not _AMBIGUOUS:
                direct_cache.setdefault(suffix, val)

        self._snapshot = (direct_cache, scope_cache)

    @staticmethod
    def normalize_key(k: str) -> str:
        """统一支持点号 '.' 或斜杠 '/' 分隔符写法"""
        return k.strip().replace("\\", ".").replace("/", ".").strip(".")

    def get(self, key: str, scope: str | None = None, default: Any = None) -> Any:
        """
        根据目标键名与一级父级(Scope)快速读取配置项(O(1) 无锁读取)
        :param key: 目标配置键名或子路径(如 'name'、'small_map'、'in_game.small_map')
        :param scope: YAML 一级父级名称(如 'game_window'、'gamepad'、'regions')
        :param default: 未找到配置项时的默认返回值
        """
        self.reload_if_needed()
        direct_cache, scope_cache = self._snapshot

        if scope is not None:
            sub_cache = scope_cache.get(scope)
            if sub_cache is not None and key in sub_cache:
                return sub_cache[key]

            norm_scope = self.normalize_key(scope)
            norm_key = self.normalize_key(key)
            sub_cache = scope_cache.get(norm_scope)
            if sub_cache is not None and norm_key in sub_cache:
                return sub_cache[norm_key]
            return default

        if key in direct_cache:
            return direct_cache[key]
        return direct_cache.get(self.normalize_key(key), default)


hot = HotReload()

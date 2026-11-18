# src/core/system/game_window.py
import ctypes
import time

from src.core.runtime.hot_reload import hot

user32 = ctypes.windll.user32
shell32 = ctypes.windll.shell32

SW_RESTORE = 9


class GameWindow:
    """游戏窗口管理类"""

    def __init__(self, auto_wait: bool = True):
        self.hwnd: int = 0
        if not shell32.IsUserAnAdmin():
            print("警告: 当前未以管理员权限运行，部分窗口操作可能失效")
        if auto_wait:
            self.wait_window()

    @property
    def window_name(self) -> str | None:
        """获取游戏窗口名称"""
        return hot.get("name", scope="game_window")

    @property
    def window_class(self) -> str | None:
        """获取游戏窗口类名 (如 UnrealWindow)"""
        return hot.get("class", scope="game_window")

    def check_window(self) -> bool:
        """检查窗口句柄是否有效"""
        if self.hwnd and user32.IsWindow(self.hwnd):
            return True
        self.hwnd = user32.FindWindowW(self.window_class, self.window_name)
        return bool(self.hwnd)

    def focus_window(self) -> None:
        """将游戏窗口恢复并置于前台"""
        if not self.check_window():
            return
        if user32.IsIconic(self.hwnd):
            user32.ShowWindow(self.hwnd, SW_RESTORE)
        user32.SetForegroundWindow(self.hwnd)

    def wait_window(self, interval: float = 1.0) -> int:
        """等待游戏窗口启动并置于前台"""
        while not self.check_window():
            time.sleep(interval)
        self.focus_window()
        return self.hwnd

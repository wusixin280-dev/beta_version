# src/core/system/keyboard_hook.py
import ctypes

user32 = ctypes.windll.user32


class KeyboradHook:
    KEY_MAPPING: dict[str, int] = {
        "ESC": 0x1B,
        "F1": 0x70,
        "F2": 0x71,
        "F3": 0x72,
        "F4": 0x73,
        "F": 0x46,
        "SPACE": 0x20,
        "TAB": 0x09,
        "SHIFT": 0x10,
        "CTRL": 0x11,
        "ALT": 0x12,
    }

    def __init__(self):
        self._key_states: dict[int, bool] = {}

    def _get_vk_code(self, key: str) -> int:
        """获取按键对应的 Windows 虚拟键码"""
        upper_key = key.upper()
        if upper_key in self.KEY_MAPPING:
            return self.KEY_MAPPING[upper_key]
        if len(upper_key) == 1:
            return ord(upper_key)
        raise ValueError(f"不支持的按键: {key}, 请在 KEY_MAPPING 中添加")

    def keyboard_hook(self, keyboard: str, edge_trigger: bool = False) -> bool:
        """
        监听全局按键状态

        Args:
            keyboard: 按键名称 (如 'F', 'SPACE', 'F2', 'ESC')
            edge_trigger: 是否为边缘触发 (True 表示仅在按下瞬间返回 True)
        Returns:
            bool: 触发返回 True, 否则返回 False
        """
        vk_code = self._get_vk_code(keyboard)
        is_down = (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0
        if not edge_trigger:
            return is_down

        was_down = self._key_states.get(vk_code, False)
        self._key_states[vk_code] = is_down
        return is_down and not was_down


keyboard_hook = KeyboradHook().keyboard_hook

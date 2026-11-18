import atexit
import math
import time
import vgamepad

WALKING_MODE = 4.0  # m/s
RUNNING_MODE = 6.0  # m/s
CAMERA_ROTATION_SPEED = 20.0  # °/s, 转满一圈需要18s

LEFT_JOYSTICK_DEADZONE = 0.0
RIGHT_JOYSTICK_DEADZONE = 0.0
VIRTUAL_CURSOR_DEADZONE = 0.0
VIRTUAL_CURSOR_SPEED = 1.0


class Gamepad:

    MAPPING: dict[str, int] = {
        "UP": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
        "DOWN": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        "LEFT": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
        "RIGHT": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        "A": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A,
        "B": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "X": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_X,
        "Y": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        "LB": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        "RB": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        "LS": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
        "RS": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
        "START": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_START,
        "BACK": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        "GUIDE": vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
    }

    TRIGGERS = {"LT", "RT"}

    def __init__(
        self,
        left_deadzone: float = LEFT_JOYSTICK_DEADZONE,
        right_deadzone: float = RIGHT_JOYSTICK_DEADZONE,
        init_delay: float = 0.5,
    ):
        self.left_deadzone = left_deadzone
        self.right_deadzone = right_deadzone
        self._vgamepad = vgamepad.VX360Gamepad()
        atexit.register(self.close_gamepad)
        if init_delay > 0:
            time.sleep(init_delay)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_gamepad()

    def update(self):
        """手动提交手柄状态更新（用于多动作原子同步）"""
        self._vgamepad.update()

    def resolve_button(self, button):
        """支持传入字符串(忽略大小写)或直接传入 XUSB_BUTTON 枚举"""
        if isinstance(button, str):
            key = button.strip().upper()
            if key in self.MAPPING:
                return self.MAPPING[key]
            raise ValueError(f"未知的手柄按键名: {button}")
        return button

    @staticmethod
    def apply_radial_deadzone(
        x: float, y: float, deadzone: float
    ) -> tuple[float, float]:
        """按 2D 向量模长(径向)进行死区映射与单位圆限幅，保持推杆方向角不变"""
        mag = math.hypot(x, y)
        if mag <= 1e-5:
            return 0.0, 0.0

        clamped_mag = min(1.0, mag)
        if deadzone <= 1e-5:
            scale = clamped_mag / mag
        else:
            mapped_mag = deadzone + (1.0 - deadzone) * clamped_mag
            scale = mapped_mag / mag
        return x * scale, y * scale

    def press_button(self, button, update: bool = True):
        """按下并保持按键（兼容 'LT'/'RT' 满程按下）"""
        if isinstance(button, str) and button.strip().upper() in self.TRIGGERS:
            key = button.strip().upper()
            if key == "LT":
                self.set_left_trigger(1.0, update=update)
            else:
                self.set_right_trigger(1.0, update=update)
            return

        btn = self.resolve_button(button)
        self._vgamepad.press_button(button=btn)
        if update:
            self._vgamepad.update()

    def release_button(self, button, update: bool = True):
        """松开按键（兼容 'LT'/'RT' 松开）"""
        if isinstance(button, str) and button.strip().upper() in self.TRIGGERS:
            key = button.strip().upper()
            if key == "LT":
                self.set_left_trigger(0.0, update=update)
            else:
                self.set_right_trigger(0.0, update=update)
            return

        btn = self.resolve_button(button)
        self._vgamepad.release_button(button=btn)
        if update:
            self._vgamepad.update()

    def set_left_stick(
        self,
        x: float,
        y: float,
        apply_deadzone: bool = False,
        radial_clamp: bool = False,
        update: bool = True,
    ):
        """设置左摇杆浮点偏移量 (-1.0 ~ 1.0)"""
        x, y = float(x), float(y)
        if apply_deadzone and self.left_deadzone > 0.0:
            x, y = self.apply_radial_deadzone(x, y, self.left_deadzone)
        elif radial_clamp:
            x, y = self.apply_radial_deadzone(x, y, 0.0)

        _x = max(-1.0, min(1.0, x))
        _y = max(-1.0, min(1.0, y))
        self._vgamepad.left_joystick_float(x_value_float=_x, y_value_float=_y)
        if update:
            self._vgamepad.update()

    def set_right_stick(
        self,
        x: float,
        y: float,
        apply_deadzone: bool = False,
        radial_clamp: bool = True,
        update: bool = True,
    ):
        """设置右摇杆视角转动 (-1.0 ~ 1.0)，默认开启径向限幅防止转向角度畸变"""
        x, y = float(x), float(y)
        if apply_deadzone and self.right_deadzone > 0.0:
            x, y = self.apply_radial_deadzone(x, y, self.right_deadzone)
        elif radial_clamp:
            x, y = self.apply_radial_deadzone(x, y, 0.0)

        _x = max(-1.0, min(1.0, x))
        _y = max(-1.0, min(1.0, y))
        self._vgamepad.right_joystick_float(x_value_float=_x, y_value_float=_y)
        if update:
            self._vgamepad.update()

    def reset_all(self, update: bool = True):
        """归零所有摇杆、扳机和按键状态"""
        self._vgamepad.reset()
        if update:
            self._vgamepad.update()

    def close_gamepad(self):
        try:
            self.reset_all(update=True)
        except Exception:
            pass

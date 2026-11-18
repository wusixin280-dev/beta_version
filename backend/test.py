import time
from src.core.visual.sift_matcher import SiftMatcher
from src.core.visual.capture_window import CaptureWindow
from src.core.runtime.shared_container import SharedContainer, TaskType
from src.core.runtime.monitor_thread import MonitorThread
from src.utils.route_draw import RouteDrawer


def test_method():
    capture = CaptureWindow()
    shared_container = SharedContainer()
    MonitorThread(shared_container=shared_container, capture=capture)
    route_drawer = RouteDrawer()
    try:
        shared_container.submit_task("small_map", task_type=TaskType.FEATURE)
        time.sleep(1.5)
        while True:
            result = shared_container.get_sift("small_map")
            print(result)
            route_drawer.update(result)
            time.sleep(0.05)
    except Exception:
        pass


if __name__ == "__main__":
    test_method()

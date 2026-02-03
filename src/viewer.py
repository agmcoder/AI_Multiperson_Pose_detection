import cv2


class OpenCVViewer:
    def __init__(self, window_name: str = "YOLO Pose Gym Tracker"):
        self.window_name = window_name

    def draw_person(self, frame, pid: int, persona, origin_y: int = 80):
        cv2.putText(
            frame,
            f"ID {pid} | SQ: {persona.reps_sentadilla}",
            (20, origin_y + pid * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def draw_total(self, frame, total: int):
        cv2.putText(
            frame,
            f"TOTAL SENTADILLAS: {total}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

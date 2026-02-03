import cv2
from pose_model import PoseModel
from squat_detector import SquatDetector
from pose_service import PoseService
from viewer import OpenCVViewer


def main():
    model = PoseModel()
    detector = SquatDetector()
    service = PoseService(cooldown_frames=10)
    viewer = OpenCVViewer()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.analyze_frame(frame)

        for r in results:
            annotated_frame = r.plot()

            if r.boxes.id is None:
                continue

            for i, kp in enumerate(r.keypoints.data):
                pid = int(r.boxes.id[i])
                puntos = kp.cpu().numpy()

                persona = service.update(pid, puntos, detector)

                viewer.draw_person(annotated_frame, pid, persona)

            total = service.total_reps()
            viewer.draw_total(annotated_frame, total)

            viewer.show(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
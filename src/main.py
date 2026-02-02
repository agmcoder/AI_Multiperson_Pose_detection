import cv2
from pose_engine import PoseDetector, PersonaEstado

def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    personas = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = detector.analizar_frame(frame)

        for r in results:
            annotated_frame = r.plot()

            if r.boxes.id is None:
                continue

            for i, kp in enumerate(r.keypoints.data):
                pid = int(r.boxes.id[i])

                if pid not in personas:
                    personas[pid] = PersonaEstado()

                persona = personas[pid]
                puntos = kp.cpu().numpy()

                nuevo_estado = detector.detectar_sentadilla(puntos)

                if persona.cooldown > 0:
                    persona.cooldown -= 1

                if persona.estado == "abajo" and nuevo_estado == "arriba" and persona.cooldown == 0:
                    persona.reps_sentadilla += 1
                    persona.cooldown = 10

                persona.estado = nuevo_estado

                cv2.putText(
                    annotated_frame,
                    f"ID {pid} | SQ: {persona.reps_sentadilla}",
                    (20, 80 + pid * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2
                )

            total = sum(p.reps_sentadilla for p in personas.values())
            cv2.putText(
                annotated_frame,
                f"TOTAL SENTADILLAS: {total}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,0,255), 3
            )

            cv2.imshow("YOLO Pose Gym Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
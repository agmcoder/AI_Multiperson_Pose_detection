import torch
from ultralytics import YOLO
import os

class PersonaEstado:
    def __init__(self):
        self.estado = "arriba"
        self.reps_sentadilla = 0
        self.cooldown = 0

class PoseDetector:
    def __init__(self, model_name="yolo26n-pose.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_name

        # Auto-descarga si no existe
        if not os.path.exists(self.model_path):
            print("⬇️ Modelo no encontrado. Descargando automáticamente...")
            YOLO(self.model_path)  # ultralytics lo descarga solo

        self.model = YOLO(self.model_path).to(self.device)
        print(f"✅ Motor cargado en: {self.device.upper()}")

    def analizar_frame(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        return results

    def detectar_sentadilla(self, keypoints):
        try:
            hip_y = keypoints[24][1]
            knee_y = keypoints[26][1]
            if hip_y > knee_y - 20:
                return "abajo"
            else:
                return "arriba"
        except:
            return "arriba"

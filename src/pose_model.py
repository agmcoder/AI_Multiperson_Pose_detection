import torch
from ultralytics import YOLO
import os


class PoseModel:
    def __init__(self, model_name="yolo26x-pose.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_name

        if not os.path.exists(self.model_path):
            print("⬇️ Modelo no encontrado. Descargando automáticamente...")
            YOLO(self.model_path)

        self.model = YOLO(self.model_path).to(self.device)
        print(f"✅ Modelo cargado en: {self.device.upper()}")

    def analyze_frame(self, frame):
        """Run model tracking/inference and return results object from ultralytics."""
        results = self.model.track(frame, persist=True, verbose=False)
        return results

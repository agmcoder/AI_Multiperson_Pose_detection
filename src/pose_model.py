import os
import sys
from typing import Dict, Tuple
import torch
from ultralytics import YOLO


class PoseModel:
    """
    Detecta hardware (GPU/VRAM), recomienda tamaño de modelo y permite selección
    automática o interactiva.

    Uso:
      PoseModel(model_name: str|None = None, interactive: bool = False)

    - model_name: "small"/"medium"/"large" o ruta/alias de pesos. Si None usa
      la recomendación o la elección interactiva.
    - interactive: si True y hay TTY permite elegir entre las opciones.
    """

    MODEL_OPTIONS: Dict[str, str] = {
        "nano": "yolov8n-pose.pt",
        "small": "yolov8s-pose.pt",
        "medium": "yolov8m-pose.pt",
        "large": "yolov8l-pose.pt",
        "xlarge": "yolov8x-pose.pt",
    }

    def __init__(self, model_name: str | None = None, interactive: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        recommended, options, hw = self.recommend_model()

        # override por variable de entorno
        env_choice = os.environ.get("POSE_MODEL_SIZE")
        if model_name is None:
            if env_choice:
                model_key = env_choice if env_choice in options else recommended
                self.model_path = options.get(model_key, env_choice)
            else:
                # Si estamos en un TTY o explicitamente pedimos interactive, permitir elegir.
                if interactive or sys.stdin.isatty():
                    choice = self.select_model_interactive(options, recommended, hw)
                    self.model_path = options[choice]
                else:
                    self.model_path = options[recommended]
                    print(f"Modelo recomendado: {recommended} -> {self.model_path}")
        else:
            if model_name in self.MODEL_OPTIONS:
                self.model_path = self.MODEL_OPTIONS[model_name]
            else:
                self.model_path = model_name

        # Cargar modelo (Ultralytics descargará pesos oficiales si es un alias válido)
        self.model = None
        try:
            self.model = YOLO(self.model_path)
        except FileNotFoundError:
            # intentar cargar alternativas oficiales
            for alt in options.values():
                try:
                    self.model = YOLO(alt)
                    self.model_path = alt
                    break
                except Exception:
                    continue
            if self.model is None:
                raise
        except Exception:
            # re-raise para que el usuario vea el error
            raise

        # mover modelo al dispositivo si es aplicable
        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass

        print(f"✅ Modelo cargado en: {self.device.upper()} ({self.model_path})")

    def get_hardware_info(self) -> Dict:
        """Recoge info relevante: cuda_available, cuda_version, gpu_count, gpus (name, total_memory_gb)."""
        hw = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": 0,
            "gpus": [],
        }
        if not hw["cuda_available"]:
            return hw
        try:
            cnt = torch.cuda.device_count()
            hw["gpu_count"] = cnt
            for i in range(cnt):
                prop = torch.cuda.get_device_properties(i)
                hw["gpus"].append(
                    {
                        "index": i,
                        "name": getattr(prop, "name", f"cuda:{i}"),
                        "total_memory_gb": round(prop.total_memory / (1024 ** 3), 2),
                    }
                )
        except Exception:
            pass
        return hw

    def recommend_model(self) -> Tuple[str, Dict[str, str], Dict]:
        """
        Devuelve (recommended_key, options_dict, hardware_info).
        Recomendación basada en VRAM mínimo entre GPUs (caso multi-GPU).
        Umbrales: <4GB -> small, 4-7.5GB -> medium, >=7.5GB -> large.
        """
        options = dict(self.MODEL_OPTIONS)
        hw = self.get_hardware_info()

        if not hw["cuda_available"] or hw["gpu_count"] == 0:
            return "small", options, hw

        try:
            mems = [g["total_memory_gb"] for g in hw["gpus"] if "total_memory_gb" in g]
            vmin = min(mems) if mems else 0.0
        except Exception:
            vmin = 0.0

        if vmin < 4.0:
            rec = "small"
        elif vmin < 7.5:
            rec = "medium"
        else:
            rec = "large"

        return rec, options, hw

    def select_model_interactive(self, options: Dict[str, str], recommended: str, hw: Dict) -> str:
        """Muestra info de HW y opciones, pide elección y devuelve la clave elegida."""
        print("Hardware detectado:")
        if not hw["cuda_available"]:
            print("  - GPU: NO disponible (se usará CPU)")
        else:
            print(f"  - CUDA: {hw.get('cuda_version')}  GPUs: {hw.get('gpu_count')}")
            for g in hw.get("gpus", []):
                print(f"    - [{g['index']}] {g['name']}  VRAM: {g['total_memory_gb']} GB")
        print("\nModelos disponibles:")
        for k, v in options.items():
            mark = " (recomendada)" if k == recommended else ""
            print(f"  {k}: {v}{mark}")
        choice = input(f"Elige modelo [{recommended}]: ").strip()
        if choice == "":
            return recommended
        if choice not in options:
            print(f"Opción inválida, usando recomendada: {recommended}")
            return recommended
        return choice

    def analyze_frame(self, frame):
        """Ejecuta inferencia/tracking y devuelve el resultado de Ultralytics."""
        return self.model.track(frame, persist=True, verbose=False)
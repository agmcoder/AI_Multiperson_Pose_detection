import os
import sys
from typing import Dict, Tuple, Callable, Optional, List, Any
import torch

# loaders
def load_yolo(weight: str, device: str):
    from ultralytics import YOLO
    model = YOLO(weight)
    try:
        model = model.to(device)
    except Exception:
        pass
    return model

def load_rtmpose(weight: str, device: str):
    """
    Stub loader for RTM/RTMPose-like implementations.
    Replace the import/creation with the real API of the chosen library.
    """
    # ejemplo: from rtm_pose import RTMPose; return RTMPose(weights=weight, device=device)
    raise NotImplementedError("RTMPose loader no implementado. Sustituye load_rtmpose por el loader real.")


class PoseModel:
    """
    Soporta múltiples implementaciones de pose (YOLO, RTM...) y menú interactivo
    con navegación por flechas para seleccionar implementación + tamaño.

    model_name puede ser:
      - None -> usar recomendación o interactivo si hay TTY
      - "yolo:medium" / "rtmpose:small" -> tipo:tamaño
      - ruta_a_pesos.pt -> se pasa directo al loader por fallback
    """

    MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
        "yolo": {
            "nano": "yolov8n-pose.pt",
            "small": "yolov8s-pose.pt",
            "medium": "yolov8m-pose.pt",
            "large": "yolov8l-pose.pt",
            "xlarge": "yolov8x-pose.pt",
        },
        "rtmpose": {
            "small": "rtmpose-small.pt",
            "medium": "rtmpose-medium.pt",
            "large": "rtmpose-large.pt",
        },
    }

    LOADERS: Dict[str, Callable[[str, str], object]] = {
        "yolo": load_yolo,
        "rtmpose": load_rtmpose,
    }

    def __init__(self, model_name: Optional[str] = None, interactive: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        recommended_impl, recommended_size, hw = self.recommend_model()

        impl_choice, size_choice = self._parse_spec(model_name, recommended_impl, recommended_size)

        if model_name is None and (interactive or sys.stdin.isatty()):
            impl_choice, size_choice = self.select_impl_and_size_interactive(recommended_impl, recommended_size, hw)

        weight = self._resolve_weight(impl_choice, size_choice)

        loader = self.LOADERS.get(impl_choice)
        if loader is None:
            raise ValueError(f"No hay loader registrado para '{impl_choice}'")

        try:
            self.model = loader(weight, self.device)
            self.impl = impl_choice
            self.size = size_choice
            self.weight = weight
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo {impl_choice}:{size_choice} -> {e}")

        print(f"✅ Modelo cargado: impl={self.impl} size={self.size} weight={self.weight} device={self.device}")

    def _parse_spec(self, spec: Optional[str], rec_impl: str, rec_size: str) -> Tuple[str, str]:
        if not spec:
            return rec_impl, rec_size
        if ":" in spec:
            impl, size = spec.split(":", 1)
            impl = impl.strip().lower()
            size = size.strip().lower()
            if impl in self.MODEL_REGISTRY and size in self.MODEL_REGISTRY[impl]:
                return impl, size
        if spec.lower() in self.MODEL_REGISTRY:
            return spec.lower(), rec_size
        return rec_impl, spec

    def _resolve_weight(self, impl: str, size_or_path: str) -> str:
        impl_map = self.MODEL_REGISTRY.get(impl, {})
        if size_or_path in impl_map:
            return impl_map[size_or_path]
        return size_or_path

    def get_hardware_info(self) -> Dict[str, Any]:
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

    def recommend_model(self) -> Tuple[str, str, Dict[str, Any]]:
        hw = self.get_hardware_info()
        impl = "yolo"
        if not hw["cuda_available"] or hw["gpu_count"] == 0:
            return impl, "nano" if impl in self.MODEL_REGISTRY and "nano" in self.MODEL_REGISTRY[impl] else "small", hw

        try:
            mems = [g["total_memory_gb"] for g in hw["gpus"]]
            vmin = min(mems) if mems else 0.0
        except Exception:
            vmin = 0.0

        if vmin < 2.0:
            size = "nano" if "nano" in self.MODEL_REGISTRY.get(impl, {}) else "small"
        elif vmin < 4.0:
            size = "small"
        elif vmin < 7.5:
            size = "medium"
        elif vmin < 12.0:
            size = "large"
        else:
            size = "xlarge" if "xlarge" in self.MODEL_REGISTRY.get(impl, {}) else "large"

        return impl, size, hw

    def _curses_select(self, items: List[str], title: str, default: int = 0) -> int:
        try:
            import curses
        except Exception:
            raise RuntimeError("curses no disponible")

        def _draw(stdscr):
            curses.curs_set(0)
            stdscr.keypad(True)
            idx = default
            while True:
                stdscr.erase()
                try:
                    stdscr.addstr(0, 0, title)
                except Exception:
                    pass
                h, w = stdscr.getmaxyx()
                start_row = 2
                for i, it in enumerate(items):
                    label = f"{i+1}. {it}"
                    if start_row + i >= h - 1:
                        break
                    if i == idx:
                        stdscr.addstr(start_row + i, 2, label[:w-4], curses.A_REVERSE)
                    else:
                        stdscr.addstr(start_row + i, 2, label[:w-4])
                k = stdscr.getch()
                if k in (curses.KEY_UP, ord('k')):
                    idx = (idx - 1) % len(items)
                elif k in (curses.KEY_DOWN, ord('j')):
                    idx = (idx + 1) % len(items)
                elif k in (10, 13):
                    return idx
                elif k in (27, ord('q')):
                    return default

        try:
            import curses
            return curses.wrapper(_draw)
        except Exception:
            raise

    def select_impl_and_size_interactive(self, rec_impl: str, rec_size: str, hw: Dict[str, Any]) -> Tuple[str, str]:
        print("Hardware detectado:")
        if not hw["cuda_available"]:
            print("  - GPU: NO disponible (CPU)")
        else:
            print(f"  - CUDA: {hw.get('cuda_version')}  GPUs: {hw.get('gpu_count')}")
            for g in hw.get("gpus", []):
                print(f"    - [{g['index']}] {g['name']}  VRAM: {g['total_memory_gb']} GB")

        impls = list(self.MODEL_REGISTRY.keys())
        default_impl_idx = impls.index(rec_impl) if rec_impl in impls else 0
        impl_choice = rec_impl

        # intentar menú con flechas via curses
        try:
            if sys.stdin.isatty():
                idx = self._curses_select(impls, "Selecciona implementación (arriba/abajo, Enter):", default_impl_idx)
                impl_choice = impls[idx]
            else:
                raise RuntimeError("no tty")
        except Exception:
            print("\nImplementaciones disponibles:")
            for i, impl in enumerate(impls, start=1):
                mark = " (recomendada)" if impl == rec_impl else ""
                print(f"  {i}. {impl}{mark}")
            raw_impl = input(f"Elige implementación [1-{len(impls)}] (vacío={default_impl_idx+1}): ").strip()
            if raw_impl != "" and raw_impl.isdigit():
                i = int(raw_impl)
                if 1 <= i <= len(impls):
                    impl_choice = impls[i - 1]

        sizes = list(self.MODEL_REGISTRY[impl_choice].keys())
        default_size_idx = sizes.index(rec_size) if rec_size in sizes else 0
        size_choice = sizes[default_size_idx]

        try:
            if sys.stdin.isatty():
                idx = self._curses_select(sizes, f"Selecciona tamaño para '{impl_choice}' (arriba/abajo, Enter):", default_size_idx)
                size_choice = sizes[idx]
            else:
                raise RuntimeError("no tty")
        except Exception:
            print(f"\nTamaños disponibles para '{impl_choice}':")
            for i, s in enumerate(sizes, start=1):
                mark = " (recomendado)" if (impl_choice == rec_impl and s == rec_size) else ""
                print(f"  {i}. {s}{mark}")
            raw_size = input(f"Elige tamaño [1-{len(sizes)}] (vacío={default_size_idx+1}): ").strip()
            if raw_size != "" and raw_size.isdigit():
                i = int(raw_size)
                if 1 <= i <= len(sizes):
                    size_choice = sizes[i - 1]

        return impl_choice, size_choice

    def analyze_frame(self, frame):
        if hasattr(self.model, "track"):
            return self.model.track(frame, persist=True, verbose=False)
        if hasattr(self.model, "predict"):
            return self.model.predict(frame)
        raise RuntimeError("La API del modelo no es compatible con los wrappers estándar. Adapta analyze_frame.")
        # filepath: /home/alejandro/projects/mario/gamificacion_pose_detection/AI_Multiperson_Pose_detection/src/pose_model.py
import os
import sys
from typing import Dict, Tuple, Callable, Optional, List, Any
import torch

# loaders
def load_yolo(weight: str, device: str):
    from ultralytics import YOLO
    model = YOLO(weight)
    try:
        model = model.to(device)
    except Exception:
        pass
    return model

def load_rtmpose(weight: str, device: str):
    """
    Stub loader for RTM/RTMPose-like implementations.
    Replace the import/creation with the real API of the chosen library.
    """
    # ejemplo: from rtm_pose import RTMPose; return RTMPose(weights=weight, device=device)
    raise NotImplementedError("RTMPose loader no implementado. Sustituye load_rtmpose por el loader real.")


class PoseModel:
    """
    Soporta múltiples implementaciones de pose (YOLO, RTM...) y menú interactivo
    con navegación por flechas para seleccionar implementación + tamaño.

    model_name puede ser:
      - None -> usar recomendación o interactivo si hay TTY
      - "yolo:medium" / "rtmpose:small" -> tipo:tamaño
      - ruta_a_pesos.pt -> se pasa directo al loader por fallback
    """

    MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
        "yolo": {
            "nano": "yolov8n-pose.pt",
            "small": "yolov8s-pose.pt",
            "medium": "yolov8m-pose.pt",
            "large": "yolov8l-pose.pt",
            "xlarge": "yolov8x-pose.pt",
        },
        "rtmpose": {
            "small": "rtmpose-small.pt",
            "medium": "rtmpose-medium.pt",
            "large": "rtmpose-large.pt",
        },
    }

    LOADERS: Dict[str, Callable[[str, str], object]] = {
        "yolo": load_yolo,
        "rtmpose": load_rtmpose,
    }

    def __init__(self, model_name: Optional[str] = None, interactive: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        recommended_impl, recommended_size, hw = self.recommend_model()

        impl_choice, size_choice = self._parse_spec(model_name, recommended_impl, recommended_size)

        if model_name is None and (interactive or sys.stdin.isatty()):
            impl_choice, size_choice = self.select_impl_and_size_interactive(recommended_impl, recommended_size, hw)

        weight = self._resolve_weight(impl_choice, size_choice)

        loader = self.LOADERS.get(impl_choice)
        if loader is None:
            raise ValueError(f"No hay loader registrado para '{impl_choice}'")

        try:
            self.model = loader(weight, self.device)
            self.impl = impl_choice
            self.size = size_choice
            self.weight = weight
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo {impl_choice}:{size_choice} -> {e}")

        print(f"✅ Modelo cargado: impl={self.impl} size={self.size} weight={self.weight} device={self.device}")

    def _parse_spec(self, spec: Optional[str], rec_impl: str, rec_size: str) -> Tuple[str, str]:
        if not spec:
            return rec_impl, rec_size
        if ":" in spec:
            impl, size = spec.split(":", 1)
            impl = impl.strip().lower()
            size = size.strip().lower()
            if impl in self.MODEL_REGISTRY and size in self.MODEL_REGISTRY[impl]:
                return impl, size
        if spec.lower() in self.MODEL_REGISTRY:
            return spec.lower(), rec_size
        return rec_impl, spec

    def _resolve_weight(self, impl: str, size_or_path: str) -> str:
        impl_map = self.MODEL_REGISTRY.get(impl, {})
        if size_or_path in impl_map:
            return impl_map[size_or_path]
        return size_or_path

    def get_hardware_info(self) -> Dict[str, Any]:
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

    def recommend_model(self) -> Tuple[str, str, Dict[str, Any]]:
        hw = self.get_hardware_info()
        impl = "yolo"
        if not hw["cuda_available"] or hw["gpu_count"] == 0:
            return impl, "nano" if impl in self.MODEL_REGISTRY and "nano" in self.MODEL_REGISTRY[impl] else "small", hw

        try:
            mems = [g["total_memory_gb"] for g in hw["gpus"]]
            vmin = min(mems) if mems else 0.0
        except Exception:
            vmin = 0.0

        if vmin < 2.0:
            size = "nano" if "nano" in self.MODEL_REGISTRY.get(impl, {}) else "small"
        elif vmin < 4.0:
            size = "small"
        elif vmin < 7.5:
            size = "medium"
        elif vmin < 12.0:
            size = "large"
        else:
            size = "xlarge" if "xlarge" in self.MODEL_REGISTRY.get(impl, {}) else "large"

        return impl, size, hw

    def _curses_select(self, items: List[str], title: str, default: int = 0) -> int:
        try:
            import curses
        except Exception:
            raise RuntimeError("curses no disponible")

        def _draw(stdscr):
            curses.curs_set(0)
            stdscr.keypad(True)
            idx = default
            while True:
                stdscr.erase()
                try:
                    stdscr.addstr(0, 0, title)
                except Exception:
                    pass
                h, w = stdscr.getmaxyx()
                start_row = 2
                for i, it in enumerate(items):
                    label = f"{i+1}. {it}"
                    if start_row + i >= h - 1:
                        break
                    if i == idx:
                        stdscr.addstr(start_row + i, 2, label[:w-4], curses.A_REVERSE)
                    else:
                        stdscr.addstr(start_row + i, 2, label[:w-4])
                k = stdscr.getch()
                if k in (curses.KEY_UP, ord('k')):
                    idx = (idx - 1) % len(items)
                elif k in (curses.KEY_DOWN, ord('j')):
                    idx = (idx + 1) % len(items)
                elif k in (10, 13):
                    return idx
                elif k in (27, ord('q')):
                    return default

        try:
            import curses
            return curses.wrapper(_draw)
        except Exception:
            raise

    def select_impl_and_size_interactive(self, rec_impl: str, rec_size: str, hw: Dict[str, Any]) -> Tuple[str, str]:
        print("Hardware detectado:")
        if not hw["cuda_available"]:
            print("  - GPU: NO disponible (CPU)")
        else:
            print(f"  - CUDA: {hw.get('cuda_version')}  GPUs: {hw.get('gpu_count')}")
            for g in hw.get("gpus", []):
                print(f"    - [{g['index']}] {g['name']}  VRAM: {g['total_memory_gb']} GB")

        impls = list(self.MODEL_REGISTRY.keys())
        default_impl_idx = impls.index(rec_impl) if rec_impl in impls else 0
        impl_choice = rec_impl

        # intentar menú con flechas via curses
        try:
            if sys.stdin.isatty():
                idx = self._curses_select(impls, "Selecciona implementación (arriba/abajo, Enter):", default_impl_idx)
                impl_choice = impls[idx]
            else:
                raise RuntimeError("no tty")
        except Exception:
            print("\nImplementaciones disponibles:")
            for i, impl in enumerate(impls, start=1):
                mark = " (recomendada)" if impl == rec_impl else ""
                print(f"  {i}. {impl}{mark}")
            raw_impl = input(f"Elige implementación [1-{len(impls)}] (vacío={default_impl_idx+1}): ").strip()
            if raw_impl != "" and raw_impl.isdigit():
                i = int(raw_impl)
                if 1 <= i <= len(impls):
                    impl_choice = impls[i - 1]

        sizes = list(self.MODEL_REGISTRY[impl_choice].keys())
        default_size_idx = sizes.index(rec_size) if rec_size in sizes else 0
        size_choice = sizes[default_size_idx]

        try:
            if sys.stdin.isatty():
                idx = self._curses_select(sizes, f"Selecciona tamaño para '{impl_choice}' (arriba/abajo, Enter):", default_size_idx)
                size_choice = sizes[idx]
            else:
                raise RuntimeError("no tty")
        except Exception:
            print(f"\nTamaños disponibles para '{impl_choice}':")
            for i, s in enumerate(sizes, start=1):
                mark = " (recomendado)" if (impl_choice == rec_impl and s == rec_size) else ""
                print(f"  {i}. {s}{mark}")
            raw_size = input(f"Elige tamaño [1-{len(sizes)}] (vacío={default_size_idx+1}): ").strip()
            if raw_size != "" and raw_size.isdigit():
                i = int(raw_size)
                if 1 <= i <= len(sizes):
                    size_choice = sizes[i - 1]

        return impl_choice, size_choice

    def analyze_frame(self, frame):
        if hasattr(self.model, "track"):
            return self.model.track(frame, persist=True, verbose=False)
        if hasattr(self.model, "predict"):
            return self.model.predict(frame)
        raise RuntimeError("La API del modelo no es compatible con los wrappers estándar. Adapta analyze_frame.")
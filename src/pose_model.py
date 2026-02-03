import os
import sys
from typing import Dict, Tuple, Callable, Optional, List, Any
import torch


# ---------- RTMO helpers ----------
class RTMOWrapper:
    """Wrapper mínimo para modelos scriptados / torch.nn.Module de RTMO que reciben (B,C,H,W)."""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, frame):
        # frame: numpy HxWx3 BGR -> convert to RGB tensor 1xCxHxW float
        import numpy as np

        img = frame[:, :, ::-1].astype("float32") / 255.0  # BGR->RGB
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        return out


def _get_rtmo_cache_dir() -> str:
    from pathlib import Path
    p = Path(os.environ.get("RTMO_CACHE_DIR", Path.home() / ".cache" / "rtmo_models"))
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _download_weight_file(filename: str, dest_dir: Optional[str] = None, base_urls: Optional[List[str]] = None) -> Optional[str]:
    """Intentar descargar `filename` desde una lista de `base_urls` o desde URL por defecto.
    Devuelve la ruta local del fichero descargado o None si no se pudo.
    """
    import urllib.request
    import tempfile
    import shutil
    from pathlib import Path

    dest = Path(dest_dir or _get_rtmo_cache_dir())
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / filename
    if out.exists():
        return str(out)

    urls: List[str] = []
    env = os.environ.get("RTMO_MODEL_BASE_URL")
    if env:
        urls.extend([u.strip().rstrip("/") for u in env.split(",") if u.strip()])
    if base_urls:
        urls.extend([u.rstrip("/") for u in base_urls])

    # defaults - intentos razonables para el 'zoo crowd'
    urls.extend([
        "https://huggingface.co/ultralytics/rtmo-crowd/resolve/main",
        "https://github.com/ultralytics/rtmo/releases/download/main",
    ])

    for base in urls:
        url = f"{base}/{filename}"
        try:
            print(f"Descargando {url} ...")
            with urllib.request.urlopen(url, timeout=20) as resp:
                if getattr(resp, 'status', 200) != 200:
                    continue
                with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                    shutil.copyfileobj(resp, tmpf)
                    tmp_path = Path(tmpf.name)
                tmp_path.replace(out)
                return str(out)
        except Exception:
            continue
    return None


# ---------- Loaders ----------
def load_yolo(weight: str, device: str):
    """Cargar modelo Ultralytics YOLO; la función importa ultralytics solo cuando se necesita."""
    try:
        from ultralytics import YOLO  # import local para evitar fallo en import global
    except Exception as e:
        raise RuntimeError(f"load_yolo: ultralytics no disponible: {e}")
    try:
        model = YOLO(weight)
        try:
            model = model.to(device)
        except Exception:
            pass
        return model
    except Exception as e:
        raise RuntimeError(f"load_yolo: error cargando '{weight}': {e}")




def load_rtmo(weight: str, device: str):
    """Cargar modelos RTMO.

    - Si la librería `rtmo` está disponible, intenta `rtmo.load(...)` con varias formas del nombre
      (nombre puro, sin sufijo `.pt`, variantes con `crowd:s`, `crowd-s`, `crowd/s`).
    - Si no existe la librería o la carga por alias falla, intenta cargar un fichero local (scripted
      con `torch.jit.load` o checkpoint con `torch.load`).
    - Si no se encuentra el fichero local, intentará descargar el peso desde el "zoo crowd".
    """
    tried: List[str] = []

    # intentar usar la librería rtmo si está instalada
    try:
        import rtmo  # type: ignore
        # intentos directos con la cadena tal cual y sin extensión
        candidates: List[str] = []
        name = os.path.basename(weight)
        if name.endswith('.pt'):
            base = name[:-3]
        else:
            base = name
        candidates.append(base)
        # si comienza por 'rtmo-' quitar prefijo
        if base.startswith('rtmo-'):
            candidates.append(base[len('rtmo-'):])
        # si contiene 'crowd' y tamaño, añadimos variantes comunes
        if 'crowd' in base:
            import re

            m = re.search(r'crowd[-_:.]?([sml])$', base)
            if m:
                s = m.group(1)
                candidates.extend([f'crowd-{s}', f'crowd:{s}', f'crowd/{s}', f'crowd_{s}'])
            else:
                candidates.append('crowd')

        # intentar cargar por cada candidato
        for c in candidates:
            tried.append(f"rtmo:{c}")
            try:
                mdl = rtmo.load(c, device=device)
                print(f"✅ RTMO cargado vía rtmo.load('{c}')")
                return mdl
            except Exception:
                continue
    except Exception:
        # rtmo no está instalado o hay problemas; se seguirá con fallback a fichero
        pass

    # siguiente: si 'weight' es una ruta local existente, intentar cargar como scripted/ckpt
    if os.path.exists(weight):
        tried.append(weight)
        try:
            scripted = torch.jit.load(weight, map_location=device)
            print(f"✅ RTMO cargado desde scriptado: {weight}")
            return RTMOWrapper(scripted, device)
        except Exception:
            try:
                ckpt = torch.load(weight, map_location=device)
                if isinstance(ckpt, torch.nn.Module):
                    print(f"✅ RTMO cargado desde checkpoint nn.Module: {weight}")
                    return RTMOWrapper(ckpt, device)
            except Exception:
                pass
        raise RuntimeError(
            "load_rtmo: el archivo existe pero no es un módulo scriptado compatible. "
            "Proporciona un modelo scriptado (.pt) o instala una librería RTMO."
        )

    # intentar descargar automáticamente desde el 'zoo crowd'
    try:
        dl = _download_weight_file(weight)
    except Exception:
        dl = None

    if dl:
        tried.append(dl)
        try:
            scripted = torch.jit.load(dl, map_location=device)
            print(f"✅ RTMO descargado y cargado: {dl}")
            return RTMOWrapper(scripted, device)
        except Exception:
            try:
                ckpt = torch.load(dl, map_location=device)
                if isinstance(ckpt, torch.nn.Module):
                    print(f"✅ RTMO descargado y cargado (checkpoint): {dl}")
                    return RTMOWrapper(ckpt, device)
            except Exception:
                pass
        raise RuntimeError(f"load_rtmo: el archivo '{dl}' fue descargado pero no es un módulo scriptado compatible.")

    # no se pudo cargar ninguna variante
    raise FileNotFoundError(f"RTMO weight not found: {weight} (intentos: {', '.join(tried)})")


# ---------- PoseModel ----------
class PoseModel:
    """
    Selector/cargador de modelos de pose (yolo / rtmo).
    Soporta selección interactiva con flechas (curses) y fallback numérico por stdin.
    """

    MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
        "yolo": {
            "nano": "yolo26n-pose.pt",
            "small": "yolo26s-pose.pt",
            "medium": "yolo26m-pose.pt",
            "large": "yolo26l-pose.pt",
            "xlarge": "yolo26x-pose.pt",
        },
        "rtmo": {
            "s": "rtmo-crowd-s.pt",
            "m": "rtmo-crowd-m.pt",
            "l": "rtmo-crowd-l.pt",
            # aliases for backward compatibility
            "small": "rtmo-crowd-s.pt",
            "medium": "rtmo-crowd-m.pt",
            "large": "rtmo-crowd-l.pt",
        },
    }

    LOADERS: Dict[str, Callable[[str, str], object]] = {
        "yolo": load_yolo,
        "rtmo": load_rtmo,
    }

    def __init__(self, model_name: Optional[str] = None, interactive: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        impl_rec, size_rec, hw = self.recommend_model()

        impl_choice, size_choice = self._parse_spec(model_name, impl_rec, size_rec)

        if model_name is None and (interactive or sys.stdin.isatty()):
            impl_choice, size_choice = self.select_impl_and_size_interactive(impl_rec, size_rec, hw)

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
        # fallback: tratar spec como ruta a pesos; usar implementación recomendada
        return rec_impl, spec

    def _resolve_weight(self, impl: str, size_or_path: str) -> str:
        impl_map = self.MODEL_REGISTRY.get(impl, {})
        if size_or_path in impl_map:
            return impl_map[size_or_path]
        return size_or_path

    def get_hardware_info(self) -> Dict[str, Any]:
        hw: Dict[str, Any] = {
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
        """Recomienda impl+size basándose en VRAM mínima entre GPUs."""
        hw = self.get_hardware_info()
        impl = "yolo"
        if not hw["cuda_available"] or hw["gpu_count"] == 0:
            return impl, "nano" if "nano" in self.MODEL_REGISTRY.get(impl, {}) else "small", hw

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

    # --- curses selector (arriba/abajo + Enter). Fallback a input numérico. ---
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

        import curses
        return curses.wrapper(_draw)

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
        """Invoca la API del modelo cargado (.track o .predict)."""
        if hasattr(self.model, "track"):
            return self.model.track(frame, persist=True, verbose=False)
        if hasattr(self.model, "predict"):
            return self.model.predict(frame)
        if callable(self.model):
            return self.model(frame)
        raise RuntimeError("La API del modelo no es compatible. Adapta analyze_frame.")


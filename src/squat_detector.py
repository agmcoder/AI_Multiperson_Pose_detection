import numpy as np


class SquatDetector:
    """Detecta sentadillas a partir de keypoints.

    Diseño orientado a SOLID:
    - Single Responsibility: esta clase solo decide estado (abajo/arriba).
    - Open/Closed: umbrales configurables en init.
    - No conoce la fuente de keypoints (inversión de dependencias).
    """

    # Candidate index pairs: (hip, knee). Se prueban hasta encontrar datos válidos.
    CANDIDATES = [(24, 26), (23, 25), (11, 13), (12, 14)]

    def __init__(self, hip_knee_margin=20):
        self.hip_knee_margin = hip_knee_margin

    def detect(self, keypoints: np.ndarray) -> str:
        """Return 'abajo' or 'arriba'.

        keypoints: numpy array shape (N, ...). Expected to index as [idx][y]
        """
        if keypoints is None:
            return "arriba"

        # attempt to find a valid hip-knee pair
        hip_y = None
        knee_y = None

        for hip_idx, knee_idx in self.CANDIDATES:
            try:
                h = keypoints[hip_idx]
                k = keypoints[knee_idx]
                # handle shapes like (N,2) or (N,3)
                hy = float(h[1])
                ky = float(k[1])
            except Exception:
                continue

            if np.isfinite(hy) and np.isfinite(ky) and hy != 0 and ky != 0:
                hip_y = hy
                knee_y = ky
                break

        if hip_y is None or knee_y is None:
            return "arriba"

        # Si la cadera está por debajo (valor y mayor) que la rodilla menos margen -> 'abajo'
        if hip_y > knee_y - self.hip_knee_margin:
            return "abajo"
        return "arriba"

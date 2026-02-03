"""Compat layer: re-export newer modules for legacy imports.

`pose_engine` used to contain model and persona classes. To keep compatibility
we re-export the new implementations here.
"""

from pose_model import PoseModel as PoseDetector
from pose_service import PersonaState as PersonaEstado


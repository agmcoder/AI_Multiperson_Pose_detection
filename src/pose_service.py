from typing import Dict


class PersonaState:
    def __init__(self):
        self.estado = "arriba"
        self.reps_sentadilla = 0
        self.cooldown = 0


class PoseService:
    def __init__(self, cooldown_frames: int = 10):
        self.personas: Dict[int, PersonaState] = {}
        self.cooldown_frames = cooldown_frames

    def _ensure_person(self, pid: int) -> PersonaState:
        if pid not in self.personas:
            self.personas[pid] = PersonaState()
        return self.personas[pid]

    def update(self, pid: int, keypoints, detector) -> PersonaState:
        """Update persona state using a detector (has method detect(keypoints)).

        Returns the PersonaState for the person.
        """
        persona = self._ensure_person(pid)

        nuevo_estado = detector.detect(keypoints)

        if persona.cooldown > 0:
            persona.cooldown -= 1

        if persona.estado == "abajo" and nuevo_estado == "arriba" and persona.cooldown == 0:
            persona.reps_sentadilla += 1
            persona.cooldown = self.cooldown_frames

        persona.estado = nuevo_estado
        return persona

    def total_reps(self) -> int:
        return sum(p.reps_sentadilla for p in self.personas.values())

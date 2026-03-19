from dataclasses import dataclass


@dataclass
class DefectControlState:
    defect_score: float
    confidence: float


class DefectControlEnv:
    """
    Minimal RL environment stub for project planning and reward-design iteration.
    """

    ACTIONS = ["pass", "inspect", "reject"]

    def __init__(self):
        self.step_idx = 0
        self.state = DefectControlState(defect_score=0.5, confidence=0.5)

    def reset(self):
        self.step_idx = 0
        self.state = DefectControlState(defect_score=0.5, confidence=0.5)
        return self.state

    def reward(self, action: str, true_is_defect: bool) -> float:
        if action == "reject" and true_is_defect:
            return 2.0
        if action == "pass" and not true_is_defect:
            return 1.0
        if action == "inspect":
            return 0.2
        return -1.5

    def step(self, action: str, true_is_defect: bool):
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.step_idx += 1
        r = self.reward(action, true_is_defect)

        # Placeholder dynamics for early experimentation.
        self.state = DefectControlState(
            defect_score=max(0.0, min(1.0, self.state.defect_score + (0.1 if true_is_defect else -0.1))),
            confidence=max(0.0, min(1.0, self.state.confidence + 0.05)),
        )

        done = self.step_idx >= 20
        return self.state, r, done

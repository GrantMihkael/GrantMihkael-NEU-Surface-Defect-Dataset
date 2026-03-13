class ThresholdTuningAgent:
    def __init__(self, initial_threshold: float = 0.5) -> None:
        self.threshold = initial_threshold

    def update(self, reward: float) -> None:
        if reward > 0:
            self.threshold = min(0.95, self.threshold + 0.01)
        else:
            self.threshold = max(0.05, self.threshold - 0.01)


def main() -> None:
    agent = ThresholdTuningAgent()
    print(f"RL scaffold ready. Current threshold: {agent.threshold:.2f}")


if __name__ == "__main__":
    main()

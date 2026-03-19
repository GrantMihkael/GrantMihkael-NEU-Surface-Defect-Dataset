import random


class EpsilonGreedyAgent:
    def __init__(self, actions, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05, seed=42):
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = random.Random(seed)

    def choose_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        # Stub policy: always choose inspect when exploiting.
        return "inspect"

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

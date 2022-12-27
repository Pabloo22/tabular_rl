import numpy as np

from tabular_rl.core import RLAgent, MarkovDecisionProcess


class DynamicProgramming(RLAgent):

    def __init__(self, mdp: MarkovDecisionProcess):
        super().__init__(mdp)

    def select_action(self, state: int) -> int:
        pass

    def fit(self) -> None:
        pass

    def save_learning(self, path: str) -> None:
        pass

    def load_learning(self, path: str) -> None:
        pass


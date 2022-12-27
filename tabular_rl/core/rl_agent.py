import abc
from typing import Union

from .agent import Agent
from .tab_env import TabEnv
from .markov_decision_proccess import MarkovDecisionProcess


class RLAgent(Agent):
    """A Reinforcement Learning agent.

    A reinforcement learning agent needs to be trained on an environment to be able to select actions.
    This differs from a rule-based agent, which can be used right away.
    """

    def __init__(self, env: Union[TabEnv, MarkovDecisionProcess] = None):
        super().__init__(env)

    @abc.abstractmethod
    def fit(self) -> None:
        """Trains the agent on the given environment."""

    def save_learning(self, path: str) -> None:
        """Saves the learned parameters to a file."""
        raise NotImplementedError

    def load_learning(self, path: str) -> None:
        """Loads the learned parameters from a file."""
        raise NotImplementedError

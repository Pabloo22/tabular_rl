import abc
from typing import Union

from .agent import Agent
from .tab_env import TabEnv
from .markov_decision_proccess import MarkovDecisionProcess


class RLAgent(Agent):
    """A Reinforcement Learning agent.

    A reinforcement learning agent needs to be trained on an environment to be able to select actions.
    This differs from a rule-based agent, which can be used right away.

    Args:
        env: The environment or MDP the agent is interacting with.
    """

    def __init__(self, env: Union[TabEnv, MarkovDecisionProcess] = None):
        super().__init__(env)

    @abc.abstractmethod
    def train(self) -> None:
        """Trains the agent on the given environment."""

    @abc.abstractmethod
    def save_learning(self, path: str) -> None:
        """Saves the learned parameters to a file."""

    @abc.abstractmethod
    def load_learning(self, path: str) -> None:
        """Loads the learned parameters from a file."""

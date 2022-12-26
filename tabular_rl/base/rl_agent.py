import abc

from .agent import Agent
from .tab_env import TabEnv


class RLAgent(Agent):
    """A Reinforcement Learning agent.

    A reinforcement learning agent needs to be trained on an environment to be able to select actions.
    This differs from a rule-based agent, which can be used right away.
    """

    def __init__(self, env: TabEnv):
        super().__init__(env)
        self.initialized = False

    @abc.abstractmethod
    def fit(self):
        """Trains the agent on the given environment."""

    @abc.abstractmethod
    def save_learning(self, path: str):
        """Saves the learned parameters to a file."""

    @abc.abstractmethod
    def load_learning(self, path: str):
        """Loads the learned parameters from a file."""


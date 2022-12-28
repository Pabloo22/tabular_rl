import abc
from typing import Union

from .tab_env import TabEnv
from .markov_decision_proccess import MarkovDecisionProcess


class Agent(abc.ABC):
    """Base class for all agents."""

    def __init__(self, env: Union[TabEnv, MarkovDecisionProcess]):
        """Initializes the agent.

        Args:
            env: The environment the agent is interacting with.
        """
        self.env = env

    @abc.abstractmethod
    def select_action(self, obs: any) -> int:
        """Selects an action given an observation.

        Args:
            obs: The observation. An observation can be anything, but it is usually an integer or a tuple of integers.

        Returns:
            The action to perform as an integer in [0, n_actions).
        """
        pass

    def __call__(self, obs) -> int:
        return self.select_action(obs)

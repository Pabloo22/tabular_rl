from typing import Union

import abc

from .tab_env import TabEnv
from .markov_decision_proccess import MarkovDecisionProcess


class Agent(abc.ABC):
    """Base class for all Reinforcement Learning agents.

    A reinforcement learning agent needs to be trained in an environment to select actions. A reinforcement learning
    agent needs to be trained in an environment to select actions. This class provides an interface for all agents.
    Note that this class implements a `train` method. We recommend using a function if you want to create an agent
    that does not need training.

    Args:
        env: The environment or MDP the agent is interacting with.
    """

    def __init__(self, env: Union[TabEnv, MarkovDecisionProcess] = None):
        """Initializes the agent.

            Args:
                env: The environment or MDP the agent is interacting with.
        """
        self.env = env

    @abc.abstractmethod
    def select_action(self, obs: any) -> int:
        """Selects an action given an observation.

        The `__call__` method will call this method. This method should be overridden by all subclasses.

        Args:
            obs: The observation. An observation can be anything, but it is usually an integer or a tuple of integers.

        Returns:
            The action to perform as an integer in [0, n_actions).
        """

    @abc.abstractmethod
    def train(self) -> None:
        """Trains the agent on the given environment."""

    def __call__(self, obs) -> int:
        return self.select_action(obs)

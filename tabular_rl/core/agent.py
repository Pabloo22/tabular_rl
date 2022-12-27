import abc
from typing import Union

from .tab_env import TabEnv
from .markov_decision_proccess import MarkovDecisionProcess


class Agent(abc.ABC):

    def __init__(self, env: Union[TabEnv, MarkovDecisionProcess]):
        self.env = env

    @abc.abstractmethod
    def select_action(self, obs) -> int:
        pass

    def __call__(self, obs) -> int:
        return self.select_action(obs)

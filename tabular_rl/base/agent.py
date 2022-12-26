import abc

from .tab_env import TabEnv


class Agent(abc.ABC):

    def __init__(self, env: TabEnv):
        self.env = env

    @abc.abstractmethod
    def select_action(self, obs) -> int:
        pass

    def __call__(self, obs) -> int:
        return self.select_action(obs)

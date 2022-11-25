import abc
import gym


class TabEnv(gym.Env, metaclass=abc.ABCMeta):
    """Abstract base class for tabular environments."""

    n_states: int
    n_actions: int

    def __init__(self, n_states: int, n_actions: int):
        super(gym.Env, self).__init__()
        self.n_states: int = n_states
        self.n_actions: int = n_actions

    @abc.abstractmethod
    def step(self, action: int) -> tuple:
        """Perform an action in the environment.

        Args:
            action: The action to perform.

        Returns:
            observation: The observation of the environment.
            reward: The reward for the action.
            done: Whether the environment is done.
            stats: The game stats.
        """
        pass

    @abc.abstractmethod
    def reset(self, seed: int = None, return_info: bool = False, options: dict = None):
        """Resets the environment.

        Returns:
            The initial state or a tuple containing the initial state and the info.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def obs2int(observation) -> int:
        """Convert a state to an integer.

        Args:
            observation: The observation to convert.

        Returns:
            The integer representation of the state.
        """
        pass

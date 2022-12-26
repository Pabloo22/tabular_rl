import abc


class TabEnv(abc.ABC):
    """Abstract base class for tabular environments."""

    n_states: int
    n_actions: int

    def __init__(self):
        self.n_states = -1
        self.n_actions = -1

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

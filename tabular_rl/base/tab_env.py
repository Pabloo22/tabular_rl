import abc
import tqdm
from typing import Callable, Tuple, Union


class TabEnv(abc.ABC):
    """Abstract base class for tabular environments."""

    n_states: int
    n_actions: int

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[any, float, bool, Union[dict, None]]:
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
    def reset(self) -> any:
        """Resets the environment.

        Returns:
            The initial state or a tuple containing the initial state and the info.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def obs2int(observation: any) -> int:
        """Convert a state to an integer.

        Args:
            observation: The observation to convert.

        Returns:
            The integer representation of the state.
        """
        pass

    def render(self):
        """Renders the environment."""
        raise NotImplementedError

    def play(self, player: Callable[[any], int], verbose: bool = True) -> float:
        """Plays the game with the agent.
        Args:
            player: The agent or function to evaluate.
            verbose: Whether to print the results.

        Returns:
            The total reward.
        """
        total_reward = 0
        obs = self.reset()
        done = False
        while not done:
            action = player(obs)
            obs, reward, done, info = self.step(action)
            total_reward += reward
            if verbose:
                self.render()

        return total_reward

    def evaluate_agent(self, agent: Callable[[any], int], n_episodes: int = 10_000, use_tqdm: bool = True) -> float:
        """Evaluates the agent.
        Args:
            agent: The agent or function to evaluate.
            n_episodes: The number of episodes to evaluate the agent.
            use_tqdm: Whether to use tqdm to display the progress.

        Returns:
            The average reward.
        """
        total_reward = 0
        for _ in tqdm.tqdm(range(n_episodes), desc="Evaluating agent", disable=not use_tqdm):
            total_reward += self.play(agent, verbose=False)

        return total_reward / n_episodes

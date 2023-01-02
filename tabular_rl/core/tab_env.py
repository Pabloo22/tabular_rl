import abc
import tqdm
from typing import Callable, Tuple, Union


class TabEnv(abc.ABC):
    """Abstract base class for tabular environments."""

    n_states: int
    n_actions: int

    def __init__(self, n_states: int, n_actions: int, discount: float = 1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[any, float, bool, Union[dict, None]]:
        """Performs an action in the environment.

        Args:
            action: The action to perform.

        Returns:
            observation: The observation of the environment.
            reward: The reward for the action.
            done: Whether the environment is done.
            info: Additional information.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> any:
        """Resets the environment.

        Returns:
            The initial state or a tuple containing the initial state and the info.
        """
        pass

    @abc.abstractmethod
    def obs2int(self, observation: any) -> int:
        """Converts an observation to an integer.

        Args:
            observation: The observation to convert.

        Returns:
            The integer representation of the state.
        """
        pass

    def render(self):
        """Renders the environment."""

    def play(self, player: Callable[[any], int], verbose: bool = True) -> float:
        """Plays the game with the agent.

        Args:
            player: An Agent or function that takes an observation and returns an action.
            verbose: Whether to render each step.

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
            A dictionary with:
             - The average reward ["avg"]
            - The standard deviation of the reward ["std"]
            - The minimum reward ["min"]
            - The maximum reward ["max"]
            - Lower and upper bounds of the 95% confidence interval for the expected reward ["ci"]

        """
        avg_reward = 0
        var = 0
        mx = float("-inf")
        mn = float("inf")
        rewards = []
        for n in tqdm.trange(n_episodes, disable=not use_tqdm, desc="Evaluating agent"):
            reward = self.play(agent, verbose=False)
            rewards.append(reward)
            old_avg_reward = avg_reward
            avg_reward += (reward - avg_reward) / (n + 1)
            var = ((n - 1)*var + n*(old_avg_reward - avg_reward)**2 + (reward - avg_reward)**2) / n if n > 0 else 0
            mx = max(mx, reward)
            mn = min(mn, reward)

        std = var**0.5
        c = 1.96 * std / n_episodes**0.5
        lower = avg_reward - c
        upper = avg_reward + c

        return {"avg": avg_reward, "std": std, "min": mn, "max": mx, "ci": (lower, upper)}


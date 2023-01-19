from typing import Callable, Tuple, Union, Dict

import abc
import tqdm


class TabEnv(abc.ABC):
    """Abstract base class for tabular environments.

    This is the tabular environment in which the agent will interact. It uses the OpenAI Gym interface. For an
    environment to be considered tabular, it must have a finite number of states and actions. Actions and states are
    represented as integers from 0 to `n_actions ` - 1, and `n_states` - 1, respectively. However, we use the concept of
    "observation" which makes reference to a more meaningful state representation. For example, in the `CarRental`
    environment, the observation is a tuple of the number of cars in each location."""

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

    @abc.abstractmethod
    def reset(self) -> any:
        """Resets the environment.

        Returns:
            The initial state or a tuple containing the initial state and the info.
        """

    @abc.abstractmethod
    def obs2int(self, observation: any) -> int:
        """Converts an observation to an integer.

        Args:
            observation: The observation to convert.

        Returns:
            The integer representation of the state.
        """

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

    def evaluate_agent(self,
                       agent: Callable[[any], int],
                       n_episodes: int = 10_000,
                       show_progress_bar: bool = True) -> Dict[str, Union[float, Tuple[float, float]]]:
        """Evaluates the agent.

        Args:
            agent: The agent or function to evaluate.
            n_episodes: The number of episodes to evaluate the agent.
            show_progress_bar: Whether to use tqdm to display the progress.

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
        for n in tqdm.trange(n_episodes, disable=not show_progress_bar, desc="Evaluating agent"):
            reward = self.play(agent, verbose=False)
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


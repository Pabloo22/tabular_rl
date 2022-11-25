from gym import spaces
import numpy as np
from typing import Callable
import tqdm

from src.base import TabEnv


class CoinFlipEnv(TabEnv):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self,
                 max_n_coin_flips: int = 30,
                 cheater_prob: float = 0.5,
                 cheater_head_prob: float = 0.75,
                 render_mode: str = "ansi",
                 seed: int = None):

        # In order to have a finite search space, we set the maximum number of coin flips to a fixed value.
        # Therefore, we can compute the total number of states in the game.
        # n_states = (max_coin_flips + 1)**2 * (max_coin_flips + 2) // 2
        # If the maximum number of coin flips is 50, the total number of states is:
        # (50 + 1)^2 * (50 + 2) // 2 = 67626
        # If the maximum number of coin flips is 30, the total number of states is:
        # (30 + 1)^2 * (30 + 2) // 2 = 15376

        # Why this formula?
        # If we compute the number of possible states for each amount of coins without
        # taking into account the number of coins left, we get:
        #     - if 0 coins have been flipped: 1 state
        #     - if 1 coin has been flipped: 2 states, 0 heads or 1 head
        #     - if 2 coins have been flipped: 3 states, 0 heads, 1 head or 2 heads
        #     - if 3 coins have been flipped: 4 states, 0 heads, 1 head, 2 heads or 3 heads
        #     - etc.
        # We can see that the number of states is equal to the sum of the number of states for each amount of coins.
        # Now, if we want to take into account the number of coins left, we need to multiply the number of states
        # computed above by the sum of possible quantities of coins left. Thanks to the fact that the number of coins
        # flips is bounded, we can just consider any number of coins left between 0 and max_coin_flips. Therefore,
        # we have to multiply the aforementioned formula by (max_coin_flips + 1).

        # The total number of states if we play without taking into account the number of coins left is:
        n_states = (max_n_coin_flips + 1) * (max_n_coin_flips + 2) // 2 * (max_n_coin_flips + 1)
        super(TabEnv, self).__init__(n_states=n_states, n_actions=3)

        self.max_n_coin_flips = max_n_coin_flips
        self.cheater_prob = cheater_prob
        self.cheater_head_prob = cheater_head_prob

        # The action is an integer in the range [0, 2]:
        #   - 0: Flip a coin.
        #   - 1: label the blob as cheater.
        #   - 2: label the blob as not cheater.
        self.action_space = spaces.Discrete(3, seed=seed)

        self.random_state = np.random.RandomState(seed=seed)

        self._cheater = self.random_state.random_sample() < cheater_prob

        # An observation is a tuple of the following information:
        #   - Percentage of heads.
        #   - Number of coin flips.
        self.observation_space = spaces.Dict({
            "heads": spaces.Box(low=0, high=1, dtype=np.float32),
            "n_coin_flips": spaces.Discrete(max_n_coin_flips + 1)
        }, seed=seed)

        self._n_heads = 0
        self._n_coin_flips = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        heads = np.array([self._n_heads / self._n_coin_flips], dtype=np.float32) if self._n_coin_flips != 0 \
            else np.array([self.cheater_prob], dtype=np.float32)

        return {"heads": heads,
                "n_coin_flips": self._n_coin_flips}

    @staticmethod
    def get_obs(n_heads: int, n_coin_flips: int):
        """Returns the observation given the number of heads and coin flips.

        Args:
            n_heads: The number of heads.
            n_coin_flips: The number of coin flips.
        """
        heads = np.array([n_heads / n_coin_flips], dtype=np.float32) if n_coin_flips != 0 \
            else np.array([0.5], dtype=np.float32)

        return {"heads": heads,
                "n_coin_flips": n_coin_flips}

    def step(self, action: int):
        """Performs a step in the environment.

        Args:
            action: The action to perform.
                - 0: Flip a coin.
                - 1: label the blob as cheater.
                - 2: label the blob as fair.
        """
        info = {}
        terminated = False
        truncated = False

        if action == 0:
            if self._n_coin_flips == self.max_n_coin_flips:
                terminated = True
            else:
                flip = self.random_state.random_sample()
                self._n_coin_flips += 1
                head = True if flip < 0.5 or (self._cheater and flip < self.cheater_head_prob) else False
                if head:
                    self._n_heads += 1

            output = self._get_obs(), -1, terminated, truncated, info
        else:
            terminated = True
            label_as_cheater = True if action == 1 else False
            reward = 15 if label_as_cheater == self._cheater else -30

            output = self._get_obs(), reward, terminated, truncated, info

        return output

    def reset(self, seed: int = None, return_info: bool = False, options: dict = None):
        if seed is not None:
            self.random_state = np.random.RandomState(seed=seed)

        self._n_heads = 0
        self._n_coin_flips = 0
        self._cheater = self.random_state.random_sample() < self.cheater_prob

        return self._get_obs() if not return_info else (self._get_obs(), {})

    def render(self, terminated=False):
        if self.render_mode == "ansi":
            if not terminated:
                print(f"number of heads {self._n_heads}/{self._n_coin_flips}")
            else:
                label = "a cheater" if self._cheater else "fair"
                print(f"The bob was {label}!")

    def close(self):
        raise NotImplementedError()

    def play(self, agent: Callable[[spaces.Dict], int], n_episodes: int = 1, verbose: bool = True):
        """Plays the game with the agent.

        Args:
            agent: The agent or function to evaluate.
            n_episodes: The number of episodes to evaluate the agent.
            verbose: Whether to print the results.
        """
        total_reward = 0
        for _ in range(n_episodes):
            obs = self.reset()
            terminated = False
            while not terminated:
                action = agent(obs)
                obs, reward, terminated, truncated, info = self.step(action)
                total_reward += reward
                if verbose:
                    self.render(terminated=terminated)

        return total_reward

    def compute_avg_score(self, agent: Callable[[spaces.Dict], int], n_episodes: int = 10_000):
        """Computes the average score of the agent.

        Args:
            agent: The agent or function to evaluate.
            n_episodes: The number of episodes to evaluate the agent.
        """
        total_reward = 0
        for _ in tqdm.tqdm(range(n_episodes), desc="Computing average score"):
            obs = self.reset()
            terminated = False
            while not terminated:
                action = agent(obs)
                obs, reward, terminated, truncated, info = self.step(action)
                total_reward += reward

        return total_reward / n_episodes

    @staticmethod
    def obs2int(obs) -> int:
        """Converts an observation to an integer."""
        percentage_heads, n_coin_flips = obs["heads"][0], obs["n_coin_flips"]
        n_heads = round(percentage_heads * n_coin_flips)
        return n_heads + n_coin_flips * (n_coin_flips + 1) // 2

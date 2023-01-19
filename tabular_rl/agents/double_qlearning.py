from typing import Optional, Union

import numpy as np
import tqdm

from tabular_rl.core import TabEnv, Agent


class DoubleQLearning(Agent):
    """A Double Q-Learning based agent.

    Double Q-Learning is an algorithm that solves particular issues in Q-Learning, especially when Q-Learning can be
    tricked to take the bad action based on some positive rewards, while the expected reward of this action is
    guaranteed to be negative.
    It does that by maintaining two Q-Value lists each updating itself from the other. In short, it finds the action
    that maximizes the Q-Value in one list, but instead of using this Q-Value, it uses the action to get a Q-Value
    from the other list.

    https://towardsdatascience.com/double-q-learning-the-easy-way-a924c4085ec3

    Args:
        env: The environment to train on.
        epsilon: The probability of taking a random action during training.
        step_size: The step size for the Q-Value update.
        init_method: The method to initialize the Q-Values. Can be either 'zeros', 'uniform', 'random', or a number.
        seed: The seed for the random number generator.
        normalize_rewards: Whether to normalize the rewards to be between 0 and 1.
        max_reward: The maximum reward that can be received. Used only if normalize_rewards is True.
        min_reward: The minimum reward that can be received. Used only if normalize_rewards is True.
    """

    def __init__(self,
                 env: TabEnv,
                 epsilon: float = 0.1,
                 step_size: float = 0.1,
                 init_method: Union[str, int, float] = "zeros",
                 seed: Optional[int] = None,
                 normalize_rewards: Optional[bool] = False,
                 max_reward: float = 1.,
                 min_reward: float = -1.):

        super().__init__(env)
        self.epsilon = epsilon
        self.step_size = step_size
        self._random_state = np.random.RandomState(seed)
        self.init_method = init_method

        self.initialized = False
        self.normalize_rewards = normalize_rewards
        self.max_reward = max_reward
        self.min_reward = min_reward

        # Attributes that will be initialized when the agent is trained
        self.q_a_ = None
        self.q_b_ = None

    def _normalize_reward(self, reward: float):
        """Normalizes the reward to be between 0 and 1"""
        return (reward - self.min_reward) / (self.max_reward - self.min_reward)

    def select_action(self, obs: tuple, training: bool = False, use_a: bool = True, use_b: bool = True) -> int:
        """Selects an action given an observation.

        Args:
            obs: The observation. An observation can be anything, but it is usually an integer or a tuple of integers.
            training: Whether the agent is in training mode or not. If True, the agent will use epsilon-greedy.
            use_a: Whether to use the Q-Values from the first list. Used during training.
            use_b: Whether to use the Q-Values from the second list. Used during training.

        Returns:
            The action to perform as an integer in [0, n_actions).
        """
        state_num = self.env.obs2int(obs)
        if training and self._random_state.rand() < self.epsilon:
            action = self._random_state.randint(0, self.env.n_actions)
        elif use_a and use_b:
            q = (self.q_a_[state_num] + self.q_b_[state_num]) / 2
            action = int(np.argmax(q))
        elif use_a:
            action = int(np.argmax(self.q_a_[state_num]))
        elif use_b:
            action = int(np.argmax(self.q_b_[state_num]))
        else:
            raise ValueError("Either use_a or use_b must be True.")

        return action

    def _initialize_q_values(self):
        """Initializes the Q-Values."""
        if self.init_method == "zeros":
            self.q_a_ = np.zeros((self.env.n_states, self.env.n_actions))
            self.q_b_ = np.zeros((self.env.n_states, self.env.n_actions))
        elif self.init_method == "uniform":
            self.q_a_ = self._random_state.uniform(0, 1, (self.env.n_states, self.env.n_actions))
            self.q_b_ = self._random_state.uniform(0, 1, (self.env.n_states, self.env.n_actions))
        elif self.init_method == "random":
            self.q_a_ = self._random_state.normal(0, 1, (self.env.n_states, self.env.n_actions))
            self.q_b_ = self._random_state.normal(0, 1, (self.env.n_states, self.env.n_actions))
        elif isinstance(self.init_method, (int, float)):
            self.q_a_ = self._random_state.normal(self.init_method, 1, (self.env.n_states, self.env.n_actions))
            self.q_b_ = self._random_state.normal(self.init_method, 1, (self.env.n_states, self.env.n_actions))
        else:
            raise ValueError("init_method must be either 'zeros', 'uniform', 'random', or a number.")

        self.initialized = True

    def _update_q_values(self, state: int, action: int, reward: float, next_state: int, new_obs: int, update_a: bool):
        if update_a:
            a_star = self.select_action(new_obs, training=False, use_b=False)
            target = reward + self.env.discount * self.q_b_[next_state, a_star]
            self.q_a_[state, action] += self.step_size * (target - self.q_a_[state, action])
        else:
            b_star = self.select_action(new_obs, use_a=False)
            target = reward + self.env.discount * self.q_a_[next_state, b_star]
            self.q_b_[state, action] += self.step_size * (target - self.q_b_[state, action])

    def train(self,
              n_episodes: int = 100_000,
              eval_interval: int = 10_000,
              n_eval_episodes: int = 100,
              verbose: bool = True,
              show_progress_bar: bool = True):
        """Trains the agent on the given environment.

        Args:
            n_episodes: The number of episodes to train the agent for.
            eval_interval: The number of episodes between each evaluation.
            n_eval_episodes: The number of episodes to evaluate the agent for.
            verbose: Whether to print the evaluation results.
            show_progress_bar: Whether to use tqdm to show the progress.
        """

        if not self.initialized:
            self._initialize_q_values()

        try:
            pbar = tqdm.trange(n_episodes, disable=not show_progress_bar)
            for episode in pbar:
                obs = self.env.reset()
                state = self.env.obs2int(obs)
                done = False
                while not done:
                    action = self.select_action(obs, training=True)
                    new_obs, reward, done, _ = self.env.step(action)
                    reward = self._normalize_reward(reward) if self.normalize_rewards else reward
                    next_state = self.env.obs2int(new_obs)
                    update_a = self._random_state.rand() < 0.5
                    self._update_q_values(state, action, reward, next_state, new_obs, update_a)
                    state = next_state
                    obs = new_obs
                if episode % eval_interval == 0 and verbose:
                    print(f"Episode {episode}: {self.env.evaluate_agent(self, n_episodes=n_eval_episodes)}")
        except KeyboardInterrupt:
            pass

    def save_learning(self, path: str):
        """Saves the agent to the given path."""
        np.savez(path, q_a=self.q_a_, q_b=self.q_b_)

    def load_learning(self, path: str):
        """Loads the agent from the given path."""
        data = np.load(path)
        self.q_a_ = data["q_a"]
        self.q_b_ = data["q_b"]
        self.initialized = True

import numpy as np
import tqdm

from tabular_rl.base import TabEnv, RLAgent


class DoubleQLearning(RLAgent):
    """
    Double Q-Learning is an algorithm that solves particular issues in Q-Learning, especially when Q-Learning can be
    tricked to take the bad action based on some positive rewards, while the expected reward of this action is
    guaranteed to be negative.
    It does that by maintaining two Q-Value lists each updating itself from the other. In short it finds the action
    that maximizes the Q-Value in one list, but instead of using this Q-Value, it uses the action to get a Q-Value
    from the other list.

    https://towardsdatascience.com/double-q-learning-the-easy-way-a924c4085ec3
    """

    def __init__(self,
                 env: TabEnv,
                 epsilon: float = 0.1,
                 discount: float = 0.9,
                 step_size: float = 0.1,
                 init_method: str or int or float = "zeros",
                 seed: int = None,
                 max_reward: float = 1.0,
                 min_reward: float = -1.0):

        super().__init__(env)
        self.epsilon = epsilon
        self.discount = discount
        self.step_size = step_size
        self.random_state = np.random.RandomState(seed)
        self.init_method = init_method

        self.q_a_ = None
        self.q_b_ = None
        self.initialized = False
        self.max_reward = max_reward
        self.min_reward = min_reward

    def normalize_reward(self, reward: float):
        """Normalizes the reward to be between 0 and 1."""
        return (reward - self.min_reward) / (self.max_reward - self.min_reward)

    def select_action(self, obs: tuple, training: bool = False, use_a: bool = True, use_b: bool = True) -> int:
        """Selects an action to perform in the current state."""
        state_num = self.env.obs2int(obs)
        if training and self.random_state.rand() < self.epsilon:
            action = self.random_state.randint(0, self.env.n_actions)
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

    def initialize_q_values(self):
        """Initializes the Q-Values."""
        if self.init_method == "zeros":
            self.q_a_ = np.zeros((self.env.n_states, self.env.n_actions))
            self.q_b_ = np.zeros((self.env.n_states, self.env.n_actions))
        elif self.init_method == "uniform":
            self.q_a_ = self.random_state.uniform(0, 1, (self.env.n_states, self.env.n_actions))
            self.q_b_ = self.random_state.uniform(0, 1, (self.env.n_states, self.env.n_actions))
        elif self.init_method == "random":
            self.q_a_ = self.random_state.normal(0, 1, (self.env.n_states, self.env.n_actions))
            self.q_b_ = self.random_state.normal(0, 1, (self.env.n_states, self.env.n_actions))
        elif isinstance(self.init_method, int) or isinstance(self.init_method, float):
            self.q_a_ = self.random_state.normal(self.init_method, 1, (self.env.n_states, self.env.n_actions))
            self.q_b_ = self.random_state.normal(self.init_method, 1, (self.env.n_states, self.env.n_actions))
        else:
            raise ValueError("init_method must be either 'zeros', 'uniform', 'random', or a number.")

        self.initialized = True

    def fit(self,
            n_episodes: int = 100_000,
            eval_interval: int = 10_000,
            n_episodes_eval: int = 100,
            verbose: bool = True):
        """Trains the agent on the given environment."""

        if not self.initialized:
            self.initialize_q_values()
        for episode in tqdm.tqdm(range(n_episodes)):
            obs = self.env.reset()
            state = self.env.obs2int(obs)
            done = False
            while not done:
                action = self.select_action(obs, training=True)
                new_obs, reward, done, _ = self.env.step(action)
                reward = self.normalize_reward(reward)
                next_state = self.env.obs2int(new_obs)
                update_a = self.random_state.rand() < 0.5
                if update_a:
                    a_star = self.select_action(new_obs, training=False, use_b=False)
                    target = reward + self.discount * self.q_b_[next_state, a_star]
                    self.q_a_[state, action] += self.step_size * (target - self.q_a_[state, action])
                else:
                    b_star = self.select_action(new_obs, use_a=False)
                    target = reward + self.discount * self.q_a_[next_state, b_star]
                    self.q_b_[state, action] += self.step_size * (target - self.q_b_[state, action])
                state = next_state
                obs = new_obs
            if episode % eval_interval == 0 and verbose:
                print(f"Episode {episode}: {self.env.evaluate_agent(self, n_episodes=100)}")

    def save_learning(self, path: str):
        """Saves the agent to the given path."""
        np.savez(path, q_a=self.q_a_, q_b=self.q_b_)

    def load_learning(self, path: str):
        """Loads the agent from the given path."""
        data = np.load(path)
        self.q_a_ = data["q_a"]
        self.q_b_ = data["q_b"]
        self.initialized = True

import numpy as np
import tqdm
from tabular_rl.core import RLAgent, MarkovDecisionProcess


class DynamicProgramming(RLAgent):

    def __init__(self, env: MarkovDecisionProcess, init_method: np.ndarray = "zeros"):
        super().__init__(env)
        self.init_method = init_method
        self.initialized = False
        self.state_value_array_ = None
        self.policy_ = None
        self.q_value_array_ = None

    def select_action(self, state: int) -> int:
        pass

    def _initialize(self) -> None:

        if self.init_method == "zeros":
            self.state_value_array_ = np.zeros(self.env.n_states)
        elif isinstance(self.init_method, float) or isinstance(self.init_method, int):
            self.state_value_array_ = np.full(self.env.n_states, self.init_method)
        else:
            raise ValueError("Invalid init_method.")

        self.policy_ = np.zeros(mdp.n_states, dtype=int)

        self.initialized = True

    def _get_q_value(self, state: int, action: int) -> np.ndarray:
        # sum(self.env.transition_probabilities_matrix[action, state, :] * (
        #             self.env.immediate_reward_matrix[action, state, :] + self.env.discount * self.V0))), (
        # self.env.n_states, self.env.n_actions)
        transition_probabilities = self.env.get_transition_probabilities(state, action)
        immediate_rewards = self.env.get_immediate_reward(state, action)
        return np.sum(transition_probabilities * (immediate_rewards + self.env.discount * self.state_value_array_))

    def _policy_evaluation(self, tol: float = 0.001, n_evaluations: int = 1000) -> None:

        mask = np.fromfunction(
            np.vectorize(lambda state, action: self.policy_[state] == action),
            (self.env.n_states, self.env.n_actions), dtype=int)

        for _ in range(n_evaluations):
            self.q_value_array_: np.ndarray = np.fromfunction(
                np.vectorize(self._get_q_value), (self.env.n_states, self.env.n_actions)
            )
            old_state_value_array = self.state_value_array_.copy()
            self.state_value_array_ = np.ma.masked_array(self.q_value_array_, mask=np.logical_not(mask)).compressed()

            if np.abs(self.state_value_array_ - old_state_value_array).max() < tol:
                break

    def fit(self,
            tol: float = 0.001,
            max_policy_evaluations: int = 1,
            max_iters: int = 100_000,
            use_tqdm: bool = True) -> None:

        if not self.initialized:
            self._initialize()

        for _ in tqdm.tqdm(range(max_iters), disable=not use_tqdm):

            old_state_value_array = self.state_value_array_.copy()
            self._policy_evaluation(tol, max_policy_evaluations)

            # Policy improvement
            self.policy_ = np.argmax(self.q_value_array_, axis=1)

            if np.abs(self.state_value_array_ - old_state_value_array).max() < tol:
                break

    def save_learning(self, path: str) -> None:
        pass

    def load_learning(self, path: str) -> None:
        pass


if __name__ == '__main__':
    transition_probability_matrix = np.array([[[0.2, 0.5, 0.3],
                                               [0, 0.5, 0.5],
                                               [0, 0, 1]],

                                              [[0.3, 0.6, 0.1],
                                          [0.1, 0.6, 0.3],
                                          [0.05, 0.4, 0.55]]])

    reward_function = np.array([[[7, 6, 6],
                                 [0, 5, 1],
                                 [0, 0, -1]],

                                [[6, 6, -1],
                                 [7, 4, 0],
                                 [6, 3, -2]]])

    mdp = MarkovDecisionProcess(3, 2, transition_probability_matrix, reward_function, 1)
    agent = DynamicProgramming(mdp)
    agent.fit()
    print(agent.policy_, agent.state_value_array_)

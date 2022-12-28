import numpy as np

from tabular_rl.core import MarkovDecisionProcess
from tabular_rl.envs import JacksRental


class JacksRentalMDP(MarkovDecisionProcess):

    def __init__(self, jack_rental_env: JacksRental, discount: float = 0.99):

        super().__init__(n_states=jack_rental_env.n_states,
                         n_actions=jack_rental_env.n_actions,
                         discount=discount)

        self.jack_rental_env = jack_rental_env

        self._transition_matrix_without_action = None
        self._reward_matrix_without_action = None

        self._initialize_matrices()

    def _initialize_matrices(self) -> None:
        pass

    def _get_new_state(self, state: int, action: int) -> int:
        pass

    def get_transition_probabilities(self, state: int, action: int) -> np.ndarray:
        new_state = self._get_new_state(state, action)
        return self._transition_matrix_without_action[new_state, :]

    def get_immediate_reward(self, state: int, action: int) -> np.ndarray:
        new_state = self._get_new_state(state, action)
        return self._reward_matrix_without_action[new_state, :]

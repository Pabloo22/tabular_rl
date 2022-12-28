from typing import Tuple

import numpy as np
import scipy.stats as stats
from functools import partial

from tabular_rl.core import MarkovDecisionProcess
from tabular_rl.envs import JacksRentalEnv


class JacksRentalMDP(MarkovDecisionProcess):

    def __init__(self, jacks_rental_env: JacksRentalEnv, discount: float = 0.99):

        super().__init__(n_states=jacks_rental_env.n_states,
                         n_actions=jacks_rental_env.n_actions,
                         discount=discount)

        self.jacks_rental_env = jacks_rental_env

        self._transition_matrix_without_action = None
        self._reward_matrix_without_action = None

        self._initialize_matrices()

    def _get_probability_and_expected_reward(self,
                                             n_cars,
                                             n_cars_next,
                                             location: int) -> Tuple[float]:
        """Returns the probability and expected reward if a day starts with `n_cars` cars at `location` and ends
        with `n_cars_next` cars."""

        n_cars, n_cars_next = int(n_cars), int(n_cars_next)
        expected_rental_requests = self.jacks_rental_env.expected_rental_requests[location]
        expected_rental_returns = self.jacks_rental_env.expected_rental_returns[location]

        transition_prob = 0
        expected_reward = 0
        c = n_cars_next - n_cars
        for n_requests in range(n_cars):
            arrival_prob = stats.poisson.pmf(n_requests + c, expected_rental_returns)
            requests_prob = stats.poisson.pmf(n_requests, expected_rental_requests)
            transition_prob += arrival_prob * requests_prob
            expected_reward += requests_prob * self.jacks_rental_env.rental_credit * n_requests

        arrival_prob = stats.poisson.pmf(n_cars + c, expected_rental_returns)
        # Probability of more than `n_cars` requests
        requests_prob = 1 - stats.poisson.cdf(n_cars - 1, expected_rental_requests)

        transition_prob += arrival_prob * requests_prob
        expected_reward += requests_prob * self.jacks_rental_env.rental_credit * n_cars

        return transition_prob, expected_reward

    def _initialize_matrices(self) -> None:
        self._transition_matrix_without_action = np.zeros((self.n_states, self.n_states))
        self._reward_matrix_without_action = np.zeros((self.n_states, self.n_states))

        transition_probs1, expected_rewards1 = np.fromfunction(
            np.vectorize(partial(self._get_probability_and_expected_reward, location=0)),
            (self.jacks_rental_env.max_n_cars + 1, self.jacks_rental_env.max_n_cars + 1),
        )
        transition_probs2, expected_rewards2 = np.fromfunction(
            np.vectorize(partial(self._get_probability_and_expected_reward, location=1)),
            (self.jacks_rental_env.max_n_cars + 1, self.jacks_rental_env.max_n_cars + 1),
        )

        for state in range(self.n_states):
            cars_first_location, cars_second_location = self.jacks_rental_env.int2obs(state)
            for new_state in range(self.n_states):
                new_cars_first_location, new_cars_second_location = self.jacks_rental_env.int2obs(new_state)

                probability_loc1 = transition_probs1[cars_first_location, new_cars_first_location]
                expected_reward_loc1 = expected_rewards1[cars_first_location, new_cars_first_location]
                probability_loc2 = transition_probs2[cars_second_location, new_cars_second_location]
                expected_reward_loc2 = expected_rewards2[cars_second_location, new_cars_second_location]

                self._transition_matrix_without_action[state, new_state] = probability_loc1 * probability_loc2
                self._reward_matrix_without_action[state, new_state] = expected_reward_loc1 + expected_reward_loc2

    def _get_new_state(self, state: int, action: int) -> int:
        car_tuple = self.jacks_rental_env.int2obs(state)
        new_car_tuple = self.jacks_rental_env.move_cars(list(car_tuple), action)
        return self.jacks_rental_env.obs2int(new_car_tuple)

    def get_transition_probabilities(self, state: int, action: int) -> np.ndarray:
        new_state = self._get_new_state(state, action)
        return self._transition_matrix_without_action[new_state, :]

    def get_immediate_reward(self, state: int, action: int) -> np.ndarray:
        new_state = self._get_new_state(state, action)
        n_moves = abs(action - self.jacks_rental_env.max_moves)
        return self._reward_matrix_without_action[new_state, :] - n_moves * self.jacks_rental_env.move_cost

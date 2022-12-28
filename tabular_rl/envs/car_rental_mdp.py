from typing import Tuple, Union

import numpy as np
import scipy.stats as stats
from functools import partial

from tabular_rl.core import MarkovDecisionProcess
from tabular_rl.envs import CarRentalEnv


class CarRentalMDP(MarkovDecisionProcess):
    """The specific MDP for the Jack's Rental problem.

    This class inherits from `MarkovDecisionProcess` and implements the methods `get_transition_probabilities` and
    `get_expected_rewards`. This allows us to save memory by not storing the transition and reward matrices. Instead, we
    compute a transition and reward matrices that do not take into account the action. This can be done because the
    transition and reward matrices are the same for many actions and states. For example, if we start with 5 cars at
    both locations and move 2 cars from the first location to the second location, the transition matrix
    for this action is the same as if we started with 3 cars at first location and 7 cars at the second location, and
    we do not move any cars. This is because at the start of the day, the number of cars at each location is the
    same in both cases.
    In the case of the expected reward, it would be the same but, in the first case, we would need
    to add the cost of moving 2 cars. Thus, is enough to compute only the transition and reward matrices for the
    states without taking into account the action and then add the cost of moving cars when computing the expected
    reward.
    """

    def __init__(self, car_rental_env: CarRentalEnv, discount: float = 0.99):

        super().__init__(n_states=car_rental_env.n_states,
                         n_actions=car_rental_env.n_actions,
                         discount=discount)

        self.car_rental_env = car_rental_env

        self._transition_matrix_without_action = None
        self._reward_matrix_without_action = None

        self._initialize_matrices()

    def _get_probability_and_expected_reward(self,
                                             n_cars: Union[int, float, np.ndarray],
                                             n_cars_next: Union[int, float, np.ndarray],
                                             location: int) -> Tuple[float, float]:
        """Returns the probability of starting with `n_cars` cars at `location` and ending the day with `n_cars_next`
        and the expected reward for this transition."""

        n_cars, n_cars_next = int(n_cars), int(n_cars_next)
        expected_rental_requests = self.car_rental_env.expected_rental_requests[location]
        expected_rental_returns = self.car_rental_env.expected_rental_returns[location]

        transition_prob = 0
        expected_reward = 0
        c = n_cars_next - n_cars
        for n_requests in range(n_cars):
            arrival_prob = stats.poisson.pmf(n_requests + c, expected_rental_returns)
            requests_prob = stats.poisson.pmf(n_requests, expected_rental_requests)
            transition_prob += arrival_prob * requests_prob
            expected_reward += requests_prob * self.car_rental_env.rental_credit * n_requests

        arrival_prob = stats.poisson.pmf(n_cars + c, expected_rental_returns)
        # Probability of more than `n_cars` requests
        requests_prob = 1 - stats.poisson.cdf(n_cars - 1, expected_rental_requests)

        transition_prob += arrival_prob * requests_prob
        expected_reward += requests_prob * self.car_rental_env.rental_credit * n_cars

        return transition_prob, expected_reward

    def _initialize_matrices(self) -> None:
        self._transition_matrix_without_action = np.zeros((self.n_states, self.n_states))
        self._reward_matrix_without_action = np.zeros((self.n_states, self.n_states))

        # Auxiliary matrices. They contain the probability of transitioning from `i` cars to `j` cars and the expected
        # reward for this transition for each location.
        transition_probs1, expected_rewards1 = np.fromfunction(
            np.vectorize(partial(self._get_probability_and_expected_reward, location=0)),
            (self.car_rental_env.max_n_cars + 1, self.car_rental_env.max_n_cars + 1),
        )
        transition_probs2, expected_rewards2 = np.fromfunction(
            np.vectorize(partial(self._get_probability_and_expected_reward, location=1)),
            (self.car_rental_env.max_n_cars + 1, self.car_rental_env.max_n_cars + 1),
        )
        # For each state, compute the transition probabilities and expected rewards for all possible states.
        for state in range(self.n_states):
            cars_first_location, cars_second_location = self.car_rental_env.int2obs(state)
            for new_state in range(self.n_states):
                new_cars_first_location, new_cars_second_location = self.car_rental_env.int2obs(new_state)

                probability_loc1 = transition_probs1[cars_first_location, new_cars_first_location]
                expected_reward_loc1 = expected_rewards1[cars_first_location, new_cars_first_location]
                probability_loc2 = transition_probs2[cars_second_location, new_cars_second_location]
                expected_reward_loc2 = expected_rewards2[cars_second_location, new_cars_second_location]

                # Since both transitions are independent, the probability of transitioning from `state` to `new_state`
                # is the product of the probabilities of transitioning from `cars_first_location` to
                # `new_cars_first_location` and from `cars_second_location` to `new_cars_second_location`.
                self._transition_matrix_without_action[state, new_state] = probability_loc1 * probability_loc2

                # The expected reward for transitioning from `state` to `new_state` is the sum of the expected rewards
                # of each location.
                self._reward_matrix_without_action[state, new_state] = expected_reward_loc1 + expected_reward_loc2

    def _get_new_state(self, state: int, action: int) -> int:
        """Returns the state that results from taking `action` in `state` without considering the rental requests and
        returns."""
        car_tuple = self.car_rental_env.int2obs(state)
        new_car_tuple = self.car_rental_env.move_cars(list(car_tuple), action)
        return self.car_rental_env.obs2int(new_car_tuple)

    def get_transition_probabilities(self, state: int, action: int) -> np.ndarray:
        """Returns the transition probabilities for all states given a state and action."""
        new_state = self._get_new_state(state, action)
        return self._transition_matrix_without_action[new_state, :]

    def get_immediate_reward(self, state: int, action: int) -> np.ndarray:
        """Returns the immediate reward for all states given a state and action."""
        new_state = self._get_new_state(state, action)
        n_moves = abs(action - self.car_rental_env.max_moves)
        return self._reward_matrix_without_action[new_state, :] - n_moves * self.car_rental_env.move_cost

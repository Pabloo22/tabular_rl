from typing import Tuple, Union

import functools
import numpy as np
import scipy.stats as stats

from tabular_rl.core import MarkovDecisionProcess
from tabular_rl.envs import CarRentalEnv


class CarRentalMDP(MarkovDecisionProcess):
    """The specific MDP for the Jack's Rental problem.

    This class inherits from `MarkovDecisionProcess` and implements the methods `get_transition_probabilities` and
    `get_expected_rewards`. This allows us to save memory by not storing the transition and reward matrices. Instead, we
    compute transition and reward matrices that do not take into account the action. This can be done because the
    transition and reward matrices are the same for many actions and states. For example, if we start with 5 cars at
    both locations and move 2 cars from the first location to the second location, the transition matrix
    for this action is the same as if we started with 3 cars at first location and 7 cars at the second location, and
    we do not move any cars. This is because at the start of the day, the number of cars at each location is the
    same in both cases.
    In the case of the expected reward, it would be the same; but, in the first case, we would need
    to add the cost of moving 2 cars. Thus, is enough to compute only the transition and reward matrices for the
    states without taking into account the action and then add the cost of moving cars when computing the expected
    reward.

    Args:
        car_rental_env: The environment for the Jack's Rental problem.
    """

    def __init__(self, car_rental_env: CarRentalEnv):

        super().__init__(n_states=car_rental_env.n_states,
                         n_actions=car_rental_env.n_actions,
                         discount=car_rental_env.discount,
                         env=car_rental_env)

        self.env = car_rental_env
        self._transition_matrix_without_action = None
        self._reward_matrix_without_action = None

        self._initialize_matrices()

    def _get_probability_and_expected_reward(self,
                                             n_cars: Union[int, float, np.ndarray],
                                             n_cars_next: Union[int, float, np.ndarray],
                                             location: int) -> Tuple[float, float]:
        """Returns the probability of starting with `n_cars` cars at `location` and ending the day with `n_cars_next`
        and the expected reward for this transition."""

        # We need to convert to int `n_cars` and `n_cars_next`to avoid when problems when using `np.vectorize` and
        # `np.fromfunction`.
        n_cars = int(n_cars)
        n_cars_next = int(n_cars_next)

        expected_rental_requests = self.env.expected_rental_requests[location]
        expected_rental_returns = self.env.expected_rental_returns[location]

        transition_prob = 0
        expected_reward = 0
        diff = n_cars_next - n_cars
        # To compute the probability of transitioning from `n_cars` to `n_cars_next`, we need add the probabilities of
        # all the possibilities that can lead to this transition. For example, if `n_cars` is 3 and `n_cars_next` is 5,
        # we need to add the probabilities of having: 0 requests and 2 returns; 1 request and 3 returns; 2 requests
        # and 4 returns; and 3 or more requests and 5 returns.

        # In the case that `n_cars_next` is the maximum number of cars, the probability of arriving
        # `n_requests + diff` cars is the probability of arriving at least `n_requests + diff` cars. If
        # more cars arrive, they are not considered.

        # To compute the immediate reward, we need to add the reward for all the possibilities that can lead to this
        # transition and do a weighted average by the probability of each possibility.
        for n_requests in range(n_cars):
            requests_prob = stats.poisson.pmf(n_requests, expected_rental_requests)
            if n_cars_next < self.env.max_cars:
                arrival_prob = stats.poisson.pmf(n_requests + diff, expected_rental_returns)
            else:
                current_cars = n_cars - n_requests
                arrival_prob = 1 - stats.poisson.cdf(self.env.max_cars - current_cars - 1, expected_rental_returns)

            transition_prob += arrival_prob * requests_prob
            expected_reward += requests_prob * arrival_prob * self.env.rental_credit * n_requests

        # After the for loop ends we need to take into account the possibility of having `n_cars` or more requests
        if n_cars_next < self.env.max_cars:
            arrival_prob = stats.poisson.pmf(n_cars_next, expected_rental_returns)
        else:
            arrival_prob = 1 - stats.poisson.cdf(self.env.max_cars - 1, expected_rental_returns)

        # Probability of `n_cars` or more requests
        requests_prob = 1 - stats.poisson.cdf(n_cars - 1, expected_rental_requests)

        transition_prob += arrival_prob * requests_prob
        expected_reward += requests_prob * arrival_prob * self.env.rental_credit * n_cars

        # Since all probabilities of each possibility must add up to 1, we need to divide each probability by the
        # sum of all probabilities:
        expected_reward = expected_reward / transition_prob if transition_prob > 0 else 0
        return transition_prob, expected_reward

    def _initialize_matrices(self) -> None:
        self._transition_matrix_without_action = np.zeros((self.n_states, self.n_states))
        self._reward_matrix_without_action = np.zeros((self.n_states, self.n_states))

        # Auxiliary matrices. They contain the probability of transitioning from `i` cars to `j` cars and the expected
        # reward for this transition for each location.
        transition_probs1, expected_rewards1 = np.fromfunction(
            np.vectorize(functools.partial(self._get_probability_and_expected_reward, location=0)),
            (self.env.max_cars + 1, self.env.max_cars + 1),
        )
        transition_probs2, expected_rewards2 = np.fromfunction(
            np.vectorize(functools.partial(self._get_probability_and_expected_reward, location=1)),
            (self.env.max_cars + 1, self.env.max_cars + 1),
        )
        for state in range(self.n_states):
            cars_first_location, cars_second_location = self.env.int2obs(state)
            for new_state in range(self.n_states):
                new_cars_first_location, new_cars_second_location = self.env.int2obs(new_state)

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
        car_tuple = self.env.int2obs(state)
        new_car_tuple = self.env.move_cars(list(car_tuple), action)
        return self.env.obs2int(new_car_tuple)

    def get_transition_probabilities(self, state: int, action: int) -> np.ndarray:
        """Returns the transition probabilities for all states given a state and action."""
        new_state = self._get_new_state(state, action)  # Move cars
        return self._transition_matrix_without_action[new_state, :]  # p(s' | s, a = no cars moved)

    def get_immediate_reward(self, state: int, action: int) -> np.ndarray:
        """Returns the immediate reward for all states given a state and action."""
        new_state = self._get_new_state(state, action)  # Move cars
        n_moves = abs(action - self.env.max_moves)
        return self._reward_matrix_without_action[new_state, :] - n_moves * self.env.move_cost

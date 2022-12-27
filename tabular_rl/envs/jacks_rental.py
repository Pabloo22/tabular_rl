import numpy as np
import scipy.stats as stats
from typing import Tuple, List, Sequence

from tabular_rl.base import TabEnv


class JacksRental(TabEnv):
    """Environment for the Jack's Car Rental problem. (Example 4.2 in Sutton and Barto
     Reinforcement Learning: An Introduction)

    Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at
    each location to rent cars. If Jack has a car available, he rents it out and is
    credited $10 by the national company. If he is out of cars at that location,
    then the business is lost. Cars become available for renting the day after they
    are returned. To help ensure that cars are available where they are needed,
    Jack can move them between the two locations overnight, at a cost of $2 per
    car moved. We assume that the number of cars requested and returned at
    each location are Poisson random variables, meaning that the probability that
    the number is n is λ^n/n! e^(-λ). Cars are rented from one location and, where λ is the expected number.
    Suppose λ is 3 and 4 for rental requests at the first and second locations and 3 and 2 for
    returns. To simplify the problem slightly, we assume that there can be no
    more than 20 cars at each location (any additional cars are returned to the
    nationwide company, and thus disappear from the problem) and a maximum
    of five cars can be moved from one location to the other in one night. We take
    the discount rate to be γ = 0.9 and formulate this as a continuing finite MDP,
    where the time steps are days, the state is the number of cars at each location
    at the end of the day, and the actions are the net numbers of cars moved
    between the two locations overnight.

    Arguments:
        max_n_cars: Maximum number of cars at each location.
        max_n_moved_cars: Maximum number of cars that can be moved between the two locations.
        expected_rental_requests: Expected number of rental requests at the first and second locations.
        expected_rental_returns: Expected number of cars returned to the first and second locations.
        rental_credit: Reward for each rented car.
        cost_of_moving: Cost for moving a car.
        initial_state: Initial number of cars at each location.
        max_episode_length: Maximum number of steps in an episode. If it is `None`, the episode length is unlimited.
        discount: Discount factor.
        seed: Seed for the random number generator.
    """

    def __init__(self,
                 max_n_cars: int = 20,
                 max_n_moved_cars: int = 5,
                 expected_rental_requests: Sequence = (3, 4),
                 expected_rental_returns: Sequence = (3, 2),
                 rental_credit: float = 10,
                 cost_of_moving: float = 2,
                 initial_state: Tuple[int, int] = (10, 10),
                 max_episode_length: int = None,
                 discount: float = 0.9,
                 seed: int = None):

        self.max_n_cars = max_n_cars
        self.max_n_moved_cars = max_n_moved_cars
        self.expected_rental_requests = np.array(expected_rental_requests)
        self.expected_rental_returns = np.array(expected_rental_returns)
        self.rental_credit = rental_credit
        self.cost_of_moving = cost_of_moving
        self.initial_state: Tuple[int, int] = initial_state
        self.discount = discount
        self.max_episode_length = max_episode_length if max_episode_length is not None else float("inf")
        self.n_steps = 0
        self.seed = seed

        self.cars: List[int, int] = list(initial_state)

        n_states = (max_n_cars + 1) ** 2
        n_actions = 2 * max_n_moved_cars + 1
        super().__init__(n_states, n_actions)

        np.random.seed(seed)

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Performs an action in the environment.

        Arguments:
            action: Action to perform:
                0: Move `max_n_moved_cars` cars from the first location to the second location.
                1: Move `max_n_moved_cars` - 1 cars from the first location to the second location.
                ...
                max_n_moved_cars - 1: Move 1 car from the first location to the second location.
                max_n_moved_cars: Do not move any cars.
                max_n_moved_cars + 1: Move 1 car from the second location to the first location.
                ...
                2 * max_n_moved_cars: Move `max_n_moved_cars` from the second location to the first location.
        """
        reward = 0

        # Move cars
        n_moved_cars = action - self.max_n_moved_cars
        self.cars[0] += n_moved_cars
        self.cars[1] -= n_moved_cars

        # Cost for moving cars
        reward -= self.cost_of_moving * abs(n_moved_cars)

        # Rent cars
        requests = np.random.poisson(self.expected_rental_requests)
        for i, request in enumerate(requests):
            cars_rented = min(self.cars[0], request)
            self.cars[i] -= cars_rented
            reward += self.rental_credit * cars_rented

        # Return cars
        returns = np.random.poisson(self.expected_rental_returns)
        for i, return_ in enumerate(returns):
            self.cars[i] = min(self.cars[i] + return_, self.max_n_cars)

        # Check if episode is done
        self.n_steps += 1
        done = self.n_steps == self.max_episode_length

        return tuple(self.cars), reward, done, None

    def obs2int(self, observation: Tuple[int, int]) -> int:
        """Converts a state to an integer."""
        return observation[0] * (self.max_n_cars + 1) + observation[1]

    def int2obs(self, state: int) -> Tuple[int, int]:
        """Converts an integer to a state."""
        return state // (self.max_n_cars + 1), state % (self.max_n_cars + 1)

    def reset(self) -> Tuple[int, int]:
        """Resets the environment."""
        self.cars = list(self.initial_state)
        self.n_steps = 0
        return self.initial_state

    def render(self):
        """Renders the environment."""
        cars_in_first_location, cars_in_second_location = self.cars
        print(f"Number of cars in first location: {cars_in_first_location}")
        print(f"Number of cars in second location: {cars_in_second_location}")
        print("-" * 50)

        if self.n_steps == self.max_episode_length:
            print("Episode done")

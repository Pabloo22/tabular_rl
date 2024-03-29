from typing import Tuple, List, Sequence, Union, Optional

import numpy as np
import matplotlib.pyplot as plt

from tabular_rl.core import TabEnv


class CarRentalEnv(TabEnv):
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

    The parameters written above are the default parameters. You can change them by passing them as arguments
    to the constructor.

    Args:
        max_cars: Maximum number of cars at each location.
        max_moves: Maximum number of cars that can be moved between the two locations.
        expected_rental_requests: Expected number of rental requests at the first and second locations.
        expected_rental_returns: Expected number of cars returned to the first and second locations.
        rental_credit: Reward for each rented car.
        move_cost: Cost for moving a car.
        initial_state: Initial number of cars at each location. If None, the initial state is (max_n_cars // 2,
            max_n_cars // 2).
        max_episode_length: Maximum number of steps in an episode. If it is `None`, the episode length is unlimited.
            We recommend to set it to a finite value to be able to use the `play` and `evaluate_agent` methods.
        discount: Discount factor.
        seed: Seed for the random number generator.
    """

    def __init__(self,
                 max_cars: int = 20,
                 max_moves: int = 5,
                 expected_rental_requests: Sequence = (3, 4),
                 expected_rental_returns: Sequence = (3, 2),
                 rental_credit: float = 10,
                 move_cost: float = 2,
                 initial_state: Optional[Tuple[int, int]] = None,
                 max_episode_length: Optional[int] = 100,
                 discount: float = 0.9,
                 seed: int = None):

        self.max_cars = max_cars
        self.max_moves = max_moves
        self.expected_rental_requests = np.array(expected_rental_requests)
        self.expected_rental_returns = np.array(expected_rental_returns)
        self.rental_credit = rental_credit
        self.move_cost = move_cost
        self.initial_state: Tuple[int, int] = initial_state if initial_state is not None else (
            max_cars // 2, max_cars // 2)
        self.max_episode_length = max_episode_length if max_episode_length is not None else float("inf")
        self.n_steps = 0
        self.seed = seed

        self.cars = list(self.initial_state)

        n_states = (max_cars + 1) ** 2
        n_actions = 2 * max_moves + 1
        super().__init__(n_states, n_actions, discount)

        if seed is not None:
            np.random.seed(seed)

    def move_cars(self, cars: Union[List[int], Tuple[int, int]], action: int) -> Tuple[int, int]:
        """Returns a tuple of the number of cars in each location after moving cars."""
        n_moves_first2second = self.max_moves - action

        cars_first_location, cars_second_location = cars
        # Check if there are enough cars in the first or second location and that there is enough space in the other
        # location
        move_cars_first2second = n_moves_first2second < 0
        if move_cars_first2second:
            n_moves_first2second = -min(cars_second_location, -n_moves_first2second)
            n_moves_first2second = -min(self.max_cars - cars_first_location, -n_moves_first2second)
        else:
            n_moves_first2second = min(cars_first_location, n_moves_first2second)
            n_moves_first2second = min(self.max_cars - cars_second_location, n_moves_first2second)

        cars_first_location -= n_moves_first2second
        cars_second_location += n_moves_first2second

        return cars_first_location, cars_second_location

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, None]:
        """Performs an action in the environment.

        Args:
            action: Action to perform:
                0: Move `max_moves` cars from the first location to the second location.
                1: Move `max_moves` - 1 cars from the first location to the second location.
                ...
                max_moves - 1: Move 1 car from the first location to the second location.
                max_moves: Do not move any cars.
                max_moves + 1: Move 1 car from the second location to the first location.
                ...
                2 * max_moves: Move `max_moves` from the second location to the first location.
        """
        reward = 0

        self.cars = list(self.move_cars(self.cars, action))

        # Cost for moving cars
        n_moves = abs(action - self.max_moves)
        reward -= self.move_cost * n_moves

        # Rent cars
        requests = np.random.poisson(self.expected_rental_requests)
        for i, request in enumerate(requests):
            cars_rented = min(self.cars[i], request)
            self.cars[i] -= cars_rented
            reward += self.rental_credit * cars_rented

        # Return cars
        returns = np.random.poisson(self.expected_rental_returns)
        for i, return_ in enumerate(returns):
            self.cars[i] = min(self.cars[i] + return_, self.max_cars)

        self.n_steps += 1
        done = self.n_steps >= self.max_episode_length

        return tuple(self.cars), reward, done, None

    def obs2int(self, observation: Tuple[int, int]) -> int:
        """Converts an observation to an integer.

        Args:
            observation: The observation to convert.

        Returns:
            The integer representation of the state.
        """
        return int(np.ravel_multi_index(observation, (self.max_cars + 1, self.max_cars + 1)))

    def int2obs(self, state: int) -> Tuple[int, int]:
        """Converts an integer to an observation.

        Args:
            state: The integer to convert.

        Returns:
            The observation represented by the integer.
        """
        return tuple(np.unravel_index(state, (self.max_cars + 1, self.max_cars + 1)))

    def reset(self) -> Tuple[int, int]:
        """Resets the environment."""
        self.cars = list(self.initial_state)
        self.n_steps = 0
        return self.initial_state

    def render(self):
        """Prints the number of cars in each location and whether the episode is done."""
        cars_in_first_location, cars_in_second_location = self.cars
        print(f"Number of cars in first location: {cars_in_first_location}")
        print(f"Number of cars in second location: {cars_in_second_location}")
        print("-" * 50)

        if self.n_steps == self.max_episode_length:
            print("Episode done")

    def transform_array(self, array: np.ndarray, transform_actions: bool = True) -> np.ndarray:
        """Transforms a policy or a state_value_array that maps states (integers) to values or actions to an array that
        that maps observations (tuples) to the same values or actions.

        The actual action is modified so that it represents the number of cars moved from the first location to the
        second location.

        Args:
            array: The policy or the state-value array to transform.
            transform_actions: If True, the actions are transformed to represent the number of cars moved from the first
                location to the second location. If False, the actions are represented as the index of the action in the
                action space or the value of the state if `array` is a state-value array.

        Returns:
            The transformed policy or state-value array.
        """

        new_policy = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=int)
        for state, action in enumerate(array):
            n_cars_first_loc, n_cars_second_loc = self.int2obs(state)
            value = self.max_moves - action if transform_actions else action
            new_policy[self.max_cars - n_cars_first_loc, n_cars_second_loc] = value

        return new_policy

    def visualize_array(self, array: np.ndarray, title: str = None, transform_actions: bool = True):
        """Visualizes the policy or the state-value array.

        Args:
            array: The policy or the state-value array to visualize.
            title: Title of the plot.
            transform_actions: If True, the actions are transformed to represent the number of cars moved from the first
                location to the second location. If False, the actions are represented as the index of the action in the
                action space or the value of the state if `array` is a state-value array.
        """
        fig, ax = plt.subplots()

        array = self.transform_array(array, transform_actions=transform_actions) if array.ndim == 1 else array

        im = ax.imshow(array, cmap="viridis")
        x_ticks = np.arange(self.max_cars + 1)
        y_ticks = list(range(self.max_cars + 1))
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks[::-1])
        ax.set_xlabel("Number of cars in second location")
        ax.set_ylabel("Number of cars in first location")

        cbar = ax.figure.colorbar(im, ax=ax)
        if transform_actions:
            cbar.ax.set_ylabel("Number of cars moved from first to second location", rotation=-90, va="bottom")

        if title is not None:
            ax.set_title(title)

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ax.text(j, i, array[i, j], ha="center", va="center", color="w")

        fig.tight_layout()

        return fig, ax


if __name__ == "__main__":
    env = CarRentalEnv(max_episode_length=10)

    print("Actions:")
    print(f"0: Move {env.max_moves} cars from the first location to the second location.")
    print(f"1: Move {env.max_moves - 1} cars from the first location to the second location.")
    print("...")
    print(f"{env.max_moves - 1}: Move 1 car from the first location to the second location.")
    print(f"{env.max_moves}: Do not move any cars.")
    print(f"{env.max_moves + 1}: Move 1 car from the second location to the first location.")
    print("...")
    print(f"{2 * env.max_moves}: Move {env.max_moves} from the second location to the first location.")

    def select_action(observation):
        return int(input("Select action: "))

    print("Total reward:", env.play(select_action, verbose=True))

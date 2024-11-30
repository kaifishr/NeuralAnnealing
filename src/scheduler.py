"""Script with temperature schedulers."""

import math


class Scheduler:
    """Abstract scheduler class."""

    def __init__(
        self,
    ) -> None:
        """Initializes abstract scheduler class."""

    def __call__(self, temp: float, iteration: int) -> None:
        """Schedules temperature decrease."""
        raise NotImplementedError


class PowerScheduler(Scheduler):
    """Decays the temperature by gamma every iteration."""

    def __init__(self, gamma: float) -> None:
        """Initializes scheduler for temperature."""
        super().__init__()
        self.gamma = gamma

    def __call__(self, temp: float, iteration: int) -> None:
        """Schedules temperature decrease."""
        return temp * self.gamma**iteration


class ExponentialScheduler(Scheduler):
    """Decays temperature exponentially."""

    def __init__(self, gamma: float, temp_initial: float, temp_final: float) -> None:
        """Initializes scheduler for temperature."""
        super().__init__()
        self.gamma = gamma
        self.temp_initial = temp_initial
        self.temp_final = temp_final
        self.total_iterations = None

        self._print_info()

    def _print_info(self) -> None:
        """Computes and prints required iterations to reach final temperature."""
        self.total_iterations = int(
            math.ceil(
                (1.0 / self.gamma) * math.log(self.temp_initial / self.temp_final)
            )
            + 1
        )
        print(f"Required iterations: {self.total_iterations}")

    def __call__(self, temp: float, iteration: int) -> None:
        """Schedules temperature decrease."""
        temp = self.temp_initial * math.exp(-self.gamma * iteration)
        return temp


class LinearScheduler(Scheduler):
    """Linear temperature decays."""

    def __init__(self, temp_initial: float, iterations_max: int) -> None:
        """Initializes scheduler for temperature."""
        super().__init__()
        self.temp_initial = temp_initial
        self.iterations_max = iterations_max

    def __call__(self, temp: float, iteration: int) -> None:
        """Schedules temperature decrease."""
        return self.temp_initial - iteration * (self.temp_initial / self.iterations_max)


class LinearSchedulerv2(Scheduler):
    """Linear temperature decays."""

    def __init__(self, gamma: float, min_temp: float = 0.0) -> None:
        """Initializes scheduler for temperature."""
        super().__init__()
        self.gamma = gamma
        self.min_temp = min_temp

    def __call__(self, temp: float, iteration: int) -> None:
        """Schedules temperature decrease."""
        new_temp = temp - self.gamma
        return new_temp if new_temp > self.min_temp else self.min_temp


class CosineAnnealingScheduler(Scheduler):
    """Temperature decays.

    See also https://arxiv.org/pdf/1608.03983.pdf
    for cosine annealing schedule for gradient descent.

    """

    def __init__(
        self, temp_min: float, temp_max: float, iter_per_cycle: int, gamma: float = None
    ) -> None:
        """Initializes scheduler for temperature."""
        super().__init__()
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.iter_per_cycle = iter_per_cycle

        self.gamma = gamma

    def __call__(self, temp: float, iteration: int) -> None:
        """Schedules temperature decrease."""
        x = math.pi * (iteration / self.iter_per_cycle) % math.pi
        temp = self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (
            1.0 + math.cos(x)
        )
        if self.gamma:
            temp *= self.gamma**iteration
        return temp

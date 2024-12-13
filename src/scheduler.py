import math


class Scheduler:

    def __init__(
        self,
        temp_start: float,
        temp_final: float = None,
        gamma: float = None,
    ) -> None:
        self.temp_start = temp_start
        self.temp_final = temp_final
        self.gamma = gamma

    def __call__(self, temp: float, iteration: int) -> float:
        raise NotImplementedError


class GeometricScheduler(Scheduler):

    def __init__(self, temp_start: float, temp_final: float, gamma: float) -> None:
        super().__init__(temp_start=temp_start, temp_final=temp_final, gamma=gamma)

    def __call__(self, temp: float, iteration: int) -> float:
        temp = self.gamma * temp
        if self.temp_final:
            temp = max(self.temp_final, temp)
        return temp


class ExponentialScheduler(Scheduler):

    def __init__(self, gamma: float, temp_start: float, temp_final: float) -> None:
        super().__init__(temp_start=temp_start, temp_final=temp_final, gamma=gamma)

    def __call__(self, temp: float, iteration: int) -> float:
        return self.temp_start * math.exp(-self.gamma * iteration)


class CosineAnnealingScheduler(Scheduler):
    """
    See also https://arxiv.org/pdf/1608.03983.pdf
    for cosine annealing schedule for gradient descent.
    """

    def __init__(
        self,
        temp_start: float,
        temp_max: float,
        iter_per_cycle: int,
        gamma: float = None,
    ) -> None:
        super().__init__(temp_start=temp_start, gamma=gamma)
        self.temp_max = temp_max
        self.iter_per_cycle = iter_per_cycle

    def __call__(self, temp: float, iteration: int) -> float:
        x = math.pi * (iteration / self.iter_per_cycle) % math.pi
        temp = self.temp_start + 0.5 * (self.temp_max - self.temp_start) * (
            1.0 + math.cos(x)
        )
        if self.gamma:
            temp *= self.gamma**iteration
        return temp

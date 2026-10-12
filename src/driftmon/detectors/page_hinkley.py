"""Page-Hinkley change detector for the per-window error-rate stream."""


class PageHinkley:
    """Detects an increase in the mean of a stream: `delta` is the tolerated
    drift per step, `lambda_` the alarm threshold on the cumulative deviation."""

    def __init__(self, delta: float, lambda_: float) -> None:
        if delta < 0 or lambda_ <= 0:
            raise ValueError("delta must be >= 0 and lambda_ must be > 0")
        self.delta = delta
        self.lambda_ = lambda_
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._cumulative = 0.0
        self._minimum = 0.0

    @property
    def score(self) -> float:
        return self._cumulative - self._minimum

    def update(self, value: float) -> bool:
        self._count += 1
        self._mean += (value - self._mean) / self._count
        self._cumulative += value - self._mean - self.delta
        self._minimum = min(self._minimum, self._cumulative)
        return self.score > self.lambda_

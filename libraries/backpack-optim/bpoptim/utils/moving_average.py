class MovingAverage:
    """
    A moving average scheme for a list of tensors.
    """

    def __init__(self, start):
        self.estimate = list(start)

    def __validate_parameters(self):
        pass

    def step(self, news):
        raise NotImplementedError

    def get(self):
        return self.estimate


class NoMovingAverage(MovingAverage):
    """
    No moving average - just return the last estimate seen
    """

    def __init__(self, start):
        super().__init__(start)

    def step(self, news):
        for estimate, new, in zip(self.estimate, news):
            estimate.data = new.data
        return self.get()


class ParametrizedMovingAverageBaseClass(MovingAverage):
    """
    Base class for moving averages parametrized by a scalar and an iteration counter
    """

    def __init__(self, start, alpha=0.9):
        self.alpha = alpha
        self.step_counter = 0
        super().__init__(start)
        self.__validate_parameters()

    def __validate_parameters(self):
        moving_average_is_ratio = (1. > self.alpha > 0)

        if not moving_average_is_ratio:
            raise ValueError(
                "Moving average parameter is invalid. " +
                "Got alpha={}, but should be 0 < alpha < 1.".format(self.alpha)
            )

    def step(self, news):
        self.step_counter += 1
        return self.get()


class BiasedMovingAverage(ParametrizedMovingAverageBaseClass):
    """
    Biased Moving average

    Implements the update `m_{t} = (1-𝛼) m_{t-1} + 𝛼 x_t` where
    - `m_t` is the estimate at step `t`
    - `x_t` is the observation at step `t`
    - `t` starts at `t=0`.
    (is biased towards the initial value `m_0`)
    """

    def step(self, news):
        for estimate, new, in zip(self.estimate, news):
            estimate.mul_(1 - self.alpha).add_(self.alpha, new)

        return super().step(news)

    def get(self):
        return self.estimate


class BiasCorrectedMovingAverage(ParametrizedMovingAverageBaseClass):
    """
    Bias corrected moving average (Adam-style)

    Implements the update `m_{t} = (1-𝛼) m_{t-1} + 𝛼 x_t` where
    - `m_t` is the estimate at step `t`
    - `x_t` is the observation at step `t`
    - `t` starts at `t=0`.
    and returns the estimate `m_t / (1-𝛼^t)`
    """

    def step(self, news):
        for estimate, new, in zip(self.estimate, news):
            estimate.mul_(1 - self.alpha).add_(self.alpha, new)

        return self.get()

    def get(self):
        return list([
            est / (1 - self.alpha ** self.step_counter) for est in self.estimate
        ])


class CappedMovingAverage(MovingAverage):
    def __init__(self, start, alpha=0.9):
        """
        Bias-corrected moving average (KFAC-style)

        Implements the update `m_{t+1} = (1-𝛼_t) m_t + 𝛼_t x_t` where
        - `𝛼_t = min(1/(t+1), 𝛼)`,
        - `m_t` is the estimate at step `t`
        - `x_t` is the observation at step `t`
        `t` starts at `t=0`.

        """
        super().__init__(start)
        self.alpha = alpha
        self.step_counter = 0

    def step(self, news):
        alpha_t = min([1. / (self.step_counter + 1), self.alpha])

        for estimate, new, in zip(self.estimate, news):
            estimate.mul_(1 - alpha_t).add_(alpha_t, new)

        return super().step(news)

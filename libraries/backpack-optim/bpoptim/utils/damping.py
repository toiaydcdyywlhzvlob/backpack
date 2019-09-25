from typing import Callable


class DampingScheme:
    """
    A base class for damping schemes
    """

    def update(self, loss_before, loss_after_func: Callable, predicted_improvement):
        """
        The update might not be needed. `loss_after_func` should be a function
        that computes the loss after the step only if needed.
        """
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class ConstantDamping(DampingScheme):
    """
    """

    def __init__(self, damping=1.):
        self.damping = damping
        self.__validate_parameters()

    def __validate_parameters(self):
        damping_is_positive = self.damping > 0
        assert damping_is_positive

    def update(self,
               loss_before=None,
               loss_after_func: Callable = None,
               predicted_improvement=None
               ):
        """
        Constant damping does not update the damping parameter.
        Parameters passed to this function are not used.
        """
        return

    def get(self):
        return self.damping


class LevenbergMarquartDamping(DampingScheme):
    """
    Levenberg-Marquardt heuristic update for the damping parameter.

    Given the predicted improvement `pred_imp` for the step 𝛿 and the true
    change in loss function, `true_imp = loss(x + 𝛿) - loss(x)`,
    compute the `improvement_ratio = true_imp / pred_imp`.

    If the ratio is large (>.75), reduce the damping parameter.
    If the ratio is small (<.25), increase the damping parameter.

    The update might not be done at each step but every `T` steps.
    Increase and decrease are multiplicative, by a factor `𝜔 ** T`, 0 < 𝜔 < 1.
    """

    def __init__(self, damping=100., update_factor=.95, update_interval=1):
        self.damping = damping
        self.update_factor = update_factor
        self.update_interval = update_interval
        self.step_counter = 0
        self.__validate_parameters()

    def __validate_parameters(self):
        damping_is_positive = self.damping > 0
        update_factor_is_ratio = (0 < self.update_factor < 1.)
        update_interval_is_int = isinstance(self.update_interval, int)
        update_interval_is_positive = self.update_interval > 0

        assert damping_is_positive
        assert update_interval_is_positive
        assert update_interval_is_int
        assert update_factor_is_ratio

    def get(self):
        return self.damping

    def update(self, loss_before, loss_after_func: Callable, predicted_improvement):
        should_update_this_iter = self.step_counter % self.update_interval == 0

        if should_update_this_iter:
            actual_improvement = loss_after_func() - loss_before
            improvement_ratio = actual_improvement / predicted_improvement
            if improvement_ratio > .75:
                self.damping *= self.update_factor ** self.update_interval
            if improvement_ratio < .25:
                self.damping /= self.update_factor ** self.update_interval

        self.step_counter += 1

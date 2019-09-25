"""
To better understand how to structure the thingy,
here are 3 algorithms
"""
from backpack import backpack
from backpack.extensions import DiagGGN, KFAC
from bpoptim.utils.moving_average import MovingAverage
from math import sqrt


class Curvature:
    def compute(self, closure):
        raise NotImplementedError


def diag_ggn_inverse_update(parameters, closure, damping):
    with backpack(DiagGGN()):
        loss = closure()
        loss.backward()

        step = []
        for p in parameters:
            step.append(p.grad / (p.diag_ggn + damping))

    return step


def kfac_simple_update(parameters, closure, damping):
    with backpack(KFAC()):
        loss = closure()
        loss.backward()

        step = []
        for p in parameters:
            ...

    return step


def PCH_CG_update(parameters, closure, damping):
    with backpack(...):
        loss = closure()
        loss.backward()

        step = []
        for p in parameters:
            ...

    return step


class KFAC_proposal():
    def __init__(
            self,
            moving_avg: MovingAverage,
            curvature: Curvature,
            update_inverse_interval=5
    ):
        self.mvg_avg = moving_avg
        self.curv = curvature
        self.inverse = None
        self.update_inv_interval = update_inverse_interval
        self.step_counter = 0

    def update(self, closure):
        self.mvg_avg.step(self.curv.compute(closure))
        pass

    def compute_step(self):
        should_update_inv = self.update_inv_interval % self.step_counter == 0

        self.step_counter += 1
        pass


class FancyDampingWrapper():

    ############################################################################
    # Init and Validation
    ############################################################################

    def __init__(
            self,
            curvature: Curvature,
            moving_average: MovingAverage,
            l2_reg=0.,  # eta
            inv_damping=None,  # gamma
            inv_damping_factor=None,  # ω2
            update_interval_inv=20,  # T3
    ):
        """
        """

        self.step_counter = 0

        self.mvg_avg = moving_average
        self.curvature_wrapper = curvature
        self.l2_reg = l2_reg
        if inv_damping is None:
            self.inv_damping = sqrt(150. + l2_reg)
        else:
            self.inv_damping = inv_damping

        if inv_damping_factor is None:
            self.inv_damping_factor = sqrt(.95) ** update_interval_inv
        else:
            self.inv_damping_factor = inv_damping_factor

        self.update_interval_inv = update_interval_inv
        self.__validate_parameters()

    def __validate_parameters(self):
        update_intervals_are_positive_ints = (
                isinstance(self.update_interval_inv, int) and
                self.update_interval_inv > 0
        )

        damping_between_0_and_1 = (0. < self.inv_damping_factor <= 1.)

        if not update_intervals_are_positive_ints:
            raise ValueError(
                "Update intervals need to be positive integers." +
                "Got {}".format(self.update_interval_inv)
            )

        if not damping_between_0_and_1:
            raise ValueError(
                "Damping factors need to be 0 < x <= 1. " +
                "Got {}".format(self.inv_damping_factor)
            )

    ############################################################################
    # Main update
    ############################################################################

    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
            and returns the loss.
        """

        loss = self.curvature_wrapper.compute_derivatives_and_stuff(closure)

        if self.__should_update_inverse() or self.__should_update_inv_damping(
        ):
            step = self.__update_inverse_and_inv_damping_and_compute_step()
        else:
            step = self.curvature_wrapper.compute_step(
                self.inv_damping, self.trust_damping, self.l2_reg)

        self.__apply_step(step)

        self.__update_trust_damping_if_needed()

        self.curvature_wrapper.end_of_step()
        self.step_counter += 1

        debug("inv_damping", self.inv_damping)
        debug("trust_damping", self.trust_damping)

        return loss

    def __update_inverse_and_inv_damping_and_compute_step(self):

        best_candidate_score = INFINITY
        best_step = None
        best_inv_damping = None
        inv_damping_candidates = self.__inv_damping_candidates()

        for inv_damping_candidate in inv_damping_candidates:

            if self.__should_update_inverse():
                self.curvature_wrapper.inverse_candidate(inv_damping_candidate)

            step = self.curvature_wrapper.compute_step(
                inv_damping_candidate, self.trust_damping, self.l2_reg)

            if len(inv_damping_candidates) == 1:
                self.curvature_wrapper.accept_inverse_candidate()
                return step
            else:
                candidate_score = self.curvature_wrapper.evaluate_step(
                    step, self.trust_damping, self.l2_reg)

                if candidate_score < best_candidate_score:
                    best_step = step
                    best_inv_damping = inv_damping_candidate
                    best_candidate_score = candidate_score
                    self.curvature_wrapper.accept_inverse_candidate()
                else:
                    self.curvature_wrapper.invalidate_inverse_candidate()

        self.inv_damping = best_inv_damping
        return best_step

    ############################################################################
    # Helpers
    ############################################################################

    def __apply_step(self, step):
        group = self.param_groups[0]
        for p, dp in zip(group['params'], step):
            p.data.add_(dp)

    def __inv_damping_candidates(self):
        if self.__should_update_inv_damping():
            return [
                self.inv_damping,
                self.inv_damping / self.inv_damping_factor,
                self.inv_damping * self.inv_damping_factor,
            ]
        else:
            return [self.inv_damping_factor]

    def __should_update_inverse(self):
        return self.__should_update_inv_damping() or (
                self.step_counter < 3 or
                (self.step_counter % self.update_interval_inversion == 0))

    def __should_update_inv_damping(self):
        return self.step_counter % self.update_interval_inv_damping == 0

    def __update_trust_damping_if_needed(self):
        should_update = ((
                                 self.step_counter % self.update_interval_trust_damping) == 0)

        if should_update:
            reduction_ratio = -self.curvature_wrapper.reduction_ratio(
                self.trust_damping, self.l2_reg)
            if reduction_ratio < .25:
                self.trust_damping /= self.trust_damping_factor
            elif reduction_ratio > .75:
                self.trust_damping *= self.trust_damping_factor

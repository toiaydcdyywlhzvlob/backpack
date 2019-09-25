"""
Damping strategy developed for KFAC

Optimizing Neural Networks with Kronecker-factored Approximate Curvature
James Martens and Roger Grosse
https://arxiv.org/abs/1503.05671
"""
from math import inf as INFINITY

import torch

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.hessianfree.rop import R_op

from .base_optimizer import BaseOptimizer
from .utils import flatten, inner_product, NUMERICAL_STABILITY_CONSTANT

MAGIC_FACTOR_FROM_KFAC_PAPER = 19. / 20.
DEBUG = True
DEBUG_STR = "DEBUG:                "


def debug(*message):
    if DEBUG:
        print(DEBUG_STR, *message)


class FancyDampingOptimizer(BaseOptimizer):
    def __init__(
            self,
            params,
            curvature,
            weight_decay=0.,  # eta
            initial_trust_damping=150.,  # lambda
            initial_inv_damping=None,  # gamma
            trust_damping_factor=None,  # ω1
            inv_damping_factor=None,  # ω2
            update_interval_trust_damping=5,  # T1
            update_interval_inv_damping=20,  # T2
            update_interval_inversion=20,  # T3
            minimum_damping=NUMERICAL_STABILITY_CONSTANT,
    ):
        """
        Implements the damping strategy developed for
        [KFAC](https://arxiv.org/abs/1503.05671)
        in a curvature-agnostic way;
        the approximation used for curvature is handled by a `CurvatureWrapper`.

        Args:
            params (iterable): iterable of parameters to optimize
            weight_decay (float, optional): L2 penalty (default: 0)
            initial_trust_damping (float, optional): damping parameter for the
                trust-region check of model quality (default: 150.)
            initial_inv_damping (float, optional): damping parameter for the
                inversion of curvature matrices
                (default: `(trust_damping, weight_decay`)
            trust_damping_factor (float, optional): multiplicative factor
                for changes in the trust damping factor
                (default: `(19/20)**update_interval_trust_damping`)
            inv_damping_factor (float, optional): multiplicative factor
                for changes in the inversion damping factor
                (default: `(19/20)**update_interval_inv_damping`)
            update_interval_trust_damping (int, optional): update trust_damping
                every _ steps (default: 5)
            update_interval_inv_damping (int, optional): update inv_damping
                every _ steps. Needs to be a multiple of
                `update_interval_inversion` (default: 20)
            update_interval_inversion (int, optional): update curvature inverse
                every _ steps (default: 20)


        """
        raise NotImplementedError("Not yet supported")

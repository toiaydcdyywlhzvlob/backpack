"""
Levenberg-Marquardt damping strategy.

Performs the update
𝜃 ← 𝜃 − 𝛾(G + (𝜆 + 𝜂)I)^{−1}g

g : Gradient of loss + L2 regularization
G : Curvature matrix
𝛾 : Step-size
𝜆 : Damping
𝜂 : L2 regularization

and updates the damping parameter using the LM heuristic

𝜌 ← actual improvement / predicted improvement
if 𝜌 < 1/4:
    𝜆 ← ω𝜆
if 𝜌 > 3/4
    𝜆 ← 𝜆/ω

where ω is a constant damping update factor
"""
import torch

from .base_optimizer import BaseOptimizer
from .utils import flatten, inner_product, NUMERICAL_STABILITY_CONSTANT

MAGIC_FACTOR_FROM_KFAC_PAPER = 19. / 20.
DEBUG = True
DEBUG_STR = "DEBUG:                "


def debug(*message):
    if DEBUG:
        print(DEBUG_STR, *message)


class LMOptimizer(BaseOptimizer):
    def __init__(
            self,
            params,
            curvature,
            lr=1.0,
            weight_decay=0.,  # eta
            initial_damping=150.,  # lambda
            damping_factor=None,  # ω1
            update_interval_damping=2,  # T1
            update_interval_inv=20,  # T3
            minimum_damping=NUMERICAL_STABILITY_CONSTANT,
    ):
        """
        """
        raise NotImplementedError("Not yet supported")

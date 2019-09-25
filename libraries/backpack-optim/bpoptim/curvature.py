"""
Curvature estimators

The role of those classes is to
- compute the curvature at each step.
- maintain a running average of the quantities.
- when needed, compute the inverse of the curvature using some damping.

"""

import math

from backpack import backpack
from backpack.extensions import KFAC, KFLR, KFRA, DiagGGNExact, DiagGGNMC
from backpack.extensions.secondorder.utils import multiply_vec_with_kron_facs
from backpack.utils.utils import einsum

from .utils import NUMERICAL_STABILITY_CONSTANT
from .inverse_curvature import (DiagonalInverseCurvature,
                                KroneckerInverseCurvature,
                                ScalarInverseCurvature)
from .moving_average import MovingAverage


class CurvatureEstimator:
    def __init__(self, param_groups):
        """
        param_groups: the param_groups of the optimizer using the estimator
        """
        self.param_groups = param_groups

    def compute_curvature(self, closure):
        """
        Update the running average of curvature.

        Closure should zero the gradients, do the forward pass
        and return the loss
        """
        raise NotImplementedError

    def compute_inverse(self, damping):
        """
        Returns an instance of InverseCurvature.

        `dampings` should be a list of dampings for each `param_groups`
        """
        raise NotImplementedError

    def multiply(self, damping, grad):
        """
        Multiply the grad vector by the damped curvature,
        where damping and grad are in [group[parameter]] format;
        ```
        grad = [           # for each group
            [              # for each parameter in the group
                vec,       # the value for that parameter
                ...
            ],
            ...
        ]
        ```
        """
        raise NotImplementedError


class ZeroCurvature(CurvatureEstimator):
    def __init__(self, param_groups):
        super().__init__(param_groups)
        self.curv = self._get_curv()

    def _get_curv(self):
        ZERO = 0.
        curv = list([
            list([ZERO for p in group['params']])
            for group in self.param_groups
        ])
        return curv

    def compute_curvature(self, closure, retain_graph=False):
        loss, output = closure()
        loss.backward(retain_graph=retain_graph)
        return loss, output

    def compute_inverse(self, damping):
        inv_curv = []

        for curv_group, damping_group in zip(self.curv, damping):
            inv_curv.append([])
            for curv_p in curv_group:
                inv_curv[-1].append(1 / (curv_p + damping_group))
        return ScalarInverseCurvature(inv_curv)

    def multiply(self, damping, grad):
        result = []
        for curv_group, damping_group, grad_group in zip(
                self.curv, damping, grad):
            result.append([])
            for curv_p, grad_p in zip(curv_group, grad_group):
                result[-1].append((curv_p + damping_group) * grad_p)
        return result


class BackpackCurvatureEstimator(CurvatureEstimator):
    def __init__(self, param_groups, bp_extension_cls, use_factors):
        super().__init__(param_groups)
        self.moving_average = MovingAverage(use_factors=use_factors)
        self.bp_extension_cls = bp_extension_cls

    def compute_curvature(self, closure, retain_graph=False):
        """Data structure for moving average is supported by backpack."""
        bp_extension = self.bp_extension_cls()
        bp_savefield = bp_extension.savefield

        with backpack(bp_extension):
            loss, output = closure()
            loss.backward(retain_graph=retain_graph)

            input_to_moving_average = list([
                list([getattr(p, bp_savefield) for p in group['params']])
                for group in self.param_groups
            ])

            self.moving_average.step(input_to_moving_average)

        return loss, output


class DiagCurvatureBase(BackpackCurvatureEstimator):
    def __init__(self, param_groups, bp_extension_cls):
        use_factors = False
        super().__init__(param_groups, bp_extension_cls, use_factors)

    def compute_inverse(self, damping):
        curv = self.moving_average.get()

        inv_curv = []
        for curv_group, damping_group in zip(curv, damping):
            inv_curv.append([])
            for curv_p in curv_group:
                inv_curv[-1].append(1 / (curv_p + damping_group))
        return DiagonalInverseCurvature(inv_curv)

    def multiply(self, damping, grad):
        curv = self.moving_average.get()
        result = []
        for curv_group, damping_group, grad_group in zip(curv, damping, grad):
            result.append([])
            for curv_p, grad_p in zip(curv_group, grad_group):
                result[-1].append((curv_p + damping_group) * grad_p)
        return result


class DiagGGNExactCurvature(DiagCurvatureBase):
    def __init__(self, param_groups):
        super().__init__(param_groups, DiagGGNExact)


class DiagGGNMCCurvature(DiagCurvatureBase):
    def __init__(self, param_groups):
        super().__init__(param_groups, DiagGGNMC)


class KroneckerFactoredCurvature(BackpackCurvatureEstimator):
    def __init__(self, param_groups, bp_extension_cls):
        use_factors = True
        super().__init__(param_groups, bp_extension_cls, use_factors)

    def multiply(self, damping, grad):
        curv = self.moving_average.get()
        result = []
        for curv_group, damping_group, grad_group in zip(curv, damping, grad):
            result.append([])
            for curv_p, grad_p in zip(curv_group, grad_group):
                damp_adapted_grad = damping_group * grad_p

                # TODO: avoid view (currently requires flattened vectors)
                grad_p_flat = grad_p.view(-1)
                curv_adapted_grad = multiply_vec_with_kron_facs(
                    curv_p, grad_p_flat)
                curv_adapted_grad = curv_adapted_grad.view_as(grad_p)

                result[-1].append(damp_adapted_grad + curv_adapted_grad)
        return result

    def compute_inverse(self, damping):
        curv = self.moving_average.get()

        inv_curv = []
        for curv_group, damping_group in zip(curv, damping):
            inv_curv.append([])
            for curv_p in curv_group:
                if len(curv_p) == 1:
                    # TODO: Not defined by KFAC
                    # HOTFIX: Just invert, shift by damping, no Tikhonov
                    shift = damping_group
                    kfac = curv_p[0]
                    inv_kfac = self.__inverse(kfac, shift=shift)

                    inv_curv[-1].append([inv_kfac])

                elif len(curv_p) == 2:
                    # G, A in Martens' notation (different order due to row-major)
                    kfac2, kfac1 = curv_p

                    # Tikhonov
                    pi = self.__compute_tikhonov_factor(kfac1, kfac2)
                    # shift for factor 1: pi * sqrt(gamma  + eta) = pi * sqrt(gamma)
                    shift1 = pi * math.sqrt(damping_group)
                    # factor 2: 1 / pi * sqrt(gamma  + eta) = 1 / pi * sqrt(gamma)
                    shift2 = 1. / pi * math.sqrt(damping_group)

                    # invert, take into account the diagonal term
                    inv_kfac1 = self.__inverse(kfac1, shift=shift1)
                    inv_kfac2 = self.__inverse(kfac2, shift=shift2)

                    inv_curv[-1].append([inv_kfac2, inv_kfac1])
                else:
                    raise ValueError(
                        "Got {} Kronecker factors, can only handle <= 2".
                        format(len(curv_p)))

        return KroneckerInverseCurvature(inv_curv)

    def __compute_tikhonov_factor(self, kfac1, kfac2):
        """Scalar pi from trace norm for factored Tikhonov regularization.

        For details, see Section 6.3 of the KFAC paper.

        TODO: Allow for choices other than trace norm.
        """
        (dim1, _), (dim2, _) = kfac1.shape, kfac2.shape
        trace1, trace2 = kfac1.trace(), kfac2.trace()
        pi_squared = (trace1 / dim1) / (trace2 / dim2)
        return pi_squared.sqrt()

    def __inverse(self, sym_mat, shift):
        """Invert sym_mat + shift * I"""
        eigvals, eigvecs = self.__eigen(sym_mat)
        # account for diagonal term added to the matrix
        eigvals.add_(shift)
        return self.__inv_from_eigen(eigvals, eigvecs)

    def __eigen(self, sym_mat):
        """Return eigenvalues and eigenvectors from eigendecomposition."""
        eigvals, eigvecs = sym_mat.symeig(eigenvectors=True)
        return eigvals, eigvecs

    def __inv_from_eigen(self, eigvals, eigvecs, truncate=NUMERICAL_STABILITY_CONSTANT):
        inv_eigvals = 1. / eigvals
        inv_eigvals.clamp_(min=0., max=1. / truncate)
        # return inv_eigvals, eigvecs
        return einsum('ij,j,kj->ik', (eigvecs, inv_eigvals, eigvecs))


class KFACCurvature(KroneckerFactoredCurvature):
    """Kronecker factorization by Martens."""
    def __init__(self, param_groups):
        bp_extension_cls = KFAC
        super().__init__(param_groups, bp_extension_cls)


class KFLRCurvature(KroneckerFactoredCurvature):
    """Kronecker factored low-rank approximation by Botev."""
    def __init__(self, param_groups):
        bp_extension_cls = KFLR
        super().__init__(param_groups, bp_extension_cls)


class KFRACurvature(KroneckerFactoredCurvature):
    """Kronecker factored recursive approximation by Botev."""
    def __init__(self, param_groups):
        bp_extension_cls = KFRA
        super().__init__(param_groups, bp_extension_cls)

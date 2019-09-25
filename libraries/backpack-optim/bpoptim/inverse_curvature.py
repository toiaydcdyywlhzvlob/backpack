"""
Inverse Curvature

General class to hold the inverse of an approximation of the curvature.
Only provides one method, the multiplication with a vector.
"""

from backpack.extensions.secondorder.utils import multiply_vec_with_kron_facs
from backpack.utils.utils import einsum


class InverseCurvature:
    def __init__(self, inv_curv):
        self.inv_curv = inv_curv

    def multiply(self, grad):
        """
        `vec` is expected to be in [group[parameter]] format;
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
        results = []
        for inv_curv_group, grad_group in zip(self.inv_curv, grad):
            results.append(self._multiply_group(inv_curv_group, grad_group))
        return results

    def _multiply_group(self, inv_curv_group, grad_group):
        raise NotImplementedError


class DiagonalInverseCurvature(InverseCurvature):
    def __init__(self, inv_curv):
        """
        `inv_curv` is expected to be in [group[parameter]] format;
        ```
        inv_curv = [       # for each group
            [              # for each parameter in the group
                vec,       # the value for that parameter
                ...
            ],
            ...
        ]
        ```
        """
        super().__init__(inv_curv)

    def _multiply_group(self, inv_curv_group, grad_group):
        group_results = []
        for inv_curv_p, grad_p in zip(inv_curv_group, grad_group):
            group_results.append(inv_curv_p * grad_p)
        return group_results


class ScalarInverseCurvature(DiagonalInverseCurvature):
    def __init__(self, inv_curv):
        """
        `inv_curv` is expected to be in [group[parameter]] format;
        ```
        inv_curv = [       # for each group
            [              # for each parameter in the group
                scalar,    # the value for that parameter
                ...
            ],
            ...
        ]
        ```
        """
        super().__init__(inv_curv)


class KroneckerInverseCurvature(InverseCurvature):
    def __init__(self, inv_curv):
        """
        `inv_curv` is expected to be in [group[parameter[kroneckers]]] format;
        ```
        inv_curv = [             # for each group
            [                    # for each parameter in the group
                [                # for each kronecker factor for the parameter
                    kfac_1_p_11, # the value of that kronecker factor
                    ...
                ],
                ...
            ],
            ...
        ]
        ```
        """
        super().__init__(inv_curv)

    def _multiply_group(self, inv_curv_group, grad_group):
        group_results = []
        for inv_curv_p, grad_p in zip(inv_curv_group, grad_group):
            # TODO: avoid view (currently requires flattened vectors)
            grad_p_flat = grad_p.view(-1)
            curv_adapted_grad = multiply_vec_with_kron_facs(
                inv_curv_p, grad_p_flat)
            curv_adapted_grad = curv_adapted_grad.view_as(grad_p)
            group_results.append(curv_adapted_grad)
        return group_results

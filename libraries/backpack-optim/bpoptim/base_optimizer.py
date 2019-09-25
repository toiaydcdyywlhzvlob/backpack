"""
Base class containing the helper functions for the optimizers.
"""
from torch import norm
from torch.optim.optimizer import Optimizer

from .utils import flatten, inner_product


class BaseOptimizer(Optimizer):

    def __init__(self, params, defaults, curvature):
        self.param_groups = None
        super().__init__(params, defaults=defaults)

        # Create curv after __init__: need the param_groups to exists
        self.curv = curvature(self.param_groups)
        self.inv_curv = None

        # information about the last step
        self.step_info = {}

    def regularized_neg_grad(self):
        """
        :return: The regularized gradient in param-group format
        """
        return [
            [
                - p.grad.data - group['weight_decay'] * p.data
                for p in group["params"]
            ] for group in self.param_groups
        ]

    def projection_on_gradient(self, step_proposal):
        """
        :return: The projection of the step proposal, in param-group format,
        on the regularized gradient
        """
        return - inner_product(
            flatten(step_proposal),
            flatten(self.regularized_neg_grad())
        )

    def model_l2_norm(self):
        """
        :return: The regularization term - L2 norm of the parameters multiplied
        by the weight decay
        """
        l2_norm = 0
        for group in self.param_groups:
            for p in group["params"]:
                l2_norm += group["weight_decay"] * norm(p.data)
        return l2_norm

    def apply_step(self, step, global_lr=None):
        """
        Apply the step, in param-group format, to the parameters of the model
        """
        for group, step_group in zip(self.param_groups, step):
            for p, dp in zip(group['params'], step_group):
                if global_lr is None:
                    p.data.add_(dp)
                else:
                    p.data.add_(global_lr, dp)

    def log_step_info(self, key, value):
        """
        Raises exception if entry already exists.
        """
        if key in self.step_info.keys():
            raise ValueError("{} would be overwritten.".format(key))
        self.step_info[key] = value

    def clear_step_info(self):
        self.step_info = {}

    def get_step_info(self):
        return self.step_info

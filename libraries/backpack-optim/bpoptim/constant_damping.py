from .base_optimizer import BaseOptimizer


class ConstantDampingOptimizer(BaseOptimizer):
    def __init__(self,
                 params,
                 curvature,
                 lr=1.,
                 damping=.1,
                 weight_decay=0.,
                 update_interval_inv=20):

        self.validate_hyperparameters(lr, damping, weight_decay,
                                      update_interval_inv)

        self.damping = damping
        self.update_interval_inversion = update_interval_inv
        self.step_counter = 0

        super().__init__(params,
                         defaults=dict(lr=lr,
                                       damping=damping,
                                       weight_decay=weight_decay),
                         curvature=curvature)
        self.group_dampings = list([
            group['damping'] + group['weight_decay']
            for group in self.param_groups
        ])

    def validate_hyperparameters(self, lr, damping, weight_decay,
                                 update_interval_inv):
        return True

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Need a closure")

        self.clear_step_info()
        self.log_step_info("damping", self.damping)
        self.zero_grad()

        loss, _ = self.curv.compute_curvature(closure)
        self.log_step_info("batch_loss_before_step", loss.item())

        if self.should_compute_inverse():
            self.inv_curv = self.curv.compute_inverse(self.group_dampings)

        step_proposal = self.inv_curv.multiply([[
            (-p.grad.data - group['weight_decay'] * p.data) * group["lr"]
            for p in group["params"]
        ] for group in self.param_groups])

        self.apply_step(step_proposal)
        return loss

    def should_compute_inverse(self):
        return self.step_counter < 3 or self.step_counter % self.update_interval_inversion == 0

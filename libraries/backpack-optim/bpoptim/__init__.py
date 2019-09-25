"""
Pytorch optimizers based on BackPACK.
"""

from .constant_damping import ConstantDampingOptimizer
from .curvature import (DiagGGNExactCurvature, DiagGGNMCCurvature,
                        KFACCurvature, KFLRCurvature, KFRACurvature,
                        ZeroCurvature)
from .fancy_damping import FancyDampingOptimizer
from .lm_damping import LMOptimizer

##############################################################################
# Exact DiagGGN optimizers                                                   #
##############################################################################


def DiagGGNConstantDampingOptimizer(params, lr=1, damping=0.1, **kwargs):
    return ConstantDampingOptimizer(params,
                                    DiagGGNExactCurvature,
                                    lr=lr,
                                    damping=damping,
                                    **kwargs)


def DiagGGNLMOptimizer(params, lr=1, initial_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return LMOptimizer(params,
                       DiagGGNExactCurvature,
                       lr=lr,
                       initial_damping=initial_damping,
                       **kwargs)


def DiagGGNFancyDampingOptimizer(params, initial_trust_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return FancyDampingOptimizer(params,
                                 DiagGGNExactCurvature,
                                 initial_trust_damping=initial_trust_damping,
                                 **kwargs)


##############################################################################
# MC-sampled DiagGGN optimizers                                              #
##############################################################################


def DiagGGNMCConstantDampingOptimizer(params, lr=1, damping=0.1, **kwargs):
    return ConstantDampingOptimizer(params,
                                    DiagGGNMCCurvature,
                                    lr=lr,
                                    damping=damping,
                                    **kwargs)


def DiagGGNMCLMOptimizer(params, lr=1, initial_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return LMOptimizer(params,
                       DiagGGNMCCurvature,
                       lr=lr,
                       initial_damping=initial_damping,
                       **kwargs)


def DiagGGNMCFancyDampingOptimizer(params,
                                   initial_trust_damping=150.,
                                   **kwargs):
    raise NotImplementedError("Not yet supported")
    return FancyDampingOptimizer(params,
                                 DiagGGNMCCurvature,
                                 initial_trust_damping=initial_trust_damping,
                                 **kwargs)


##############################################################################
# Zero curvature optimizers                                                  #
##############################################################################


def ZeroConstantDampingOptimizer(params, lr=1, damping=0.1, **kwargs):
    return ConstantDampingOptimizer(params,
                                    ZeroCurvature,
                                    lr=lr,
                                    damping=damping,
                                    **kwargs)


def ZeroLMOptimizer(params, lr=1, initial_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return LMOptimizer(params,
                       ZeroCurvature,
                       lr=lr,
                       initial_damping=initial_damping,
                       **kwargs)


def ZeroFancyDampingOptimizer(params, initial_trust_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return FancyDampingOptimizer(params,
                                 ZeroCurvature,
                                 initial_trust_damping=initial_trust_damping,
                                 **kwargs)


##############################################################################
# KFAC curvature optimizers                                                  #
##############################################################################


def KFACConstantDampingOptimizer(params, lr=1, damping=0.1, **kwargs):
    return ConstantDampingOptimizer(params,
                                    KFACCurvature,
                                    lr=lr,
                                    damping=damping,
                                    **kwargs)


def KFACLMOptimizer(params, lr=1, initial_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return LMOptimizer(params,
                       KFACCurvature,
                       lr=lr,
                       initial_damping=initial_damping,
                       **kwargs)


def KFACFancyDampingOptimizer(params, initial_trust_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return FancyDampingOptimizer(params,
                                 KFACCurvature,
                                 initial_trust_damping=initial_trust_damping,
                                 **kwargs)


##############################################################################
# KFLR curvature optimizers                                                  #
##############################################################################


def KFLRConstantDampingOptimizer(params, lr=1, damping=0.1, **kwargs):
    return ConstantDampingOptimizer(params,
                                    KFLRCurvature,
                                    lr=lr,
                                    damping=damping,
                                    **kwargs)


def KFLRLMOptimizer(params, lr=1, initial_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return LMOptimizer(params,
                       KFLRCurvature,
                       lr=lr,
                       initial_damping=initial_damping,
                       **kwargs)


def KFLRFancyDampingOptimizer(params, initial_trust_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return FancyDampingOptimizer(params,
                                 KFLRCurvature,
                                 initial_trust_damping=initial_trust_damping,
                                 **kwargs)


##############################################################################
# KFRA curvature optimizers                                                  #
##############################################################################


def KFRAConstantDampingOptimizer(params, lr=1, damping=0.1, **kwargs):
    return ConstantDampingOptimizer(params,
                                    KFRACurvature,
                                    lr=lr,
                                    damping=damping,
                                    **kwargs)


def KFRALMOptimizer(params, lr=1, initial_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return LMOptimizer(params,
                       KFRACurvature,
                       lr=lr,
                       initial_damping=initial_damping,
                       **kwargs)


def KFRAFancyDampingOptimizer(params, initial_trust_damping=150., **kwargs):
    raise NotImplementedError("Not yet supported")
    return FancyDampingOptimizer(params,
                                 KFRACurvature,
                                 initial_trust_damping=initial_trust_damping,
                                 **kwargs)


"""
# How the parameters, steps and second-order approximations are stored


One key concept to understand the constructs in this package
is how Pytorch deals with parameters that require different hyperparameters.
In this context, a hyperparameter would be a learning rate or l2 regularization,
while the parameters are the weights and biases of the optimized model.


If the optimizer applies the same hyperparameters to all parameters,
the optimizer is constructed using
```
opt = MyOptimizer(model.parameters(), lr=0.1, weight_decay=0.001)
```
But a common use-case is to not apply l2 regularization to the biases.
In this setting, the optimizer is constructed with
```
params = [
    {'params': biases, 'weight_decay':0},
    {'params': weights}
]
opt = MyOptimizer(params, lr=0.1, weight_decay=0.001)
```
`biases` and `weights` are iterables of tensors, the parameters of the model.
The biases will use `weight_decay=0`, while the weights will use the default, `0.001`.


The optimizer can access those groups of parameters using `self.param_groups`.
For the optimizer above, this attribute would be
```
self.param_groups = [
    {'params': biases, 'weight_decay': 0, 'lr': 0.1},
    {'params': weights, 'weight_decay': 0.001, 'lr': 0.1},
]
```

To work with this structure and second order methods, the steps and
curvature approximations are stored as a list of list of tensors.
For the model described above, a step would be
```
step = [                   # a list containing
    [bias_step_1, ...],    # a list of steps, or curvature, for all biases
    [bias_weight_1, ...],  # a list of steps, or curvature, for all weights
]
```
See `curvature.py`, `inverse_curvature.py` and `moving_average.py`
for more details.

"""
"""

# How an optimizer is constructed

┌───────────────────────────────────────────────────────────────────┐
│ The Optimizer class from Pytorch is extended by a Damping Scheme. │
│                                                                   │
│ - It maintains a damping parameter for the Curvature object       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ Curvature                                            │         │
│  │                                                      │         │
│  │ - Compute the curvature using BackPACK               │         │
│  │ - Compute a damped inverse on demand                 │         │
│  │                                                      │         │
│  │  ┌────────────────────────────────────────────────┐  │         │
│  │  │ MovingAverage                                  │  │         │
│  │  │                                                │  │         │
│  │  │ - Maintain a moving average of the curvature   │  │         │
│  │  │                                                │  │         │
│  │  └────────────────────────────────────────────────┘  │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                   │
│ - The damping scheme can query the curvature to get a             │
│   CurvatureInverse object                                         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ CurvatureInverse                                     │         │
│  │                                                      │         │
│  │ - Given a vector, left-multiply by the inverse       │         │
│  │   of the curvature.                                  │         │
│  │                                                      │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                   │
│ - The damping scheme can use the curvature inverse on the         │
│   gradient to get a direction proposal, and might do              │
│   additional work before applying the update.                     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

"""

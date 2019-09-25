import pprint

import bpoptim

from .grid_search import BPGridSearch
from .tuning import (TuningConstantDamping, TuningConstantDampingNoCurvature,
                     TuningDiagGGNExact, TuningDiagGGNMC, TuningFancyDamping,
                     TuningKFAC, TuningKFLR, TuningKFRA, TuningLMDamping,
                     TuningZero)

PROBLEMS = [
    'mnist_logreg',
    'fmnist_2c2d',
    'cifar10_3c3d',
    'cifar100_allcnnc',
]

BATCH_SIZES = [
    None,
    # 64,
    # 128,
    # 256,
    # 512,
]


class GridSearchFactory():
    Zero = "Zero"
    DiagGGNExact = "DiagGGN"
    DiagGGNMC = "DiagGGN_MC"
    KFAC = "KFAC"
    KFLR = "KFLR"
    KFRA = "KFRA"

    CURVATURES = [
        Zero,
        DiagGGNExact,
        DiagGGNMC,
        KFAC,
        KFLR,
        KFRA,
    ]

    CURVATURES_TUNING = {
        Zero: TuningZero,
        DiagGGNExact: TuningDiagGGNExact,
        DiagGGNMC: TuningDiagGGNMC,
        KFAC: TuningKFAC,
        KFLR: TuningKFLR,
        KFRA: TuningKFRA
    }

    CONSTANT = "const"
    LM = "LM"
    FANCY = "fancy"

    DAMPINGS = [CONSTANT, LM, FANCY]

    DAMPINGS_TUNING = {
        CONSTANT: TuningConstantDamping,
        LM: TuningLMDamping,
        FANCY: TuningFancyDamping
    }

    DAMPED_OPTIMS = {
        (Zero, CONSTANT): bpoptim.ZeroConstantDampingOptimizer,
        (Zero, LM): bpoptim.ZeroLMOptimizer,
        (Zero, FANCY): bpoptim.ZeroFancyDampingOptimizer,
        (DiagGGNExact, CONSTANT): bpoptim.DiagGGNConstantDampingOptimizer,
        (DiagGGNExact, LM): bpoptim.DiagGGNLMOptimizer,
        (DiagGGNExact, FANCY): bpoptim.DiagGGNFancyDampingOptimizer,
        (DiagGGNMC, CONSTANT): bpoptim.DiagGGNMCConstantDampingOptimizer,
        (DiagGGNMC, LM): bpoptim.DiagGGNMCLMOptimizer,
        (DiagGGNMC, FANCY): bpoptim.DiagGGNMCFancyDampingOptimizer,
        (KFAC, CONSTANT): bpoptim.KFACConstantDampingOptimizer,
        (KFAC, LM): bpoptim.KFACLMOptimizer,
        (KFAC, FANCY): bpoptim.KFACFancyDampingOptimizer,
        (KFLR, CONSTANT): bpoptim.KFLRConstantDampingOptimizer,
        (KFLR, LM): bpoptim.KFLRLMOptimizer,
        (KFLR, FANCY): bpoptim.KFLRFancyDampingOptimizer,
        (KFRA, CONSTANT): bpoptim.KFRAConstantDampingOptimizer,
        (KFRA, LM): bpoptim.KFRALMOptimizer,
        (KFRA, FANCY): bpoptim.KFRAFancyDampingOptimizer,
    }

    def make_grid_search(self,
                         curv_str,
                         damping_str,
                         deepobs_problem,
                         output_dir="../grid_search",
                         generation_dir="../grid_search_command_scripts"):
        optim_cls = self._get_damped_optimizer(curv_str, damping_str)
        tune_curv, tune_damping = self.get_tunings(curv_str, damping_str)

        return BPGridSearch(deepobs_problem,
                            optim_cls,
                            tune_curv,
                            tune_damping,
                            output_dir=output_dir,
                            generation_dir=generation_dir)

    def get_tunings(self, curv_str, damping_str):
        tune_curv = self._get_curv_tuning(curv_str)
        tune_damping = self._get_damping_tuning(damping_str)

        # no tuning of damping parameter for constant damping and no curvature
        if curv_str == self.Zero and damping_str == self.CONSTANT:
            tune_damping = TuningConstantDampingNoCurvature()

        return tune_curv, tune_damping

    def get_curvature_and_damping(self, optim_cls):
        """Return (curvature, damping) from the optimizer class."""
        for (curv, damp), cls in self.DAMPED_OPTIMS.items():
            if optim_cls == cls:
                return (curv, damp)
        raise ValueError(
            "No (curvature, damping) found for {}".format(optim_cls))

    def get_all_optim_classes(self):
        return [optim_cls for (_, optim_cls) in self.DAMPED_OPTIMS.items()]

    def _get_damped_optimizer(self, curv_str, damping_str):
        key = self._check_damped_optim_exists(curv_str, damping_str)
        return self.DAMPED_OPTIMS[key]

    def _get_damping_tuning(self, damping_str):
        key = self._check_damping_tuning_exists(damping_str)
        return self.DAMPINGS_TUNING[key]()

    def _get_curv_tuning(self, curv_str):
        key = self._check_curv_tuning_exists(curv_str)
        return self.CURVATURES_TUNING[key]()

    def _check_curv_tuning_exists(self, curv_str):
        if curv_str not in self.CURVATURES_TUNING.keys():
            raise ValueError(
                "Curvature tuning {} not registered. Supported: {}.".format(
                    curv_str, pprint.pformat(self.CURVATURES_TUNING.keys())))
        return curv_str

    def _check_damping_tuning_exists(self, damping_str):
        if damping_str not in self.DAMPINGS_TUNING.keys():
            raise ValueError(
                "Damping tuning {} not registered. Supported: {}.".format(
                    damping_str, pprint.pformat(self.DAMPINGS_TUNING.keys())))
        return damping_str

    def _check_damped_optim_exists(self, curv_str, damping_str):
        key = (curv_str, damping_str)
        if key not in self.DAMPED_OPTIMS.keys():
            raise ValueError(
                "Damped optimizer {} not registered. Supported: {}.".format(
                    key, pprint.pformat(self.DAMPED_OPTIMS.keys())))
        return key

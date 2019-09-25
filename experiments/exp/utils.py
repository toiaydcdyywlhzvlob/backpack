"jk" "Utility functions and settings." ""

##############################################################################
# Settings                                                                   #
import os

##############################################################################
# grid search
from bp_dops_integration.experiments import (BATCH_SIZES, PROBLEMS,
                                             GridSearchFactory)

DAMPINGS = GridSearchFactory.DAMPINGS
CURVATURES = GridSearchFactory.CURVATURES

# best runs
SEEDS = list(range(42, 52))
# available modes: "final", "best"
# available metrics: "train_losses", "train_accuracies",
#                    "test_losses", "test_accuracies",
#                    "valid_losses", "valid_accuracies"
BEST_STRATEGY = [
    ("final", "valid_accuracies"),
    # ("best", "valid_accuracies"),
]

# plots
BASELINES = "../baselines/"
PLOTS_REL_PATH = "../plots/"


def baseline_dir(problem):
    return os.path.join(BASELINES, problem)


##############################################################################
# Plotting utilities                                                         #
##############################################################################


def plot_filename(curvature,
                  damping,
                  problem,
                  best_mode,
                  best_metric,
                  rel_path=None,
                  append_to_rel_path=None):
    if rel_path is None:
        rel_path = PLOTS_REL_PATH
    if append_to_rel_path is None:
        append_to_rel_path = ""

    filename = "{}_{}_{}_{}".format(
        curvature,
        damping,
        best_mode,
        best_metric,
    )
    return os.path.join(rel_path, append_to_rel_path, problem, filename)


##############################################################################
# Excluded settings                                                          #
##############################################################################


def allowed_best_runs(filter_func=None):
    """Return all combinations (curv, damp, prob, mode, metric) parametrizing
    the allowed best runs.
    """
    combinations = []
    for (curv, damp, prob) in allowed_combinations(filter_func=filter_func):
        for (mode, metric) in BEST_STRATEGY:
            combinations.append((curv, damp, prob, mode, metric))
    return combinations


def allowed_curvatures(damping, problem, filter_func=None):
    """Return all allowed curvatures for fixed damping scheme and problem."""
    combinations = _allowed_combinations([damping], [problem],
                                         CURVATURES,
                                         filter_func=filter_func)
    curvatures = [curv for (curv, damp, prob) in combinations]
    return sorted(list(set(curvatures)))


def allowed_dampings(curvature, problem, filter_func=None):
    """Return all allowed dampings for fixed curvature and problem."""
    combinations = _allowed_combinations(DAMPINGS, [problem], [curvature],
                                         filter_func=filter_func)
    dampings = [damp for (curv, damp, prob) in combinations]
    return sorted(list(set(dampings)))


def allowed_combinations(filter_func=None):
    """Return tuples (curv, damp, prob) of allowed configs."""
    return _allowed_combinations(DAMPINGS,
                                 PROBLEMS,
                                 CURVATURES,
                                 filter_func=filter_func)


def _allowed_combinations(dampings, problems, curvatures, filter_func=None):
    """
    Allow filtering by specifying `filter_func`. It maps a tuple of strings
    for (curvature, damping, problem) to a boolean value which specifies
    whether the experiment should be included or not
    """
    if filter_func is None:

        def default_filter_func(curv, damp, prob):
            return True

        filter_func = default_filter_func

    allow = []
    for damp in dampings:
        for prob in problems:
            for curv in curvatures:
                skip_due_filter = not filter_func(curv, damp, prob)
                if skip_due_filter:
                    _print_exclude_message(damp, prob, curv, filter_func)
                    continue

                if not _exclude(damp, prob, curv):
                    allow.append((curv, damp, prob))
    return allow


def _exclude(damping, problem, curvature):
    """Return whether the run should be excluded."""
    def is_zero(curvature):
        return curvature == GridSearchFactory.Zero

    def is_kfra(curvature):
        return curvature == GridSearchFactory.KFRA

    def is_diag_ggn_exact(curvature):
        return curvature == GridSearchFactory.DiagGGNExact

    def is_kflr(curvature):
        return curvature == GridSearchFactory.KFLR

    def is_fmnist_2c2d(problem):
        return problem == "fmnist_2c2d"

    def is_cifar10_3c3d(problem):
        return problem == "cifar10_3c3d"

    def is_cifar100_allcnnc(problem):
        return problem == "cifar100_allcnnc"

    def is_fancy(damping):
        return damping == GridSearchFactory.FANCY

    def is_lm(damping):
        return damping == GridSearchFactory.LM

    def exclude_zero(damping, problem, curvature):
        return is_zero(curvature)

    def exclude_lm_and_fancy(damping, problem, curvature):
        lm = is_lm(damping)
        fancy = is_fancy(damping)
        return lm or fancy

    def exclude_KFRA_for_fmnist_cifar10_cifar100(damping, problem, curvature):
        kfra = is_kfra(curvature)
        fmnist = is_fmnist_2c2d(problem)
        cifar10 = is_cifar10_3c3d(problem)
        cifar100 = is_cifar100_allcnnc(problem)
        return kfra and (cifar10 or cifar100 or fmnist)

    def exclude_DiagGGNExact_for_cifar100(damping, problem, curvature):
        diaggn = is_diag_ggn_exact(curvature)
        cifar100 = is_cifar100_allcnnc(problem)
        return diaggn and cifar100

    def exclude_KFLR_for_cifar100(damping, problem, curvature):
        kflr = is_kflr(curvature)
        cifar100 = is_cifar100_allcnnc(problem)
        return kflr and cifar100

    # add more criteria to exclude runs
    criteria = [
        exclude_zero,
        exclude_lm_and_fancy,
        exclude_KFRA_for_fmnist_cifar10_cifar100,
        exclude_DiagGGNExact_for_cifar100,
        exclude_KFLR_for_cifar100,
    ]

    for criterion in criteria:
        if criterion(damping, problem, curvature) is True:
            _print_exclude_message(damping, problem, curvature, criterion)
            return True
    return False


def _print_exclude_message(damping, problem, curvature, criterion):
    print("Exclude run {} {} {} by criterion {}".format(
        damping, problem, curvature, criterion.__name__))

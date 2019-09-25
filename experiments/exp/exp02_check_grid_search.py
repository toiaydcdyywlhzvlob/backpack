"""Check that grid search is complete."""
import re
from contextlib import redirect_stdout
from io import StringIO

import pandas

import deepobs.analyzer.analyze as analyze
from exp01_grid_search import BATCH_SIZES, create_grid_search
from utils import PROBLEMS


def multi_batch_grid_dims(filter_func=None):
    """Dimension of the grid, including the batch sizes."""
    experiments = create_grid_search(filter_func)
    num_batch_sizes = len(BATCH_SIZES)

    return [
        num_batch_sizes * experiment.get_grid_dim()
        for experiment in experiments
    ]


def filter_problem(target_problem):
    def filter_func(curv, damp, prob):
        return prob == target_problem

    return filter_func


##############################################################################
# Utility functions                                                          #
##############################################################################
def print_dashed_line():
    print("-" * 78)


def check_manually(path):
    """Print to console."""
    print("\nMANUAL CHECK\n")
    analyze.check_output(path)


def check_complete(path, filter_func=None):
    print("\nCHECK PROBLEMS AND OPTIMS COMPLETE\n")
    statistics = _get_statistics_from_deepobs_check(path)

    # pairs (problem, optim_cls)
    optims_problems = [(item[0], item[1]) for item in statistics]

    passing = True
    for experiment in create_grid_search(filter_func):
        optim_cls = experiment._get_optim_name()
        problem = experiment.get_deepobs_problem()

        item = (problem, optim_cls)
        if item not in optims_problems:
            passing = False
            print("{:55} missing".format(str(item)))
        else:
            print("{:55} found".format(str(item)))
    return passing


def check_each_run_has_num_seeds(expected_num_seeds, path):
    print("\nCHECK EACH RUN HAS {} SEEDS\n".format(expected_num_seeds))
    passing = True
    for stat in _get_statistics_from_deepobs_check(path):
        _, _, _, num_seeds = stat

        if not num_seeds == expected_num_seeds:
            passing = False
            print("{:70} Expect num_seeds {}, got {}".format(
                str(stat), expected_num_seeds, num_seeds))
        else:
            print("{:70} passed".format(str(stat)))
    return passing


def _check_output_as_str(path):
    with StringIO() as buf, redirect_stdout(buf):
        analyze.check_output(path)
        check_str = buf.getvalue()
    return check_str


def _parse_deepobs_check_output_str(string):
    regex = r'(.*) \| (.*)\: (.*) setting\(s\) with (.*) seed\(s\)\.'
    match = re.match(regex, string)
    problem, optim_cls, num_settings, num_seeds = match.groups()
    return problem, optim_cls, int(num_settings), int(num_seeds)


def _get_statistics_from_deepobs_check(path):
    check_str = _check_output_as_str(path)
    stats = []
    for line in check_str.splitlines():
        (problem, optim_cls, num_settings,
         num_seeds) = _parse_deepobs_check_output_str(line)
        stats.append((problem, optim_cls, num_settings, num_seeds))
    return stats


##############################################################################
# Grid search specific tests                                                 #
##############################################################################
def check_grid_search_manually():
    check_manually(get_results_path())


def check_grid_search_each_run_has_1_seed():
    return check_each_run_has_num_seeds(1, get_results_path())


def check_grid_search_complete(filter_func=None):
    print("\nCHECK GRID SEARCH COMPLETE\n")
    statistics = _get_statistics_from_deepobs_check(get_results_path())

    # key-value pairs (problem, optim_cls) : num_settings
    optims_problems_settings = {(item[0], item[1]): item[2]
                                for item in statistics}

    passing = True
    for experiment, grid_dim in zip(create_grid_search(filter_func),
                                    multi_batch_grid_dims(filter_func)):
        optim_cls = experiment._get_optim_name()
        problem = experiment.get_deepobs_problem()
        key = (problem, optim_cls)

        try:
            dim = optims_problems_settings[key]
        except KeyError:
            print("{:55} Expect grid dim {}, but no data found".format(
                str(key), grid_dim))
            passing = False
            continue

        if dim != grid_dim:
            passing = False
            print("{:55} Expect grid dim {}, but found {}".format(
                str(key), grid_dim, dim))
        else:
            print("{} passed".format(key))
    return passing


def get_results_path():
    first_grid_search = create_grid_search()[0]
    results_path = first_grid_search.get_output_dir()
    return results_path


def summarize_grid_search_progress():
    headers = ["Problem", "Found all", "Grid complete"]
    result = []
    for problem in PROBLEMS:
        filter_func = filter_problem(problem)
        complete = check_complete(get_results_path(), filter_func)
        grid_complete = check_grid_search_complete(filter_func)
        result.append([problem, complete, grid_complete])
    result = pandas.DataFrame(result, columns=headers)
    return result


if __name__ == "__main__":
    check_grid_search_manually()
    print_dashed_line()
    single_seed = check_grid_search_each_run_has_1_seed()
    print_dashed_line()
    summary = summarize_grid_search_progress()

    print_dashed_line()
    print("GRID SEARCH TEST SUMMARY\n")
    print(summary.to_string(index=False))
    print("\nFound runs evaluated on single seed: {}".format(single_seed))
    print_dashed_line()

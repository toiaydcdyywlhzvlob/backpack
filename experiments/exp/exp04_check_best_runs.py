"""Check that best runs are complete."""
import pandas

from bp_dops_integration.best_run import BPBestRun
from exp02_check_grid_search import (PROBLEMS,
                                     _get_statistics_from_deepobs_check,
                                     analyze, check_each_run_has_num_seeds,
                                     check_manually, create_grid_search,
                                     filter_problem, print_dashed_line)
from utils import BEST_STRATEGY


def check_best_run_manually():
    """Print to console."""
    for path in get_results_paths():
        check_manually(path)


def check_best_run_each_has_10_seeds():
    results = []
    for path in get_results_paths():
        results.append(check_each_run_has_num_seeds(10, path))
    return results


def check_best_run_complete(filter_func=None):
    print("\nCHECK BEST RUN COMPLETE\n")

    results = []
    for path in get_results_paths():
        statistics = _get_statistics_from_deepobs_check(path)

        # key-value pairs (problem, optim_cls) : num_settings
        optims_problems_settings = {(item[0], item[1]): item[2]
                                    for item in statistics}

        passing = True
        for experiment in create_grid_search(filter_func):
            optim_cls = experiment._get_optim_name()
            problem = experiment.get_deepobs_problem()
            key = (problem, optim_cls)

            expected_dim = 1
            try:
                dim = optims_problems_settings[key]

            except KeyError:
                print("{:55} Expect single run, but no data found".format(
                    str(key)))
                passing = False
                continue

            if dim != expected_dim:
                passing = False
                print("{:55} Expect single run, but found {}".format(
                    str(key), dim))
            else:
                print("{} passed".format(key))
        results.append(passing)
    return results


def get_results_paths():
    paths = []
    first_grid_search = create_grid_search()[0]
    for mode, metric in BEST_STRATEGY:
        first_best_run = BPBestRun(first_grid_search, mode, metric)
        paths.append(first_best_run.get_output_dir())
    return paths


def summarize_best_runs_progress():
    headers = ["Problem", "Found all optims"]
    result = []
    for problem in PROBLEMS:
        filter_func = filter_problem(problem)
        single_run = check_best_run_complete(filter_func)
        result.append([problem, single_run])
    result = pandas.DataFrame(result, columns=headers)
    return result


if __name__ == "__main__":
    check_best_run_manually()
    print_dashed_line()
    have_10_seeds = check_best_run_each_has_10_seeds()
    print_dashed_line()
    summary = summarize_best_runs_progress()

    print_dashed_line()
    print("BEST RUN TEST SUMMARY\n")
    print(summary.to_string(index=False))
    print("\nFound runs evaluated on 10 different seeds: {}".format(
        have_10_seeds))
    print_dashed_line()

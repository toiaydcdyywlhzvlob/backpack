import argparse
import sys

from bp_dops_integration.best_run import BPBestRun
from exp01_grid_search import create_grid_search

parser = argparse.ArgumentParser(
    description="Arguments for re-running best run")

parser.add_argument("--damping", type=str, help="Damping strategy label")
parser.add_argument("--problem", type=str, help="Problem label")
parser.add_argument("--curvature", type=str, help="Curvature label")
parser.add_argument("--best_mode", type=str, help="Mode to determine best")
parser.add_argument("--best_metric", type=str, help="Metric to determine best")
parser.add_argument("--extended_logs",
                    dest="extended_logs",
                    action="store_true",
                    help="Log extended metrics")
parser.set_defaults(extended_logs=False)
parser.add_argument("--output_dir", type=str, help="Output directory")
parser.add_argument("--random_seed", type=int, help="Random seed")

ARGS, LEFT = parser.parse_known_args()
sys.argv = [sys.argv[0], *LEFT]


def rerun_best_for_seed(damping, problem, curvature, mode, metric, output_dir,
                        seed, extended_logs):
    def filter_config(curv, damp, prob):
        return (curv == curvature) and (damp == damping) and (prob == problem)

    search = create_grid_search(filter_func=filter_config)[0]
    best_run = BPBestRun(search, mode, metric, output_dir=output_dir)
    best_run.rerun_best_for_seeds([seed], extended_logs=extended_logs)


rerun_best_for_seed(ARGS.damping, ARGS.problem, ARGS.curvature, ARGS.best_mode,
                    ARGS.best_metric, ARGS.output_dir, ARGS.random_seed,
                    ARGS.extended_logs)

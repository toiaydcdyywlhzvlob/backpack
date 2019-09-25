import argparse
import sys

from utils import PROBLEMS

ALL = "all"
CHOICES = PROBLEMS + [ALL]

parser = argparse.ArgumentParser(description="Choose the DeepOBS problem")

parser.add_argument("--dobs_problem",
                    type=str,
                    default="all",
                    choices=CHOICES,
                    help=f"DeepOBS problem")


def make_filter_func():
    ARGS, LEFT = parser.parse_known_args()
    sys.argv = [sys.argv[0], *LEFT]

    choice = ARGS.dobs_problem

    def filter_func(curv, damp, prob):
        # all problems
        if choice == ALL:
            return True

        # specific problem
        else:
            if choice == prob:
                return True
            else:
                return False

    print(f"Choosen problem(s): {choice}")
    return filter_func

"""Create the runscripts for repeating the best runs."""
import os

from utils import BEST_STRATEGY, SEEDS, allowed_combinations

PYTHON = "python3"
EXEC_SCRIPT = "../exp/repeat_best.py"

# best runs
BEST_RUN_SCRIPT_DIR = "../best_run_command_scripts"
BEST_RUN_OUTPUT_DIR = "../best_run"
BEST_RUN_SEEDS = SEEDS


def write_scripts(output_dir,
                  runscript_dir,
                  seeds,
                  extended_logs,
                  filter_func=None):
    for (curv, damp, prob) in allowed_combinations(filter_func=filter_func):
        for mode, metric in BEST_STRATEGY:
            _write_script(damp, prob, curv, mode, metric, seeds, output_dir,
                          runscript_dir, extended_logs)


def _write_script(damping, problem, curvature, mode, metric, seeds, output_dir,
                  runscript_dir, extended_logs):
    script = _script_filename(damping, problem, curvature, mode, metric)
    script_path = os.path.join(runscript_dir, script)

    print("[BEST-RUN] Create runscript in {}".format(script_path))

    with open(script_path, "w") as f:
        for seed in seeds:
            cmd = _command(damping, problem, curvature, mode, metric,
                           output_dir, seed, extended_logs)
            f.write(cmd)


def _script_filename(damping, problem, curvature, mode, metric):
    return f"jobs_{problem}_{curvature}_{damping}_{mode}_{metric}.txt"


def _command(damping, problem, curvature, mode, metric, output_dir, seed,
             extended_logs):
    problem_arg = f" --problem {problem}"
    curvature_arg = f" --curvature {curvature}"
    damping_arg = f" --damping {damping}"
    best_mode_arg = f" --best_mode {mode}"
    best_metric_arg = f" --best_metric {metric}"
    extended_logs_arg = " --extended_logs" if extended_logs else ""
    output_dir_arg = f" --output_dir {output_dir}"
    random_seed_arg = f" --random_seed {seed}"

    return f"{PYTHON} {EXEC_SCRIPT}{problem_arg}{curvature_arg}{damping_arg}{best_mode_arg}{best_metric_arg}{extended_logs_arg}{output_dir_arg}{random_seed_arg}\n"


if __name__ == "__main__":
    from control import make_filter_func

    filter_func = make_filter_func()

    os.makedirs(BEST_RUN_SCRIPT_DIR, exist_ok=True)
    write_scripts(output_dir=BEST_RUN_OUTPUT_DIR,
                  runscript_dir=BEST_RUN_SCRIPT_DIR,
                  seeds=BEST_RUN_SEEDS,
                  extended_logs=False,
                  filter_func=filter_func)

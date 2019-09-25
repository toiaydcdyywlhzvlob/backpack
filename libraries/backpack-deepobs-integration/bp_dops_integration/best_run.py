import os

import numpy as np

from deepobs.tuner.tuner_utils import (_determine_available_metric,
                                       create_setting_analyzer_ranking)

from .runners import BPOptimRunner, BPOptimRunnerExtendedLogging


def my_rerun_setting(runner,
                     optimizer_class,
                     hyperparam_names,
                     optimizer_path,
                     output_dir,
                     seeds=np.arange(42, 52),
                     rank=1,
                     mode='final',
                     metric='valid_accuracies'):
    """Modification from DeepOBS."""
    metric = _determine_available_metric(optimizer_path, metric)
    optimizer_path = os.path.join(optimizer_path)

    setting_analyzer_ranking = create_setting_analyzer_ranking(
        optimizer_path, mode, metric)
    setting = setting_analyzer_ranking[rank - 1]

    runner = runner(optimizer_class, hyperparam_names)

    hyperparams = setting.aggregate['optimizer_hyperparams']
    training_params = setting.aggregate['training_params']
    testproblem = setting.aggregate['testproblem']
    num_epochs = setting.aggregate['num_epochs']
    batch_size = setting.aggregate['batch_size']

    results_path = output_dir
    for seed in seeds:
        runner.run(testproblem,
                   hyperparams=hyperparams,
                   random_seed=int(seed),
                   num_epochs=num_epochs,
                   batch_size=batch_size,
                   output_dir=results_path,
                   **training_params)


class BestRunBase():
    """Rerun the best run from the grid search for multiple seeds."""
    def __init__(self, grid_search, mode, metric, output_dir="../best_run"):
        self.grid_search = grid_search
        self.mode = mode
        self.metric = metric
        self.output_dir = os.path.join(output_dir,
                                       "{}_{}".format(self.mode, self.metric))

    def get_mode(self):
        return self.mode

    def get_metric(self):
        return self.metric

    def get_runner_cls(self, extended_logs=False):
        """Use BPOptimRunnerExtendedLogging if enabled."""
        runner_cls = self.grid_search.get_runner_cls()
        if extended_logs:
            if runner_cls == BPOptimRunner:
                runner_cls = BPOptimRunnerExtendedLogging
            else:
                raise ValueError(
                    "Extended logs not supported for runner class {}".format(
                        runner_cls))
        return runner_cls

    def rerun_best(self, extended_logs=False):
        my_rerun_setting(self.get_runner_cls(extended_logs=extended_logs),
                         self.grid_search.get_optim_cls(),
                         self.grid_search.get_hyperparams(),
                         self.grid_search.get_path(),
                         self.output_dir,
                         mode=self.mode,
                         metric=self.metric)

    def rerun_best_for_seeds(self, seeds, extended_logs=False):
        my_rerun_setting(self.get_runner_cls(extended_logs=extended_logs),
                         self.grid_search.get_optim_cls(),
                         self.grid_search.get_hyperparams(),
                         self.grid_search.get_path(),
                         self.output_dir,
                         seeds=seeds,
                         mode=self.mode,
                         metric=self.metric)

    def get_output_dir(self):
        return self.output_dir

    def get_path(self):
        return os.path.join(self.output_dir,
                            self.grid_search.get_path_appended_by_deepobs())

    def get_problem_path(self):
        return os.path.join(
            self.grid_search.get_generation_dir(), self.output_dir,
            self._get_dirname(),
            self.grid_search.get_problem_path_appended_by_deepobs())

    def _get_dirname(self):
        return "{}_{}".format(self.mode, self.metric)


class BPBestRun(BestRunBase):
    pass

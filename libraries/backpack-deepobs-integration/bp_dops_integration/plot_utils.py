"""Plotting of extended logging quantities."""
import os
import matplotlib.pyplot as plt

from deepobs.analyzer.analyze import (
    _get_optimizer_name_and_testproblem_from_path,
    _plot_hyperparameter_sensitivity, _preprocess_path,
    create_setting_analyzer_ranking, plot_hyperparameter_sensitivity_2d,
    plot_optimizer_performance)


def plot_optimizer_extended_logs(path,
                                 ax=None,
                                 mode='most',
                                 metric='valid_accuracies',
                                 reference_path=None,
                                 which='mean_and_std',
                                 custom_metrics=None):
    """Analog to `plot_optimizer_performance` for extended logging metrics."""
    raise NotImplementedError("Custom metrics not supported by DeepOBS yet")
    if custom_metrics is None:
        custom_metrics = []

    num_dobs_plots = 4
    num_plots = num_dobs_plots + len(custom_metrics)

    if ax is None:
        _, ax = plt.subplots(num_plots, 1, sharex="col")

    # DeepOBS plots
    ax = plot_optimizer_performance(path,
                                    ax=ax,
                                    mode=mode,
                                    metric=metric,
                                    reference_path=reference_path,
                                    which=which)

    # Custom metrics plots
    ax = _plot_optimizer_extended_logs(path,
                                       ax,
                                       mode=mode,
                                       metric=metric,
                                       which=which,
                                       custom_metrics=custom_metrics)

    for idx, custom_metric in enumerate(custom_metrics, num_dobs_plots):
        # set y labels
        ax[idx].set_ylabel(custom_metric, fontsize=14)
        ax[idx].tick_params(labelsize=12)
        # show optimizer legends
        ax[idx].legend(fontsize=12)

    ax[-1].set_xlabel('epochs', fontsize=14)

    return ax


def _plot_optimizer_extended_logs(path,
                                  ax,
                                  mode='most',
                                  metric='valid_accuracies',
                                  which='mean_and_std',
                                  custom_metrics=None):
    raise NotImplementedError("Custom metrics not supported by DeepOBS yet")

    all_possible_custom_metrics = [
        "batch_loss_before_step",
        "batch_loss_after_step",
        "l2_reg_before_step",
        "l2_reg_after_step",
        "batch_loss_grad_norm_before_step",
        "batch_loss_grad_norm_after_step",
        "damping",
        "trust_damping",
        "inv_damping",
        "parameter_change_norm",
        "batch_loss_improvement",
    ]

    if custom_metrics is None:
        custom_metrics = []

    num_dobs_plots = 4

    pathes = _preprocess_path(path)

    for optimizer_path in pathes:
        setting_analyzer_ranking = create_setting_analyzer_ranking(
            optimizer_path,
            mode,
            metric,
            custom_metrics=all_possible_custom_metrics)
        setting = setting_analyzer_ranking[0]

        def items_in_aggregate_with_key_containing(string):
            return [(key, value) for (key, value) in setting.aggregate.items()
                    if string in key]

        optimizer_name = os.path.basename(optimizer_path)

        for idx, metric in enumerate(custom_metrics, num_dobs_plots):
            if idx == num_dobs_plots:
                _, testproblem = _get_optimizer_name_and_testproblem_from_path(
                    optimizer_path)
                ax[idx].set_title(testproblem, fontsize=18)

            # reuse same color/style if multiple lines
            color, linestyle = None, None

            for metric_name, metric_data in items_in_aggregate_with_key_containing(
                    metric):

                if which == 'mean_and_std':
                    center = metric_data['mean']
                    std = metric_data['std']
                    low, high = center - std, center + std
                elif which == 'median_and_quartiles':
                    center = metric_data['median']
                    low = metric_data['lower_quartile']
                    high = metric_data['upper_quartile']
                else:
                    raise ValueError("Unknown value which={}".format(which))

                # label = "{}, {}".format(optimizer_name,
                #                         metric_name).replace("_", "\_")
                label = metric_name.replace("_", "\_")

                line, = ax[idx].plot(center,
                                     label=label,
                                     color=color,
                                     linestyle=linestyle)
                if color is None and linestyle is None:
                    color = line.get_color()
                    linestyle = line.get_linestyle()
                ax[idx].fill_between(range(len(center)),
                                     low,
                                     high,
                                     facecolor=color,
                                     alpha=0.3)

    return ax

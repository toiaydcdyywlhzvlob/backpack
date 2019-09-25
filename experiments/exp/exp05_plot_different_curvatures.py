"""Plot best runs for same damping with different curvature and baselines."""

from bp_dops_integration.plotting import (BPPerformanceVaryCurvaturePlot,
                                          TikzExport)
from utils import (allowed_best_runs, allowed_curvatures, baseline_dir,
                   plot_filename)


def plot_different_curvatures(damping,
                              problem,
                              best_mode,
                              best_metric,
                              use_custom_style=False,
                              reference_path=None,
                              run_dir="../best_run",
                              custom_metrics=None,
                              filter_func=None):
    curvatures = allowed_curvatures(damping, problem, filter_func=filter_func)
    plotter = BPPerformanceVaryCurvaturePlot(curvatures,
                                             damping,
                                             problem,
                                             best_mode,
                                             best_metric,
                                             reference_path=reference_path,
                                             run_dir=run_dir)
    return plotter.add_plots(use_custom_style=use_custom_style,
                             which="median_and_quartiles",
                             custom_metrics=custom_metrics)


if __name__ == "__main__":
    from control import make_filter_func

    tikz = TikzExport()

    filter_func = make_filter_func()

    damp_prob_mode_metric = set([
        (damp, prob, mode, metric)
        for (_, damp, prob, mode,
             metric) in allowed_best_runs(filter_func=filter_func)
    ])

    append_dir = "performance/"
    run_dir = "../best_run"

    # remove damping scheme from legendentry
    rm_damping_label = True

    for (damp, prob, mode, metric) in damp_prob_mode_metric:
        baseline = baseline_dir(prob)
        fig, ax = plot_different_curvatures(damp,
                                            prob,
                                            mode,
                                            metric,
                                            use_custom_style=True,
                                            reference_path=baseline,
                                            run_dir=run_dir,
                                            filter_func=filter_func)
        filename = plot_filename("curvatures",
                                 damp,
                                 prob,
                                 mode,
                                 metric,
                                 append_to_rel_path=append_dir)

        tikz.save_fig(filename,
                      fig=fig,
                      post_process=True,
                      rm_damping_label=rm_damping_label)
        tikz.save_subplots(filename,
                           fig=fig,
                           post_process=True,
                           rm_damping_label=rm_damping_label)

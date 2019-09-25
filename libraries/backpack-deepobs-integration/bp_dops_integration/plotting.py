import os
import traceback
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
from deepobs.analyzer.analyze import (_plot_hyperparameter_sensitivity,
                                      _preprocess_path,
                                      create_setting_analyzer_ranking,
                                      plot_hyperparameter_sensitivity_2d,
                                      plot_optimizer_performance)
from palettable.colorbrewer.sequential import (Blues_5, Greens_5, Greys_4,
                                               Greys_5, Oranges_5, Oranges_6,
                                               Purples_4, Purples_5, Reds_5,
                                               YlGn_5)
from tikzplotlib import save as tikz_save

from .best_run import BPBestRun
from .experiments import GridSearchFactory
from .plot_utils import plot_optimizer_extended_logs


class TikzExport():
    """Handle matplotlib export to TikZ."""
    def __init__(self, extra_axis_parameters={"zmystyle"}):
        """
        Note:
        -----
        Extra axis parameters are inserted in alphabetical order.
        By prepending 'z' to the style, it will be inserted last.
        Like that, you can overwrite the previous axis parameters
        in 'zmystyle'.
        """
        self.extra_axis_parameters = extra_axis_parameters

    def save_fig(self,
                 out_file,
                 fig=None,
                 png_preview=True,
                 override_externals=True,
                 post_process=True,
                 rm_damping_label=False):
        """Save matplotlib figure as TikZ. Optional PNG out.

        Create the directory if it does not exist.
        """
        if fig is not None:
            self.set_current(fig)

        tex_file, png_file = [
            self._add_extension(out_file, extension)
            for extension in ['tex', 'png']
        ]

        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        if png_preview:
            plt.savefig(png_file, bbox_inches='tight')

        self._fix_underscore_latex_formatting_in_title()

        tikz_save(tex_file,
                  override_externals=override_externals,
                  extra_axis_parameters=self.extra_axis_parameters)

        if post_process is True:
            self.post_process(out_file, rm_damping_label=rm_damping_label)

    def save_subplots(self,
                      out_path,
                      fig=None,
                      png_preview=True,
                      override_externals=True,
                      post_process=True,
                      rm_damping_label=False):
        """Save subplots of figure into single TikZ figures.

        Use ylabel in output filename if existent.
        """
        if fig is None:
            fig = plt.gcf()

        for idx, subplot_fig in enumerate(self.axes_as_individual_figs(fig)):
            assert len(subplot_fig.get_axes()) == 1
            filename = "{}".format(idx)

            ylabel = subplot_fig.get_axes()[0].get_ylabel()
            if ylabel != "":
                filename = "{}_{}".format(filename, ylabel)

            out_file = os.path.join(out_path, filename)
            self.save_fig(out_file,
                          fig=subplot_fig,
                          png_preview=png_preview,
                          override_externals=override_externals,
                          post_process=post_process,
                          rm_damping_label=rm_damping_label)

    @staticmethod
    def set_current(fig):
        plt.figure(fig.number)

    @staticmethod
    def _fix_underscore_latex_formatting_in_title():
        def __fix_underscore(title):
            latex, verbose = r"\_", "_"
            # make all correct underscores verbose
            title = title.replace(latex, verbose)
            # latex formatting
            title = title.replace(verbose, latex)
            return title

        fig = plt.gcf()
        for ax in fig.get_axes():
            ax_title = ax.get_title()
            fixed_title = __fix_underscore(ax_title)
            ax.set_title(fixed_title)

    def post_process(self, tikz_file, rm_damping_label=False):
        """Remove from matplotlib2tikz export what should be configurable.

        Write processed file to `tikz_file + '_processed.tex'`.
        """
        unprocessed_file = self._add_extension(tikz_file, 'tex')
        with open(unprocessed_file, "r") as f:
            content = f.readlines()

        content = self._remove_linewidths(content)
        content = self._remove_some_arguments(content)
        content = self._rename_legend_entries(
            content, rm_damping_label=rm_damping_label)

        joined_content = "".join(content)

        out_file = self._add_extension(tikz_file,
                                       "tex",
                                       add_to_filename="_processed")
        with open(out_file, "w") as f:
            f.write(joined_content)

    @staticmethod
    def _add_extension(filename, extension, add_to_filename=None):
        if add_to_filename is None:
            add_to_filename = ''
        return "{}{}.{}".format(filename, add_to_filename, extension)

    @staticmethod
    def _remove_linewidths(lines):
        """Remove line width specifications."""
        linewidths = [
            r'ultra thick',
            r'very thick',
            r'semithick',
            r'thick',
            r'very thin',
            r'ultra thin',
            r'thin',
        ]
        new_lines = []
        for line in lines:
            for width in linewidths:
                line = line.replace(width, '')
            new_lines.append(line)
        return new_lines

    @staticmethod
    def _remove_some_arguments(lines):
        """Remove lines containing certain specifications."""
        # remove lines containing these specifications
        to_remove = [
            r"legend cell align",
            # r"legend style",
            r"x grid style",
            r"y grid style",
            # r"tick align",
            # r"tick pos",
            r"ytick",
            r"xtick",
            r"yticklabels",
            r"xticklabels",
            r"ymode",
            r"log basis y",
        ]

        for pattern in to_remove:
            lines = [line for line in lines if pattern not in line]

        return lines

    def _rename_legend_entries(self, lines, rm_damping_label=False):
        """Proper formatting of optimizer names."""
        rename_dict = self._get_rename_dict_legend_entries(
            rm_damping_label=rm_damping_label)

        new_lines = []

        for line in lines:
            for old, new in rename_dict.items():
                line = line.replace(old, new)
            new_lines.append(line)

        return new_lines

    @staticmethod
    def _get_rename_dict_legend_entries(rm_damping_label=False):
        def legendentry(string):
            return r"\addlegendentry{" + string + r"}"

        factory = GridSearchFactory()
        optim_classes = factory.get_all_optim_classes()

        rename = {}

        for optim_cls in optim_classes:
            (curv, damp) = factory.get_curvature_and_damping(optim_cls)
            curv = curv.replace("_", "-")
            old_entry = legendentry(optim_cls.__name__)
            new_entry = legendentry(curv)
            if not rm_damping_label:
                new_entry += r"}, " + damp

            rename[old_entry] = new_entry

        return rename

    @staticmethod
    def axes_as_individual_figs(fig):
        """Return a list of figures, each containing a single axes.

        `fig` is messed up during this procedure as the axes are being removed
        and inserted into other figures.

        Note: MIGHT BE UNSTABLE
        -----
        https://stackoverflow.com/questions/6309472/matplotlib-can-i-create-axessubplot-objects-then-add-them-to-a-figure-instance

        Axes deliberately aren't supposed to be shared between different figures now.
        As a workaround, you could do this fig2._axstack.add(fig2._make_key(ax), ax),
        but it's hackish and likely to change in the future.
        It seems to work properly, but it may break some things.
        """
        fig_axes = fig.get_axes()

        # breaks fig
        for ax in fig_axes:
            fig.delaxes(ax)

        fig_list = []
        for ax in fig_axes:
            new_fig = plt.figure()
            new_fig._axstack.add(new_fig._make_key(ax), ax)
            new_fig.axes[0].change_geometry(1, 1, 1)
            fig_list.append(new_fig)

        return fig_list


class BPPlotBase():
    """Basic plotting functionality."""
    def __init__(self, paths, reference_path=None):
        self.paths = paths
        self.reference_path = reference_path

    def add_plots(
            self,
            fig=None,
            ax=None,
            mode="most",
            metric="valid_accuracies",
            # which="median_and_quartiles",
            which="mean_and_std",
            use_custom_style=True,
            custom_metrics=None):
        """Plot performance metrics."""
        if custom_metrics is None:
            custom_metrics = []

        if use_custom_style is True:
            custom_style_setter = self.get_custom_style()
        else:
            custom_style_setter = self.do_nothing()

        fig, ax = None, None
        with custom_style_setter() as _:
            for path_idx, path in enumerate(self.paths, 1):
                last_path = (path_idx == len(self.paths))
                reference_path = None if not last_path else self.reference_path
                fig, ax = plot_optimizer_performance(
                    path,
                    fig=fig,
                    ax=ax,
                    mode=mode,
                    metric=metric,
                    reference_path=reference_path,
                    which=which)
        return fig, ax

    def get_custom_style(self):
        """Return a callable context that modifies the linestyles."""
        raise NotImplementedError("No custom style context defined")

    @staticmethod
    def do_nothing():
        @contextmanager
        def nothing():
            try:
                yield None
            finally:
                pass

        return nothing


class BPDirectories():
    """Get paths of best run and grid search for experiments."""
    def __init__(self, run_dir="../best_run"):
        self.run_dir = run_dir

    def _get_path_to_grid_search(self, curvature, damping, problem):
        experiment = self._get_grid_search(curvature, damping, problem)
        return experiment.get_path()

    def _get_path_to_best_run(self, curvature, damping, problem, best_mode,
                              best_metric):
        experiment = self._get_grid_search(curvature, damping, problem)
        best_run = BPBestRun(experiment,
                             best_mode,
                             best_metric,
                             output_dir=self.run_dir)
        return best_run.get_path()

    @staticmethod
    def _get_grid_search(curvature, damping, problem):
        factory = GridSearchFactory()
        return factory.make_grid_search(curvature, damping, problem)


class BPCustomLinestyles():
    """Custom Line styles for different curves.

    * Vary dash for damping scheme
    * Vary color for curvature
    * Base lines: Dotted lines in black / grey
    """
    BASELINE_COLORS = Greys_5.mpl_colors[::-1]
    BASELINE_DASHES = ["-", "-"]

    DASHES = {
        GridSearchFactory.CONSTANT: "--",
        GridSearchFactory.LM: "-",
        GridSearchFactory.FANCY: "-.",
    }

    COLOR_LIST_IDX = {
        GridSearchFactory.CONSTANT: 3,
        GridSearchFactory.LM: 3,
        GridSearchFactory.FANCY: 3,
    }

    COLOR_LISTS = {
        GridSearchFactory.Zero: Greys_5.mpl_colors,
        GridSearchFactory.DiagGGNExact: Reds_5.mpl_colors,
        GridSearchFactory.DiagGGNMC: Oranges_6.mpl_colors,
        GridSearchFactory.KFAC: Blues_5.mpl_colors,
        GridSearchFactory.KFLR: Greens_5.mpl_colors,
        GridSearchFactory.KFRA: Purples_4.mpl_colors,
    }

    def __init__(self, curv_damp_list, num_baselines=2):
        """
        `curv_damp_list:` A list of tuples containing the curvature and the
        damping for the plots in the correct order.
        """
        self.num_baselines = num_baselines
        self.__old_prop_cycle = None
        self.color_cycler = self.make_color_cycler(curv_damp_list)
        self.dash_cycler = self.make_dash_cycler(curv_damp_list)

    def make_color_cycler(self, curv_damp_list):
        colors = []
        for curv, damp in curv_damp_list:
            color_list = self.COLOR_LISTS[curv]
            color_idx = self.COLOR_LIST_IDX[damp]
            color = color_list[color_idx]
            colors.append(color)

        for idx in range(self.num_baselines):
            colors.append(self.BASELINE_COLORS[idx])
        return matplotlib.cycler(color=colors)

    def make_dash_cycler(self, curv_damp_list):
        dampings = [damp for (_, damp) in curv_damp_list]
        dashes = [self.DASHES[damp] for damp in dampings]
        for idx in range(self.num_baselines):
            dashes.append(self.BASELINE_DASHES[idx])
        return matplotlib.cycler(linestyle=dashes)

    @staticmethod
    def set_prop_cycle(cycle):
        matplotlib.rcParams['axes.prop_cycle'] = cycle

    @staticmethod
    def get_prop_cycle():
        return matplotlib.rcParams['axes.prop_cycle']

    def __enter__(self):
        self.__old_prop_cycle = self.get_prop_cycle()
        custom_cycle = self.color_cycler + self.dash_cycler
        self.set_prop_cycle(custom_cycle)
        return None

    def __exit__(self, exc_type, exc_value, tb):
        self.set_prop_cycle(self.__old_prop_cycle)
        self.__old_prop_cycle = None

        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)


##############################################################################
# Optimizer performance plots                                                #
##############################################################################


class BPPerformanceVaryCurvaturePlot(BPPlotBase, BPDirectories):
    """Plot performance for different curvatures."""
    def __init__(self,
                 curvatures,
                 damping,
                 problem,
                 best_mode,
                 best_metric,
                 reference_path=None,
                 run_dir="../best_run"):
        BPDirectories.__init__(self, run_dir=run_dir)
        self.curvatures = curvatures
        self.damping = damping
        paths_to_best_runs = self.get_paths_to_runs(problem, best_mode,
                                                    best_metric)
        BPPlotBase.__init__(self,
                            paths_to_best_runs,
                            reference_path=reference_path)

    def get_paths_to_runs(self, problem, best_mode, best_metric):
        return [
            self._get_path_to_best_run(curv, damp, problem, best_mode,
                                       best_metric)
            for (curv, damp) in self.get_curv_damp_pairs()
        ]

    def get_curv_damp_pairs(self):
        return [(curv, self.damping) for curv in self.curvatures]

    def get_custom_style(self):
        curv_damp_list = self.get_curv_damp_pairs()

        def create_custom_style():
            return BPCustomLinestyles(curv_damp_list)

        return create_custom_style


class BPPerformanceVaryDampingPlot(BPPlotBase, BPDirectories):
    """Plot performance for different dampings."""
    def __init__(self,
                 curvature,
                 dampings,
                 problem,
                 best_mode,
                 best_metric,
                 reference_path=None,
                 run_dir="../best_run"):
        BPDirectories.__init__(self, run_dir=run_dir)
        self.curvature = curvature
        self.dampings = dampings
        paths_to_best_runs = self.get_paths_to_runs(problem, best_mode,
                                                    best_metric)
        BPPlotBase.__init__(self,
                            paths_to_best_runs,
                            reference_path=reference_path)

    def get_paths_to_runs(self, problem, best_mode, best_metric):
        return [
            self._get_path_to_best_run(curv, damp, problem, best_mode,
                                       best_metric)
            for (curv, damp) in self.get_curv_damp_pairs()
        ]

    def get_curv_damp_pairs(self):
        return [(self.curvature, damp) for damp in self.dampings]

    def get_custom_style(self):
        curv_damp_list = self.get_curv_damp_pairs()

        def create_custom_style():
            return BPCustomLinestyles(curv_damp_list)

        return create_custom_style


##############################################################################
# Hyperparameter sensitivity plots                                           #
##############################################################################
class SensitivityPlot(BPPlotBase):
    def _add_hyperparameter_sensitivity_plots_1d(self,
                                                 hyperparam,
                                                 ax=None,
                                                 mode="final",
                                                 metric="valid_accuracies",
                                                 plot_std=False,
                                                 xscale="log"):
        raise NotImplementedError("Needs fix from DeepOBS merge")
        if ax is None:
            ax = self.default_ax()
        for path in self.paths:
            ax = _plot_hyperparameter_sensitivity(path,
                                                  hyperparam,
                                                  ax,
                                                  mode=mode,
                                                  metric=metric,
                                                  plot_std=plot_std)
            ax.set_xscale(xscale)
        return ax

    def _add_hyperparameter_sensitivity_plots_2d(self,
                                                 hyperparams,
                                                 ax=None,
                                                 mode="final",
                                                 metric="valid_accuracies",
                                                 xscale="log",
                                                 yscale="log"):
        if ax is None:
            ax = self.default_ax()
        for path in self.paths:
            ax = plot_hyperparameter_sensitivity_2d(path,
                                                    hyperparams,
                                                    mode=mode,
                                                    metric=metric,
                                                    xscale=xscale,
                                                    yscale=yscale)
        return ax

    @staticmethod
    def default_ax():
        _, ax = plt.subplots()
        return ax


class BPSensitivityVaryCurvaturePlot(SensitivityPlot, BPDirectories):
    """Plot hyperparameter sensitivity for different curvatures."""
    def __init__(self,
                 curvatures,
                 damping,
                 problem,
                 best_mode,
                 best_metric,
                 reference_path=None):
        raise NotImplementedError("Needs fix from DeepOBS merge")
        self.best_mode = best_mode
        self.best_metric = best_metric
        self.hyperparams = self._get_hyperparams(curvatures, damping, problem)
        paths_to_grid_search = self.get_paths_to_grid_search(
            curvatures, damping, problem)
        super().__init__(paths_to_grid_search, reference_path=reference_path)

    def _get_hyperparams(self, curvatures, damping, problem):
        grid_search = self._get_grid_search(curvatures[0], damping, problem)
        hyperparams = list(grid_search.get_hyperparams().keys())
        return hyperparams

    def get_paths_to_grid_search(self, curvatures, damping, problem):
        return [
            self._get_path_to_grid_search(curvature, damping, problem)
            for curvature in curvatures
        ]

    def add_hyperparameter_sensitivity_plots(self, ax=None):
        num_hyperparams = len(self.hyperparams)
        if num_hyperparams == 1:
            return self._add_hyperparameter_sensitivity_plots_1d(ax=ax)
        elif num_hyperparams == 2:
            return self._add_hyperparameter_sensitivity_plots_2d(ax=ax)
        else:
            raise ValueError(
                "Number of hyperparameters exceeds 2, got {}".format(
                    self.hyperparams))

    def _add_hyperparameter_sensitivity_plots_1d(self,
                                                 hyperparam=None,
                                                 ax=None,
                                                 mode=None,
                                                 metric=None,
                                                 plot_std=False,
                                                 xscale="log"):
        if hyperparam is None:
            hyperparam = self.hyperparams[0]
        if mode is None:
            mode = self.best_mode
        if metric is None:
            metric = self.best_metric

        return super()._add_hyperparameter_sensitivity_plots_1d(
            self.hyperparams[0],
            ax=ax,
            mode=self.best_mode,
            metric=self.best_metric,
            plot_std=plot_std,
            xscale=xscale)

    def _add_hyperparameter_sensitivity_plots_2d(self,
                                                 hyperparams=None,
                                                 ax=None,
                                                 mode=None,
                                                 metric=None,
                                                 xscale="log",
                                                 yscale="log"):
        if hyperparams is None:
            hyperparams = self.hyperparams
        if mode is None:
            mode = self.best_mode
        if metric is None:
            metric = self.best_metric

        return super()._add_hyperparameter_sensitivity_plots_2d(
            hyperparams,
            ax=ax,
            mode=self.best_mode,
            metric=self.best_metric,
            xscale=xscale,
            yscale=yscale)

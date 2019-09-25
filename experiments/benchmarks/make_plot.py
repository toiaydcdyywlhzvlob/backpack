import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
Config and parameters
"""

font = {'family': 'serif', 'size': 12}
matplotlib.rc('font', **font)


#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}\usepackage{times}"]


def rgb_to_u(xs):
    return list([x / 255.0 for x in xs])

barcolors = [
    rgb_to_u([54, 75, 154]),
    rgb_to_u([110, 166, 205]),
    rgb_to_u([194, 228, 239]),
]


colors = [
    rgb_to_u([221, 170, 51]),
    rgb_to_u([187, 85, 102]),
    rgb_to_u([0, 68, 136]),
    rgb_to_u([0, 0, 0]),
]

label_to_linestyle = {
    "BackPACK": "--x",
    "For-loop": "-",
    "Gradient (for ref.)": "-.",
}
label_to_linewidth = {
    "BackPACK": 5,
    "For-loop": 5,
    "Gradient (for ref.)": 3,
}

"""
Helpers
"""


def median(bench):
    Ns = sorted(list(bench.keys()))
    quart1 = list([np.percentile(bench[n], .25) for n in Ns])
    med = list([np.percentile(bench[n], .5) for n in Ns])
    quart3 = list([np.percentile(bench[n], .75) for n in Ns])
    return quart1, med, quart3


def quartiles(bench):
    return np.array([np.percentile(bench, .25), np.percentile(bench, .75)])


def hide_frame(axes, top=True, right=True, left=False, bottom=False):
    for ax in axes:
        ax.spines['top'].set_visible(not top)
        ax.spines['right'].set_visible(not right)
        ax.spines['left'].set_visible(not left)
        ax.spines['bottom'].set_visible(not bottom)


"""
Plotting functions
"""


def method_name_to_label(mname):
    if mname == "Var":
        return "Variance"
    if mname == "DiagGGNMC":
        return "DiagGGN-MC"
    if mname == "DiagGGNExact":
        return "DiagGGN"
    if mname == "BatchL2":
        return "Batch L2"
    if mname == "diagH":
        return "Diag Hessian"
    if mname == "diagGGN":
        return "Diag GGN"
    if mname == "BatchGrad":
        return "Indiv. Grad"
    if mname == "SecondMoment":
        return "2nd Moment"
    if mname == "_grad":
        return "Grad (ref.)"
    return mname


to_ms = 1000


def make_cifar10_plot():
    fig = plt.figure(figsize=(12, 3.25))

    grid = plt.GridSpec(
        1, 1,
        wspace=0.15, hspace=0.1,
        left=0.075,
        right=0.975,
        top=0.9,
        bottom=0.35,
    )
    ax = fig.add_subplot(grid[0])

    with open("benchmark_cifar10.pk", 'rb') as handle:
        results = pickle.load(handle)

        methods_to_idx = {
            "_grad": 0,
            "SecondMoment": 1,
            "Var": 2,
            "BatchL2": 3,
            "BatchGrad": 4,
            "DiagGGNMC": 5,
            "KFAC": 6,
            "KFLR": 7,
            "DiagGGNExact": 8,
        }

        small_batch = np.zeros(len(methods_to_idx))
        med_batch = np.zeros(len(methods_to_idx))
        large_batch = np.zeros(len(methods_to_idx))
        huge_batch = np.zeros(len(methods_to_idx))
        small_batch_quant = np.zeros((len(methods_to_idx), 2))
        med_batch_quant = np.zeros((len(methods_to_idx), 2))
        large_batch_quant = np.zeros((len(methods_to_idx), 2))
        huge_batch_quant = np.zeros((len(methods_to_idx), 2))
        n_small = 32
        n_med = 64
        n_large = 96
        n_huge = 128
        for name, bench in results.items():
            print(name, bench.keys())
            for method_name, method_id in methods_to_idx.items():
                if method_name in name:
                    small_batch[method_id] = np.median(bench[n_small]) * to_ms
                    med_batch[method_id] = np.median(bench[n_med]) * to_ms
                    large_batch[method_id] = np.median(bench[n_large]) * to_ms
                    huge_batch[method_id] = np.median(bench[n_huge]) * to_ms

                    small_batch_quant[method_id, :] = quartiles(bench[n_small]) * to_ms
                    med_batch_quant[method_id, :] = quartiles(bench[n_med]) * to_ms
                    large_batch_quant[method_id, :] = quartiles(bench[n_large]) * to_ms
                    huge_batch_quant[method_id, :] = quartiles(bench[n_huge]) * to_ms

    labels = []
    for method_name in methods_to_idx.keys():
        labels.append(method_name_to_label(method_name))

    x = np.arange(len(labels))
    width = 0.2

    def makelabel(n):
        if n == n_small:
            return "Batch size " + str(n)
        return "" + str(n)

    def gray(s):
        return [s, s, s]

    ax.grid(axis='y')


    def errorbars(xx, batch, batch_quant):
        ax.errorbar(
            xx,
            batch,
            yerr=(batch.reshape(-1, 1) - batch_quant).T,
            capsize=4, elinewidth=1, markeredgewidth=1,
            fmt="none",
            color="k",
            barsabove=True
        )

    xx = x - width * 1.5
    ax.bar(xx, small_batch, width, label=makelabel(n_small), color=gray(.2))
    errorbars(xx, small_batch, small_batch_quant)
    xx = x - width / 2
    ax.bar(xx, med_batch, width, label=makelabel(n_med), color=gray(.4))
    errorbars(xx, med_batch, med_batch_quant)
    xx = x + width / 2
    ax.bar(xx, large_batch, width, label=makelabel(n_large), color=gray(.6))
    errorbars(xx, large_batch, large_batch_quant)
    xx = x + width * 1.5
    ax.bar(xx, huge_batch, width, label=makelabel(n_huge), color=gray(.8))
    errorbars(xx, huge_batch, huge_batch_quant)

    ax.set_ylabel('Time [ms]')
    ax.set_title('Benchmark: CIFAR10 on 3C3D')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([0, '', 50, '', 100])
    ax.legend(ncol=4)

    ax.set_xlim([-0.5, 8.5])

    plt.tight_layout()
    return fig


def make_cifar10_plot_relative():
    fig = plt.figure(figsize=(5, 4.5))

    grid = plt.GridSpec(
        1, 1,
        wspace=0.15, hspace=0.1,
        left=0.175,
        right=0.975,
        top=0.8,
        bottom=0.45,
    )
    ax = fig.add_subplot(grid[0])

    with open("benchmark_cifar10.pk", 'rb') as handle:
        results = pickle.load(handle)

        methods_to_idx = {
            "_grad": 0,
            "SecondMoment": 1,
            "BatchL2": 2,
            "BatchGrad": 3,
            "Var": 4,
            "DiagGGNMC": 5,
            "KFAC": 6,
            "KFLR": 7,
            "DiagGGNExact": 8,
        }

        huge_batch = np.zeros(len(methods_to_idx))
        huge_batch_quant = np.zeros((len(methods_to_idx), 2))
        n_huge = 128
        for name, bench in results.items():
            print(name, bench.keys())
            for method_name, method_id in methods_to_idx.items():
                if method_name in name:
                    huge_batch[method_id] = np.median(bench[n_huge]) * to_ms

                    huge_batch_quant[method_id, :] = quartiles(bench[n_huge]) * to_ms

    labels = []
    for method_name in methods_to_idx.keys():
        labels.append(method_name_to_label(method_name))

    x = np.arange(len(labels))
    width = 0.5

    def gray(s):
        return [s, s, s]

    ax.grid(axis='y')

    def errorbars(xx, batch, batch_quant):
        ax.errorbar(
            xx,
            batch/ huge_batch[0],
            yerr=(batch.reshape(-1, 1) - batch_quant).T/ huge_batch[0],
            capsize=4, elinewidth=1, markeredgewidth=1,
            fmt="none",
            color="k",
            barsabove=True
        )

    xx = x
    ax.bar(xx, [(v / huge_batch[0]) for v in huge_batch], width, color=gray(.8))
    errorbars(xx, huge_batch, huge_batch_quant)

    ax.set_ylabel('Relative Time\n(rel. to Gradient)')
    ax.set_title('CIFAR10 on 3C3D')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    #ax.set_yticks([0, 25, 50, 75, 100])
    #ax.set_yticklabels([0, '', 50, '', 100])
    ax.set_ylim([0,2])
#    ax.legend(ncol=4)

    ax.set_xlim([-0.5, 8.5])

    plt.tight_layout()
    return fig


def make_appendix_diagh_plot():
    fig = plt.figure(figsize=(8, 3))

    grid = plt.GridSpec(
        1, 1,
        wspace=0.15, hspace=0.1,
        left=0.125,
        right=0.975,
        top=0.88,
        bottom=0.15,
    )
    ax = fig.add_subplot(grid[0])

    with open("additional_benchmark_cifar10_sigmoid.pk", 'rb') as handle:
        results = pickle.load(handle)

        methods_to_idx = {
            "_grad": 0,
            "diagGGN": 1,
            "diagH": 2,
        }

        small_batch = np.zeros(len(methods_to_idx))
        med_batch = np.zeros(len(methods_to_idx))
        large_batch = np.zeros(len(methods_to_idx))
        huge_batch = np.zeros(len(methods_to_idx))
        small_batch_quant = np.zeros((len(methods_to_idx), 2))
        med_batch_quant = np.zeros((len(methods_to_idx), 2))
        large_batch_quant = np.zeros((len(methods_to_idx), 2))
        huge_batch_quant = np.zeros((len(methods_to_idx), 2))
        n_small = 4
        n_med = 8
        n_large = 16
        n_huge = 32
        for name, bench in results.items():
            print(name, bench.keys())
            for method_name, method_id in methods_to_idx.items():
                if method_name in name:
                    small_batch[method_id] = np.median(bench[n_small]) * to_ms
                    med_batch[method_id] = np.median(bench[n_med]) * to_ms
                    large_batch[method_id] = np.median(bench[n_large]) * to_ms
                    huge_batch[method_id] = np.median(bench[n_huge]) * to_ms

                    small_batch_quant[method_id, :] = quartiles(bench[n_small]) * to_ms
                    med_batch_quant[method_id, :] = quartiles(bench[n_med]) * to_ms
                    large_batch_quant[method_id, :] = quartiles(bench[n_large]) * to_ms
                    huge_batch_quant[method_id, :] = quartiles(bench[n_huge]) * to_ms

    labels = []
    for method_name in methods_to_idx.keys():
        labels.append(method_name_to_label(method_name))

    x = np.arange(len(labels))
    width = 0.2

    def makelabel(n):
        if n == n_small:
            return "Batch = " + str(n)
        return "" + str(n)

    def gray(s):
        return [s, s, s]

    ax.grid(axis='y')

    def errorbars(xx, batch, batch_quant):
        ax.errorbar(
            xx,
            batch,
            yerr=(batch.reshape(-1, 1) - batch_quant).T,
            capsize=4, elinewidth=1, markeredgewidth=1,
            fmt="none",
            color="k",
            barsabove=True
        )

    xx = x - width * 1.5
    ax.bar(xx, small_batch, width, label=makelabel(n_small), color=gray(.2))
    errorbars(xx, small_batch, small_batch_quant)
    xx = x - width / 2
    ax.bar(xx, med_batch, width, label=makelabel(n_med), color=gray(.4))
    errorbars(xx, med_batch, med_batch_quant)
    xx = x + width / 2
    ax.bar(xx, large_batch, width, label=makelabel(n_large), color=gray(.6))
    errorbars(xx, large_batch, large_batch_quant)
    xx = x + width * 1.5
    ax.bar(xx, huge_batch, width, label=makelabel(n_huge), color=gray(.8))
    errorbars(xx, huge_batch, huge_batch_quant)

    ax.set_ylabel('Time [ms]')
    ax.set_title('CIFAR10 on 3C3D with one sigmoid')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    #ax.set_yticks([0, 25, 50, 75, 100])
    #ax.set_yticklabels([0, '', 50, '', 100])
    ax.set_yscale("log")
    ax.legend(ncol=2)

    ax.set_xlim([-0.5, 2.5])

    plt.tight_layout()
    return fig


def make_cifar100_plot():
    fig = plt.figure(figsize=(12, 3.25))

    grid = plt.GridSpec(
        1, 2,
        wspace=0.15, hspace=0.1,
        left=0.075,
        right=0.975,
        top=0.9,
        bottom=0.35,
        width_ratios=[2.75, 1]
    )
    ax = fig.add_subplot(grid[0])

    with open("benchmark_cifar100.pk", 'rb') as handle:
        results = pickle.load(handle)

        methods_to_idx = {
            "_grad": 0,
            "SecondMoment": 1,
            "BatchL2": 2,
            "DiagGGNMC": 3,
            "KFAC": 4,
            "BatchGrad": 5,
            "Var": 6,
            #            "KFLR": 7,
            #            "DiagGGNExact": 8,
        }

        small_batch = np.zeros(len(methods_to_idx))
        med_batch = np.zeros(len(methods_to_idx))
        large_batch = np.zeros(len(methods_to_idx))
        huge_batch = np.zeros(len(methods_to_idx))
        small_batch_quant = np.zeros((len(methods_to_idx), 2))
        med_batch_quant = np.zeros((len(methods_to_idx), 2))
        large_batch_quant = np.zeros((len(methods_to_idx), 2))
        huge_batch_quant = np.zeros((len(methods_to_idx), 2))
        n_small = 16
        n_med = 32
        n_large = 48
        n_huge = 64
        for name, bench in results.items():
            print(name, bench.keys())
            for method_name, method_id in methods_to_idx.items():
                if method_name in name:
                    small_batch[method_id] = np.median(bench[n_small]) * to_ms
                    med_batch[method_id] = np.median(bench[n_med]) * to_ms
                    large_batch[method_id] = np.median(bench[n_large]) * to_ms
                    huge_batch[method_id] = np.median(bench[n_huge]) * to_ms

                    small_batch_quant[method_id, :] = quartiles(bench[n_small]) * to_ms
                    med_batch_quant[method_id, :] = quartiles(bench[n_med]) * to_ms
                    large_batch_quant[method_id, :] = quartiles(bench[n_large]) * to_ms
                    huge_batch_quant[method_id, :] = quartiles(bench[n_huge]) * to_ms

    labels = []
    for method_name in methods_to_idx.keys():
        labels.append(method_name_to_label(method_name))

    x = np.arange(len(labels))
    width = 0.2

    def makelabel(n):
        if n == n_small:
            return "Batch = " + str(n)
        return "" + str(n)

    def gray(s):
        return [s, s, s]

    ax.grid(axis='y')

    def errorbars(xx, batch, batch_quant):
        ax.errorbar(
            xx,
            batch,
            yerr=(batch.reshape(-1, 1) - batch_quant).T,
            capsize=4, elinewidth=1, markeredgewidth=1,
            fmt="none",
            color="k",
            barsabove=True
        )

    xx = x - width * 1.5
    ax.bar(xx, small_batch, width, label=makelabel(n_small), color=gray(.2))
    errorbars(xx, small_batch, small_batch_quant)
    xx = x - width / 2
    ax.bar(xx, med_batch, width, label=makelabel(n_med), color=gray(.4))
    errorbars(xx, med_batch, med_batch_quant)
    xx = x + width / 2
    ax.bar(xx, large_batch, width, label=makelabel(n_large), color=gray(.6))
    errorbars(xx, large_batch, large_batch_quant)
    xx = x + width * 1.5
    ax.bar(xx, huge_batch, width, label=makelabel(n_huge), color=gray(.8))
    errorbars(xx, huge_batch, huge_batch_quant)

    ax.set_ylabel('Time [ms]')
    ax.set_title('Benchmark: CIFAR100 on All-CNN-C')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([0, '', 50, '', 100])
    ax.legend(ncol=4, labelspacing=0.1, columnspacing=0.5, handletextpad=0.5)

    ax.set_xlim([-0.5, 6.5])

    ax = fig.add_subplot(grid[1])

    with open("benchmark_cifar100.pk", 'rb') as handle:
        results = pickle.load(handle)

        methods_to_idx = {
            "_grad": 0,
            "KFAC": 1,
            "KFLR": 2,
        }

        small_batch = np.zeros(len(methods_to_idx))
        med_batch = np.zeros(len(methods_to_idx))
        small_batch_quant = np.zeros((len(methods_to_idx), 2))
        med_batch_quant = np.zeros((len(methods_to_idx), 2))
        n_small = 16
        n_med = 32
        for name, bench in results.items():
            print(name, bench.keys())
            for method_name, method_id in methods_to_idx.items():
                if method_name in name:
                    small_batch[method_id] = np.median(bench[n_small]) * to_ms
                    med_batch[method_id] = np.median(bench[n_med]) * to_ms

                    small_batch_quant[method_id, :] = quartiles(bench[n_small]) * to_ms
                    med_batch_quant[method_id, :] = quartiles(bench[n_med]) * to_ms

    labels = []
    for method_name in methods_to_idx.keys():
        labels.append(method_name_to_label(method_name))

    x = np.arange(len(labels)) / 2
    width = 0.2

    def makelabel(n):
        return str(n)

    def gray(s):
        return [s, s, s]

    ax.grid(axis='y')

    def errorbars(xx, batch, batch_quant):
        ax.errorbar(
            xx,
            batch,
            yerr=(batch.reshape(-1, 1) - batch_quant).T,
            capsize=4, elinewidth=1, markeredgewidth=1,
            fmt="none",
            color="k",
            barsabove=True
        )

    xx = x - width / 2
    ax.bar(xx, small_batch, width, label=makelabel(n_small), color=gray(.2))
    errorbars(xx, small_batch, small_batch_quant)
    xx = x + width / 2
    ax.bar(xx, med_batch, width, label=makelabel(n_med), color=gray(.4))
    errorbars(xx, med_batch, med_batch_quant)

    #    ax.set_ylabel('Time [ms]')
    ax.set_title('KFLR')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
    ax.set_yticklabels([0, '', 200, '', 400, '', 600])
    ax.legend(ncol=1)

    ax.set_xlim([-0.25, 1.25])

    return fig


def make_forloop_comparison():
    font = {'family': 'serif', 'size': 18}
    matplotlib.rc('font', **font)


    fig = plt.figure(figsize=(6, 3))

    grid = plt.GridSpec(
        1, 1,
        wspace=0.15, hspace=0.1,
        left=0.15,
        right=0.975,
        top=0.89,
        bottom=0.2,
    )
    ax = fig.add_subplot(grid[0])

    with open("mytest_benchmark.pk", 'rb') as handle:
        results = pickle.load(handle)

        med_batch = [0, 0, 0]
        large_batch = [0, 0, 0]
        huge_batch = [0, 0, 0]
        n_med = 64
        n_large = 128
        n_huge = 256

        to_ms = 1000
        for name, bench in results.items():
            if "forloop" in name:
                med_batch[0] = np.median(bench[n_med]) * to_ms
                large_batch[0] = np.median(bench[n_large]) * to_ms
                huge_batch[0] = np.median(bench[n_huge]) * to_ms
            elif "BatchGrad" in name:
                med_batch[1] = np.median(bench[n_med]) * to_ms
                large_batch[1] = np.median(bench[n_large]) * to_ms
                huge_batch[1] = np.median(bench[n_huge]) * to_ms
            elif "_grad" in name:
                med_batch[2] = np.median(bench[n_med]) * to_ms
                large_batch[2] = np.median(bench[n_large]) * to_ms
                huge_batch[2] = np.median(bench[n_huge]) * to_ms

    labels = ['For-loop', 'BackPACK', 'Gradient\n(Ref.)']

    x = np.arange(len(labels))
    width = 0.25

    def makelabel(n):
        if n == n_med:
            return "Batch size " + str(n)
        return "                 " + str(n)

    def gray(s):
        return [s, s, s]

    #ax.grid(axis='y')
    xx = x - width
    ax.bar(xx[2], med_batch[2], width, color=gray(.2))
    ax.bar(xx[:2], med_batch[:2], width, label=makelabel(n_med), color=barcolors[0])
    xx = x
    ax.bar(xx[2], large_batch[2], width, color=gray(.4))
    ax.bar(xx[:2], large_batch[:2], width, label=makelabel(n_large), color=barcolors[1])
    xx = x + width
    ax.bar(xx[:2], huge_batch[:2], width, label=makelabel(n_huge), color=barcolors[2])
    ax.bar(xx[2], huge_batch[2], width, color=gray(.6))


    ax.axhline(med_batch[2], color=gray(.2), alpha=.4)
    ax.axhline(large_batch[2], color=gray(.4), alpha=.4)
    ax.axhline(huge_batch[2],color=gray(.6), alpha=.4)

    ax.set_ylabel('Time [ms]')
    ax.set_title('Batch gradients')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.set_yticks([0, 100, 200, 300, 400])
    #ax.set_yticklabels([0, '', 200, '', 400])
    ax.legend()

    return fig


def make_compressed_plot():
    fig = plt.figure(figsize=(12, 2.75))

    grid = plt.GridSpec(
        1, 2,
        wspace=0.1,
        hspace=0.1,
        left=0.05,
        right=0.975,
        top=0.9,
        bottom=0.25,
        width_ratios=[1.3, 1]

    )
    ax = fig.add_subplot(grid[0])

    with open("benchmark_cifar10.pk", 'rb') as handle:
        results = pickle.load(handle)

        methods_to_idx = {
            "_grad": 0,
            "SecondMoment": 1,
            "BatchL2": 2,
            "DiagGGNMC": 3,
            "KFAC": 4,
            "BatchGrad": 5,
            "KFLR": 6,
            "DiagGGNExact": 7,
        }

        med_batch = np.zeros(len(methods_to_idx))
        large_batch = np.zeros(len(methods_to_idx))
        huge_batch = np.zeros(len(methods_to_idx))
        med_batch_quant = np.zeros((len(methods_to_idx), 2))
        large_batch_quant = np.zeros((len(methods_to_idx), 2))
        huge_batch_quant = np.zeros((len(methods_to_idx), 2))
        n_med = 32
        n_large = 48
        n_huge = 64
        for name, bench in results.items():
            print(name, bench.keys())
            for method_name, method_id in methods_to_idx.items():
                if method_name in name:
                    med_batch[method_id] = np.median(bench[n_med]) * to_ms
                    large_batch[method_id] = np.median(bench[n_large]) * to_ms
                    huge_batch[method_id] = np.median(bench[n_huge]) * to_ms

                    med_batch_quant[method_id, :] = quartiles(bench[n_med]) * to_ms
                    large_batch_quant[method_id, :] = quartiles(bench[n_large]) * to_ms
                    huge_batch_quant[method_id, :] = quartiles(bench[n_huge]) * to_ms

    labels = []
    for method_name in methods_to_idx.keys():
        labels.append(method_name_to_label(method_name))

    x = np.arange(len(labels))
    width = 0.25

    def makelabel(n):
        if n == n_med:
            return "Batch = " + str(n)
        return "" + str(n)

    def gray(s):
        return [s, s, s]


    def errorbars(xx, batch, batch_quant):
        if False:
            ax.errorbar(
                xx,
                batch,
                yerr=(batch.reshape(-1, 1) - batch_quant).T,
                capsize=4, elinewidth=1, markeredgewidth=1,
                fmt="none",
                color="k",
                barsabove=True,
                alpha=.5
            )

    xx = x - width
    ax.bar(xx[0], med_batch[0], width, color=gray(.2))
    ax.bar(xx[1:], med_batch[1:], width, label=makelabel(n_med), color=barcolors[0])
    errorbars(xx, med_batch, med_batch_quant)
    xx = x
    ax.bar(xx[0], large_batch[0], width, color=gray(.4))
    ax.bar(xx[1:], large_batch[1:], width, label=makelabel(n_large), color=barcolors[1])
    errorbars(xx, large_batch, large_batch_quant)
    xx = x + width
    ax.bar(xx[0], huge_batch[0], width, color=gray(.6))
    ax.bar(xx[1:], huge_batch[1:], width, label=makelabel(n_huge), color=barcolors[2])
    errorbars(xx, huge_batch, huge_batch_quant)


    ax.axhline(med_batch[0], color=gray(.2), alpha=.4)
    ax.axhline(large_batch[0], color=gray(.4), alpha=.4)
    ax.axhline(huge_batch[0],color=gray(.6), alpha=.4)

    ax.set_ylabel('Time [ms]')
    ax.set_title('3C3D on CIFAR10')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticks([0, 25, 50])
    ax.set_yticklabels([0, '', 50])
    ax.legend(ncol=4, loc="upper left", labelspacing=0.1, columnspacing=0.5, handletextpad=0.5)

    xlabels = ax.get_xticklabels()
    new_xlabels = []
    for i, x in enumerate(xlabels):
        if i % 2 == 0:
            new_xlabels.append("\n"+x._text)
        else:
            new_xlabels.append(x._text)
    ax.set_xticklabels(new_xlabels, rotation=0)

    ax.set_xlim([-0.5, len(methods_to_idx) - 0.5])



    ax = fig.add_subplot(grid[1])


    with open("benchmark_cifar100.pk", 'rb') as handle:
        results = pickle.load(handle)

        methods_to_idx = {
            "_grad": 0,
            "SecondMoment": 1,
            "BatchL2": 2,
            "DiagGGNMC": 3,
            "KFAC": 4,
            "BatchGrad": 5,
        }

        med_batch = np.zeros(len(methods_to_idx))
        large_batch = np.zeros(len(methods_to_idx))
        huge_batch = np.zeros(len(methods_to_idx))
        med_batch_quant = np.zeros((len(methods_to_idx), 2))
        large_batch_quant = np.zeros((len(methods_to_idx), 2))
        huge_batch_quant = np.zeros((len(methods_to_idx), 2))
        n_med = 32
        n_large = 48
        n_huge = 64
        for name, bench in results.items():
            print(name, bench.keys())
            for method_name, method_id in methods_to_idx.items():
                if method_name in name:
                    med_batch[method_id] = np.median(bench[n_med]) * to_ms
                    large_batch[method_id] = np.median(bench[n_large]) * to_ms
                    huge_batch[method_id] = np.median(bench[n_huge]) * to_ms

                    med_batch_quant[method_id, :] = quartiles(bench[n_med]) * to_ms
                    large_batch_quant[method_id, :] = quartiles(bench[n_large]) * to_ms
                    huge_batch_quant[method_id, :] = quartiles(bench[n_huge]) * to_ms

    labels = []
    for method_name in methods_to_idx.keys():
        labels.append(method_name_to_label(method_name))

    x = np.arange(len(labels))
    width = 0.25

    def makelabel(n):
        if n == n_med:
            return "Batch = " + str(n)
        return "" + str(n)

    def gray(s):
        return [s, s, s]

    xx = x - width
    ax.bar(xx[0], med_batch[0], width, color=gray(.2))
    ax.bar(xx[1:], med_batch[1:], width, label=makelabel(n_med), color=barcolors[0])
    errorbars(xx, med_batch, med_batch_quant)
    xx = x
    ax.bar(xx[0], large_batch[0], width, color=gray(.4))
    ax.bar(xx[1:], large_batch[1:], width, label=makelabel(n_large), color=barcolors[1])
    errorbars(xx, large_batch, large_batch_quant)
    xx = x + width
    ax.bar(xx[0], huge_batch[0], width, color=gray(.6))
    ax.bar(xx[1:], huge_batch[1:], width, label=makelabel(n_huge), color=barcolors[2])
    errorbars(xx, huge_batch, huge_batch_quant)

    ax.set_title('All-CNN-C on CIFAR100')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([0, '', 50, '', 100])
    ax.legend(ncol=4, labelspacing=0.1, columnspacing=0.5, handletextpad=0.5)

    ax.axhline(med_batch[0], color=gray(.2), alpha=.4)
    ax.axhline(large_batch[0], color=gray(.4), alpha=.4)
    ax.axhline(huge_batch[0],color=gray(.6), alpha=.4)

    ax.set_xlim([-0.5, len(methods_to_idx) - 0.5])

    xlabels = ax.get_xticklabels()
    new_xlabels = []
    for i, x in enumerate(xlabels):
        if i % 2 == 0:
            new_xlabels.append("\n"+x._text)
        else:
            new_xlabels.append(x._text)
    ax.set_xticklabels(new_xlabels, rotation=0)

    return fig


if __name__ == "__main__":


    fig = make_forloop_comparison()
    name = "bench_barplot.pdf"
    fig.savefig(name, bbox_inches='tight', transparent=True)
    fig.savefig("notransparency-" + name, bbox_inches='tight', transparent=False)
    plt.show()

    if False:
        fig = make_compressed_plot()
        name = "bench_compressed.pdf"
        fig.savefig(name, bbox_inches='tight', transparent=True)
        fig.savefig("notransparency-" + name, bbox_inches='tight', transparent=False)
        plt.show()
        fig = make_appendix_diagh_plot()
        name = "bench_diagh.pdf"
        fig.savefig(name, bbox_inches='tight', transparent=True)
        fig.savefig("notransparency-" + name, bbox_inches='tight', transparent=False)
        plt.show()



        fig = make_cifar100_plot()
        name = "bench_cifar100.pdf"
        fig.savefig(name, bbox_inches='tight', transparent=True)
        fig.savefig("notransparency-" + name, bbox_inches='tight', transparent=False)
        plt.show()

        fig = make_cifar10_plot()
        name = "bench_cifar10.pdf"
        fig.savefig(name, bbox_inches='tight', transparent=True)
        fig.savefig("notransparency-" + name, bbox_inches='tight', transparent=False)
        plt.show()

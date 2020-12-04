import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.algorithms.definitions import get_definitions
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (get_plot_label, compute_metrics,
                                           create_linestyles, create_pointset)
from ann_benchmarks.results import (store_results, load_all_results,
                                    get_unique_algorithms, get_algorithm_name)

from matplotlib.colors import hsv_to_rgb
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

import seaborn as sns

hues = [0.6083, 0.7771, 0.275, 9416, 0.0]
sats = [0.2,] # [0.2, 0.15, 0.1]
vals = [0.8,] # [0.5, 0.66, 0.8]
# color_sequence = [tuple(hsv_to_rgb((h, sats[i] if h > 0 else 0.0, vals[i]))) + (0.8,)
#                   for h in hues for i in range(len(sats))]
color_sequence = sns.color_palette('YlGnBu', 5, desat=0.4)
# color_sequence = sns.hls_palette(4, l=0.8, s=0.3)

def create_plot(all_data, raw, x_log, y_log, xn, yn, fn_out, linestyles,
                batch, dataset, x_lims=None, smooth=True):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))
    color_counter = 0
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        if algo != "pynndescent":
            xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
            color, faded, linestyle, marker = linestyles[algo]

            color = color_sequence[color_counter]
            color_counter += 1

            if x_lims is not None and smooth:
                xs = np.array(xs)
                mask = ((xs > x_lims[0] - 0.01 * x_lims[0]) &
                        (xs < x_lims[1] + 0.01 * x_lims[1]))
                xs = xs[mask]
                ys = np.array(ys)[mask]

            if len(xs) <= 2:
                marker = '.'
                new_xs = xs
                new_ys = ys
            elif smooth:
                smooth = make_interp_spline(xs, ys, k=1)
                new_xs = np.linspace(np.min(xs), np.max(xs), 1000)
                new_ys = smooth(new_xs)
                new_ys = gaussian_filter1d(new_ys, sigma=3)
            else:
                new_xs = xs
                new_ys = ys

            handle, = plt.plot(
                new_xs, new_ys,
                # xs, ys,
                '-', label=algo, color=color,
                ms=7, mew=3, lw=3, linestyle=linestyle,
                marker=marker
            )
            handles.append(handle)
            if raw:
                handle2, = plt.plot(axs, ays, '-', label=algo, color=faded,
                                    ms=5, mew=2, lw=2, linestyle=linestyle,
                                    marker=marker)
            labels.append(get_algorithm_name(algo, batch))

    if "pynndescent" in all_data.keys():
        xs, ys, ls, axs, ays, als = create_pointset(all_data["pynndescent"], xn, yn)
        color = (0.75, 0.0, 0.0, 0.9)
        marker = ','

        if x_lims is not None and smooth:
            xs = np.array(xs)
            mask = ((xs > x_lims[0] - 0.01 * x_lims[0]) &
                    (xs < x_lims[1] + 0.01 * x_lims[1]))
            xs = xs[mask]
            ys = np.array(ys)[mask]

        if smooth:
            smooth = make_interp_spline(xs, ys, k=1)
            new_xs = np.linspace(np.min(xs), np.max(xs), 1000)
            new_ys = smooth(new_xs)
            new_ys = gaussian_filter1d(new_ys, sigma=3)
        else:
            new_xs = xs
            new_ys = ys

        handle, = plt.plot(
            new_xs, new_ys,
            # xs, ys,
            '-', label="pynndescent", color=color,
            ms=7, mew=3, lw=3, linestyle='-',
            marker=','
        )
        handles.append(handle)
        labels.append(get_algorithm_name("pynndescent", batch))

    if x_log:
        plt.gca().set_xscale('log')
    if y_log:
        plt.gca().set_yscale('log')
    plt.gcf().suptitle(dataset, fontsize=24)
    plt.gca().set_title(get_plot_label(xm, ym), fontsize=18)
    plt.gca().set_ylabel(ym['description'])
    plt.gca().set_xlabel(xm['description'])
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(handles, labels, loc='center left',
                     bbox_to_anchor=(1, 0.5), prop={'size': 9})
#    plt.grid(b=True, which='major', color='0.65', linestyle='-')

    # Allow minor grid lines
    plt.grid(b=True, which='major', color='w', linewidth=1.0)
    plt.grid(b=True, which='minor', color='w', linewidth=0.5, alpha=0.66)

    if 'lim' in xm:
        plt.xlim(xm['lim'])
    if 'lim' in ym:
        plt.ylim(ym['lim'])

    if x_lims is not None:
        plt.xlim(x_lims)
        ax = plt.gca()
        data = np.vstack([line.get_xydata() for line in ax.lines])
        sub_data = data[(data.T[0] > x_lims[0]) & (data.T[0] < x_lims[1])]
        y_min = sub_data.T[1].min()
        y_max = sub_data.T[1].max()
        if y_log:
            y_range = np.log10(y_max) - np.log10(y_min)
            y_min = 10**(np.log10(y_min) - 0.1 * y_range)
            y_max = 10**(np.log10(y_max) + 0.1 * y_range)
        else:
            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
        plt.ylim((y_min, y_max))
        # plt.autoscale(axis='y')
    plt.savefig(fn_out, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        metavar="DATASET",
        default='glove-100-angular')
    parser.add_argument(
        '--count',
        default=10)
    parser.add_argument(
        '--definitions',
        metavar='FILE',
        help='load algorithm definitions from FILE',
        default='algos.yaml')
    parser.add_argument(
        '--limit',
        default=-1)
    parser.add_argument(
        '-o', '--output')
    parser.add_argument(
        '-x', '--x-axis',
        help='Which metric to use on the X-axis',
        choices=metrics.keys(),
        default="k-nn")
    parser.add_argument(
        '-y', '--y-axis',
        help='Which metric to use on the Y-axis',
        choices=metrics.keys(),
        default="qps")
    parser.add_argument(
        '-X', '--x-log',
        help='Draw the X-axis using a logarithmic scale',
        action='store_true')
    parser.add_argument(
        '-Y', '--y-log',
        help='Draw the Y-axis using a logarithmic scale',
        action='store_true')
    parser.add_argument(
        '--raw',
        help='Show raw results (not just Pareto frontier) in faded colours',
        action='store_true')
    parser.add_argument(
        '--batch',
        help='Plot runs in batch mode',
        action='store_true')
    parser.add_argument(
        '--recompute',
        help='Clears the cache and recomputes the metrics',
        action='store_true')
    parser.add_argument(
        '--x_lim_left',
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '--x_lim_right',
        type=float,
        default=1.05,
    )
    parser.add_argument(
        '--smooth',
        help='Apply a little smoothing to the lines',
        action='store_true')
    args = parser.parse_args()

    if not args.output:
        args.output = 'results/%s.png' % get_algorithm_name(
            args.dataset, args.batch)
        print('writing output to %s' % args.output)

    dataset = get_dataset(args.dataset)
    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(args.dataset, count, True, args.batch)
    linestyles = create_linestyles(sorted(unique_algorithms))
    runs = compute_metrics(np.array(dataset["distances"]),
                           results, args.x_axis, args.y_axis, args.recompute)
    if not runs:
        raise Exception('Nothing to plot')

    x_lims = (args.x_lim_left, args.x_lim_right)
    create_plot(runs, args.raw, args.x_log,
                args.y_log, args.x_axis, args.y_axis, args.output,
                linestyles, args.batch, args.dataset, x_lims=x_lims, smooth=args.smooth)

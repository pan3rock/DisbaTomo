#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde
import os
from objective_function import (ObjectiveFunctionDerivativeUsed, Forward)

import yaml

params = {'axes.labelsize': 14,
          'axes.titlesize': 16,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.fontsize': 14}
plt.rcParams.update(params)


def kde_scipy(x, x_grid, weights):
    if len(x) == 1:
        ret = np.zeros_like(x_grid)
        ind = np.argmin(np.abs(x[0] - x_grid))
        ret[ind] = 1.0
    else:
        if np.all(np.isclose(x, x[0])):
            x += 1.0e-10 * np.random.random(len(x))
        kde = gaussian_kde(x, weights=weights)
        ret = kde.evaluate(x_grid)
    return ret


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    num = len(values)
    if num == 1:
        variance = 0
    else:
        variance = np.average(
            (values - average)**2, weights=weights) * num / (num - 1)
    return (average, np.sqrt(variance))


def plot_disp(config, sid, all_disp, file_out, no_show):
    config_plot = config['plot']
    dir_output = config['dir_output']
    wave_type = config.get('wave_type', 'rayleigh')
    dir_output = os.path.join(dir_output, sid)

    results = [
        np.load(dir_output + '/' + f, allow_pickle=True)
        for f in os.listdir(dir_output)
    ]
    misfit = np.asarray([r['fi'] for r in results])
    roots_inv = [r['ri'].item() for r in results]
    models_inv = [r['mi'] for r in results]
    weight = np.exp(-misfit)
    weight /= np.sum(weight)

    if len(weight) == 1:
        ratio_show = 1
    else:
        ratio_show = config_plot['percentage_show'] * 0.01
    num_show = int(misfit.shape[0] * ratio_show)
    ind_sort = np.argsort(weight)[::-1][:num_show]
    weight = weight[ind_sort]
    weight = weight / np.amax(weight) * 0.2
    roots_inv = [roots_inv[i] for i in ind_sort]
    models_inv = [models_inv[i] for i in ind_sort]

    num_show = int(misfit.shape[0] * ratio_show)

    # weight *= 1.0 / np.mean(weight) * 0.05
    # weight[weight > 0.1] = 0.1
    dir_data = config['dir_data']
    file_data = os.path.join(dir_data, '{:s}.txt'.format(sid))
    data = np.loadtxt(file_data)

    _, ax = plt.subplots()
    modes = []

    if all_disp:
        weights_mode = config.get("weights_mode", None)
        for ind, (w, model) in enumerate(zip(weight, models_inv)):
            forward = Forward(model, data, wave_type)
            forward.compute(weights_mode)
            disp = forward.disp
            for m, val in disp.items():
                modes.append(m)
                pinv, = ax.plot(val[:, 0], val[:, 1], 'k-', alpha=w)
    else:
        for ind, (w, r) in enumerate(zip(weight, roots_inv)):
            if ind >= num_show:
                continue
            for mode, v in r.items():
                rinv = np.array(v)
                if np.size(rinv) > 0:
                    pinv, = ax.plot(rinv[:, 0], rinv[:, 1], 'k-', alpha=w)
                    modes.append(mode)

    modes = set(modes)
    for mode in modes:
        disp_m = data[data[:, 2].astype(int) == mode]
        pdata, = ax.plot(disp_m[:, 0], disp_m[:, 1], 'r.', alpha=0.5)

    plt.legend(handles=[pinv, pdata],
               labels=['inversion', 'data'],
               loc='lower left')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase velocity (km/s)')
    plt.tight_layout()
    if file_out:
        plt.savefig(file_out, dpi=300)
    if not no_show:
        plt.show()


def plot_model(config_inv, sid, plot_init, file_out, no_show):
    config_plot = config_inv['plot']
    zmax = config_plot['zmax']
    vsmin, vsmax = config_plot['vs_lim']
    file_model_data = config_plot.get('model_data', None)
    file_model_output = config_plot.get("model_output", "model_output.txt")
    dir_output = config_inv['dir_output']
    dir_output = os.path.join(dir_output, sid)
    file_data = '{:s}.txt'.format(sid)

    results = [np.load(dir_output + '/' + f) for f in os.listdir(dir_output)]
    misfit = np.asarray([r['fi'] for r in results])

    model_stat = np.asarray([r['mi'][:, 3] for r in results])
    model_0 = np.asarray([r['m0'][:, 3] for r in results])

    weight = np.exp(-misfit)

    if len(weight) == 1:
        ratio_show = 1
    else:
        ratio_show = config_plot['percentage_show'] * 0.01
    num_show = int(misfit.shape[0] * ratio_show)
    ind_sort = np.argsort(weight)[::-1][:num_show]
    weight = weight[ind_sort]
    weight /= np.sum(weight)
    model_stat = model_stat[ind_sort, :]

    if plot_init:
        model_stat = model_0[ind_sort, :]
        weight = np.ones_like(weight)
        weight /= np.sum(weight)
        file_model_data = None

    if zmax < 0.5:
        unit = 'm'
        km2m = 1000
    else:
        unit = 'km'
        km2m = 1

    plt.figure()

    file_model_init = config_inv['model_init']
    model_init = np.loadtxt(file_model_init)
    z = model_init[:, 1]
    hw = config_inv['init_half_width']
    vs = model_stat

    z_plot = np.append(z, zmax) * km2m
    wmax = np.amax(weight)

    labels = []
    handles = []
    for i in range(vs.shape[0]):
        vs_plot = np.append(vs[i, :], vs[i, -1])
        alpha = weight[i] / wmax * 0.2
        p1, = plt.step(vs_plot, z_plot, 'k-',
                       alpha=alpha, linewidth=2)

    handles.append(p1)
    if plot_init:
        labels.append('initial model')
    else:
        labels.append('inverted model')

    ml = []
    sl = []
    zl = []
    for i in range(vs.shape[1]):
        mean, std = weighted_avg_and_std(vs[:, i], weight)
        ml.append(mean)
        sl.append(std)
        zl.append((z_plot[i] + z_plot[i + 1]) / 2.0)

    mp = ml[:] + [
        ml[-1],
    ]
    zp = z_plot[:]
    if not plot_init:
        p2, = plt.step(mp, zp, '-', c='r', alpha=0.8, linewidth=2)
        handles.append(p2)
        labels.append('estimated model')

    vs_init = model_init[:, 3]
    hw = config_inv['init_half_width']
    v1, v2 = vs_init - hw, vs_init + hw
    vs1_plot = np.append(v1, v1[-1])
    vs2_plot = np.append(v2, v2[-1])
    plt.step(vs1_plot,
             z_plot,
             '--',
             c='gray',
             alpha=0.8)
    plt.step(vs2_plot, z_plot, '--', c='gray', alpha=0.8)

    of = ObjectiveFunctionDerivativeUsed(config_inv, file_data)
    x = (np.asarray(ml) - v1) / (v2 - v1)
    model = of._update_model(x)
    std = np.asarray(sl).reshape(-1, 1)
    model = np.hstack((model, std))
    fmt = "%5d%12.6f%12.6f%12.6f%12.6f%12.6f"
    np.savetxt(file_model_output, model, fmt=fmt)
    print(("{:>7s}" + "{:>12s}" * 5).format('No.', 'z', 'rho', 'vs', 'vp',
                                            'std'))
    for i in range(model.shape[0]):
        print(("{:7.0f}" + "{:12.4f}" * 5).format(*model[i, :]))

    if file_model_data:
        model_data = np.loadtxt(file_model_data)
        z = model_data[:, 1]
        z = np.append(z, [
            zmax,
        ]) * km2m
        vs = model_data[:, 3]
        vs = np.append(vs, [
            vs[-1],
        ])
        p3, = plt.step(vs,
                       z,
                       '-',
                       c='b',
                       alpha=0.6,
                       linewidth=2)
        handles.append(p3)
        labels.append('data model')

    if plot_init:
        file_model_init = config_inv['model_init']
        model_init = np.loadtxt(file_model_init)
        z = model_init[:, 1]
        z = np.append(z, [
            zmax,
        ]) * km2m
        vs = model_init[:, 3]
        vs = np.append(vs, [
            vs[-1],
        ])
        p4, = plt.step(vs, z, 'r-', alpha=0.6)
        handles.append(p4)
        labels.append('reference model')

    plt.legend(handles, labels, loc='lower left')

    if plot_init:
        plt.title('initial model distribution')

    plt.xlim([vsmin, vsmax])
    plt.ylim([0, zmax * km2m])
    plt.xlabel('Vs (km/s)')
    plt.ylabel('Depth ({:s})'.format(unit))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if file_out:
        plt.savefig(file_out, dpi=300)
    if not no_show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot inversion result')
    parser.add_argument('-c', '--config', default='config_inv.yml')
    parser.add_argument('--data', help='data name')
    parser.add_argument('--plot_model', action='store_true')
    parser.add_argument('--plot_disp', action='store_true')
    parser.add_argument('--all_disp', action='store_true')
    parser.add_argument('--plot_init', action='store_true')
    parser.add_argument('--out', default=None,
                        help='filename of output figure')
    parser.add_argument('--no_show', action='store_true')
    args = parser.parse_args()
    file_config = args.config
    dataname = args.data
    show_model = args.plot_model
    show_disp = args.plot_disp
    all_disp = args.all_disp
    show_init = args.plot_init
    file_out = args.out
    no_show = args.no_show

    with open(file_config, 'r') as fp:
        config = yaml.safe_load(fp)

    if show_model:
        plot_model(config, dataname, show_init, file_out, no_show)
    if show_disp:
        plot_disp(config, dataname, all_disp, file_out, no_show)

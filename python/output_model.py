#!/usr/bin/env python
import numpy as np
import argparse
import pandas as pd
import yaml
import os
import shutil
import tqdm


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    values = np.asarray(values).T
    average = np.average(values, weights=weights, axis=1)
    # Fast and numerically precise:
    num = len(values)
    if num == 1:
        variance = 0
    else:
        variance = np.average(
            (values - average.reshape(-1, 1))**2, weights=weights, axis=1) * num / (num - 1)
    return (average, np.sqrt(variance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='output inverted models')
    parser.add_argument('-c', '--config', default='config_inv.yml')
    parser.add_argument('--ratio_best', type=float, default=0.5,
                        help='ratio of models with minimum misfits')
    args = parser.parse_args()
    file_config = args.config
    ratio_best = args.ratio_best

    with open(file_config, 'r') as stream:
        config = yaml.safe_load(stream)
    file_logging = config.get('file_logging', 'record.log')
    dir_inv = config.get('dir_output', 'inversion')

    dir_out = 'model_inv'
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)

    record = pd.read_fwf(file_logging)
    data = set(record['No.data'])

    for dnm in tqdm.tqdm(data, total=len(data)):
        rec = record[record['No.data'] == dnm]
        num = rec.shape[0]
        num_best = int(num * ratio_best)
        rec_sort = rec.sort_values('f(m)')
        index = rec_sort['No.m0'].to_list()[:num_best]
        fitness = rec_sort['f(m)'].to_numpy()[:num_best]
        weight = np.exp(-fitness)
        weight /= np.sum(weight)
        rho = []
        vs = []
        vp = []
        for i in index:
            path = os.path.join(dir_inv, str(dnm), '{:d}.npz'.format(i))
            mi = np.load(path)['mi']
            z = mi[:, 1]
            rho.append(mi[:, 2])
            vs.append(mi[:, 3])
            vp.append(mi[:, 4])

        rho, _ = weighted_avg_and_std(rho, weight)
        vs, std = weighted_avg_and_std(vs, weight)
        vp, _ = weighted_avg_and_std(vp, weight)

        nl = z.shape[0]
        model = np.zeros([nl, 5])
        model[:, 0] = np.arange(nl) + 1.0
        model[:, 1] = z
        model[:, 2] = rho
        model[:, 3] = vs
        model[:, 4] = vp

        path_out = os.path.join(dir_out, '{}.txt'.format(dnm))
        std = np.asarray(std).reshape(-1, 1)
        model = np.hstack((model, std))
        fmt = "%5d%15.6f%15.6f%15.6f%15.6f%20.5e"
        np.savetxt(path_out, model, fmt=fmt)

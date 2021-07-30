#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import yaml
import os


params = {'axes.labelsize': 14,
          'axes.titlesize': 16,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.fontsize': 14}
plt.rcParams.update(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name')
    parser.add_argument('-c', '--config', default='config_inv.yml')
    parser.add_argument('--out', default=None, help='figure name of output')
    args = parser.parse_args()
    data_name = args.data_name
    file_config = args.config
    file_out = args.out

    with open(file_config, 'r') as fp:
        config = yaml.safe_load(fp)

    dir_lcurve = config['l-curve']['dir_out']

    path = os.path.join(dir_lcurve, data_name + '.txt')
    curve = pd.read_fwf(path)
    ind = np.argsort(curve['factor'])

    plt.figure(figsize=(10, 10))
    plt.plot(np.log(curve['f_residual'][ind]),
             np.log(curve['f_reg'][ind]), 'k.-')
    for x, y, a in zip(curve['f_residual'][ind], curve['f_reg'][ind], curve['factor'][ind]):
        plt.annotate('{:9.6f}'.format(a), (np.log(x), np.log(y)))
    plt.xlabel('Residual norm $\log\|Ax - b\|_2$')
    plt.ylabel('Regularized norm $\log\| L (x - x_0)\|_2$')
    plt.title('L-curve')
    plt.tight_layout()

    if file_out:
        plt.savefig(file_out, dpi=300)
    plt.show()

#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse

params = {'axes.labelsize': 14,
          'axes.titlesize': 16,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.fontsize': 14}
plt.rcParams.update(params)


if __name__ == '__main__':
    msg = "plot dispersion curves"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('file_disp', default=None,
                        help='file of dispersion curves')
    parser.add_argument('--approx', default=None, help='file of approximate')
    parser.add_argument('--out', default=None, help=' output figure name')
    args = parser.parse_args()
    file_disp = args.file_disp
    file_approx = args.approx
    file_out = args.out

    disp = np.loadtxt(file_disp)
    modes = set(disp[:, 2].astype(int))

    plt.figure()
    for m in modes:
        d = disp[disp[:, 2] == m]
        plt.plot(d[:, 0], d[:, 1], 'k-')

    if file_approx:
        app = np.loadtxt(file_approx)
        plt.plot(app[:, 0], app[:, 1], 'k.')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase velocity (km/s)')
    plt.tight_layout()

    if file_out:
        plt.savefig(file_out, dpi=300)
    plt.show()

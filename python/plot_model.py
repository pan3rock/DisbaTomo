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
    msg = "plot model"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('file_model', default=None, help='file of the model')
    parser.add_argument('--out', default=None, help=' output figure name')
    args = parser.parse_args()
    file_model = args.file_model
    file_out = args.out

    model = np.loadtxt(file_model)
    z = model[:, 1]
    z = np.append(z, z[-1]*1.2)
    vs = model[:, 3]
    vs = np.append(vs, vs[-1])

    _, ax = plt.subplots()
    ax.step(vs, z, 'k-')
    ax.set_ylim([0.0, z[-1]])
    ax.invert_yaxis()
    ax.set_xlabel('Vs (km/s)')
    ax.set_ylabel('Depth (km)')
    plt.tight_layout()

    if file_out:
        plt.savefig(file_out, dpi=300)
    plt.show()

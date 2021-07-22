#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse
from dpcxx import GradientEval

params = {'axes.labelsize': 14,
          'axes.titlesize': 16,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.fontsize': 14}
plt.rcParams.update(params)


if __name__ == '__main__':
    msg = "plot sensitivity kernel"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('file_model', default=None, help='file of the model')
    parser.add_argument('--disp', required=True,
                        help='file of dispersion curves')
    parser.add_argument('--mode', default=0, type=int,
                        help='mode of dispersion curves')
    parser.add_argument('--out', default=None, help=' output figure name')
    parser.add_argument('--plot_dispersion', action='store_true')
    parser.add_argument('--cmin', type=float, default=None)
    parser.add_argument('--cmax', type=float, default=None)
    parser.add_argument('--fmin', type=float, default=None)
    parser.add_argument('--fmax', type=float, default=None)
    parser.add_argument('--zmax', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None,
                        help='vmax for kernel value')
    parser.add_argument('--love', action='store_true')
    args = parser.parse_args()
    file_model = args.file_model
    file_disp = args.disp
    file_out = args.out
    mode = args.mode
    plot_dispersion = args.plot_dispersion
    cmin = args.cmin
    cmax = args.cmax
    fmin = args.fmin
    fmax = args.fmax
    zmax = args.zmax
    vmax = args.vmax
    love = args.love

    model = np.loadtxt(file_model)
    z = model[:, 1]
    nl = model.shape[0]
    disp = np.loadtxt(file_disp)
    disp = disp[disp[:, 2] == mode, :]
    nf = disp.shape[0]
    freqs = disp[:, 0]
    cs = disp[:, 1]

    if love:
        wave_type = 'love'
    else:
        wave_type = 'rayleigh'

    gradEval = GradientEval(model, wave_type)
    ker = np.zeros([nl, nf])
    for i in range(nf):
        f, c = freqs[i], cs[i]
        ker[:, i] = gradEval.compute(f, c)

    fig, ax = plt.subplots()
    if vmax is None:
        vmax = np.amax(np.abs(ker))
    vmin = -vmax
    print("vmax = ", vmax)
    ax.pcolormesh(freqs,
                  z,
                  ker,
                  cmap='seismic',
                  shading='auto',
                  vmin=vmin,
                  vmax=vmax)

    if zmax is None:
        zmax = z.max()
    if fmin is None:
        fmin = freqs.min()
    if fmax is None:
        fmax = freqs.max()

    ax.set_xlim([fmin, fmax])
    ax.set_ylim([0.0, zmax])

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Depth (km)')
    ax.tick_params('both')
    ax.invert_yaxis()

    if plot_dispersion:
        ax.tick_params('y', colors='r')
        ax.yaxis.label.set_color('r')
        ax2 = ax.twinx()
        ax2.plot(freqs, cs, 'k.-')
        ax2.tick_params('y', colors='k')
        ax2.set_xlim([fmin, fmax])
        if cmin is not None and cmax is not None:
            ax2.set_ylim([cmin, cmax])
        ax2.set_ylabel('Phase velocity (km/s)')

    plt.tight_layout()

    if file_out:
        plt.savefig(file_out, dpi=300)
    plt.show()

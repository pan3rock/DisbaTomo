#!/usr/bin/env python
import numpy as np
import argparse
from disba import PhaseDispersion

import yaml


if __name__ == '__main__':
    msg = 'calculate dispersion curves using disba'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        '-c', '--config', default='config_forward.yml', help='configure file')
    parser.add_argument('--mode', type=int, default=0,
                        help='maximum mode (start from 0)')
    parser.add_argument('--data', help='file of data')
    parser.add_argument('--love', action='store_true',
                        help='whether to compute love waves')
    args = parser.parse_args()
    file_config = args.config
    max_mode = args.mode
    file_data = args.data
    love = args.love

    with open(file_config, 'r') as fp:
        config = yaml.safe_load(fp)

    file_model = config['file_model']
    file_out = config.get('file_out', 'disp.txt')

    model = np.loadtxt(file_model)
    z = model[:, 1]
    tn = np.diff(z)
    tn = np.append(tn, [0, ])
    rho = model[:, 2]
    vs = model[:, 3]
    vp = model[:, 4]
    pd = PhaseDispersion(tn, vp, vs, rho)

    if love:
        wave_type = 'love'
    else:
        wave_type = 'rayleigh'

    disp = dict()
    if file_data:
        data = np.loadtxt(file_data)
        modes = set(data[:, 2].astype(int))
        for mode in modes:
            d = data[data[:, 2].astype(int) == mode, :]
            freqs = np.sort(d[:, 0])
            periods = 1.0 / freqs[::-1]

            cp = pd(periods, mode, wave=wave_type)
            disp[mode] = (1.0 / cp.period[::-1], cp.velocity[::-1])
    else:
        fmin = config['fmin']
        fmax = config['fmax']
        nf = config['nf']
        freqs = np.linspace(fmin, fmax, nf)
        periods = 1.0 / freqs[::-1]

        for mode in range(max_mode + 1):
            cp = pd(periods, mode, wave=wave_type)
            disp[mode] = (1.0 / cp.period[::-1], cp.velocity[::-1])

    with open(file_out, 'w') as fp:
        for mode, (fs, cs) in disp.items():
            for f, c in zip(fs, cs):
                fp.write('{:15.8f}{:15.8f}{:10d}\n'.format(f, c, mode))

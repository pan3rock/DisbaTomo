#!/usr/bin/env python
import numpy as np
import argparse
from disba import PhaseDispersion

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


if __name__ == '__main__':
    msg = 'calculate dispersion curves using disba'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        '-c', '--config', default='config_forward.yml', help='configure file')
    parser.add_argument('--mode', type=int, default=0,
                        help='maximum mode (start from 0)')
    parser.add_argument('--love', action='store_true',
                        help='whether to compute love waves')
    args = parser.parse_args()
    file_config = args.config
    max_mode = args.mode
    love = args.love

    with open(file_config, 'r') as fp:
        config = yaml.load(fp, Loader=Loader)

    fmin = config['fmin']
    fmax = config['fmax']
    nf = config['nf']
    file_model = config['file_model']
    file_out = config.get('file_out', 'disp.txt')

    freqs = np.linspace(fmin, fmax, nf)
    periods = 1.0 / freqs[::-1]

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
    for mode in range(max_mode + 1):
        cp = pd(periods, mode, wave=wave_type)
        disp[mode] = (1.0 / cp.period[::-1], cp.velocity[::-1])

    with open(file_out, 'w') as fp:
        for mode, (fs, cs) in disp.items():
            for f, c in zip(fs, cs):
                fp.write('{:15.8f}{:15.8f}{:10d}\n'.format(f, c, mode))

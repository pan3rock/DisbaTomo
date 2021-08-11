#!/usr/bin/env python
from objective_function import (ObjectiveFunctionDerivativeUsed)
import os
import numpy as np
import pathlib
import time
from scipy.optimize import minimize
import argparse
import shutil

from mpi_master_slave import Master, Slave
from mpi_master_slave import WorkQueue
from mpi4py import MPI

import yaml


class InversionMultiple(object):
    def __init__(self, slaves, config, file_data):
        self.config = config
        self.file_data = file_data
        self.master = Master(slaves)
        self.njobs = len(slaves)
        self.work_queue = WorkQueue(self.master)
        self.options = config.get('option_bfgs', dict())
        self.dir_out = config['l-curve']['dir_out']
        model_init = np.loadtxt(config['model_init'])
        self.nl = model_init.shape[0]

        if os.path.exists(self.dir_out):
            shutil.rmtree(self.dir_out)
        os.makedirs(self.dir_out)

        for fnm in self.file_data:
            path_out = os.path.join(self.dir_out, fnm)
            with open(path_out, 'w') as flog:
                flog.write('{:>15s}{:>15s}{:>15s}{:>15s}\n'.format(
                    'factor', 'f_full', 'f_residual', 'f_reg'))

    def terminate_slaves(self):
        self.master.terminate_slaves()

    def run(self):
        if self.njobs == 1:
            self.options['disp'] = True
        else:
            self.options['disp'] = False

        lcurve = self.config['l-curve']
        amin = lcurve['amin']
        amax = lcurve['amax']
        na = lcurve['na']
        alpha = np.linspace(amin, amax, na)
        factor = 10 ** alpha

        num_data = len(self.file_data)
        if na > num_data:
            outer_loop, inner_loop = self.file_data, factor
            sequence_data_factor = True
        else:
            outer_loop, inner_loop = factor, self.file_data
            sequence_data_factor = False

        for ind_o2, ol2 in enumerate(outer_loop):
            for ind_i2, il2 in enumerate(inner_loop):
                if sequence_data_factor:
                    il, ol = il2, ol2
                    ind_i = ind_i2
                else:
                    il, ol = ol2, il2
                    ind_i = ind_o2
                self.work_queue.add_work(
                    data=(ind_i, self.config, self.nl, ol, il, self.options))

            while not self.work_queue.done():
                self.work_queue.do_work()

                for slave_return_data in self.work_queue.get_completed_work():
                    ind_ai, ind_data, res = slave_return_data
                    path_out = os.path.join(
                        self.dir_out, '{:s}.txt'.format(ind_data))
                    with open(path_out, 'a') as flog:
                        flog.write('{:15.5e}{:15.5e}{:15.5e}{:15.5e}\n'.format(
                            factor[ind_ai], res['f_full'], res['f_residual'], res['f_reg']))
                time.sleep(0.03)


class InversionOne(Slave):
    def __init__(self):
        super(InversionOne, self).__init__()

    def do_work(self, data):
        ind_ai, config, nl, file_data, factor, options = data
        prob = ObjectiveFunctionDerivativeUsed(config, file_data)
        prob.smooth['factor'] = factor
        x0 = np.ones(nl) * 0.5

        def func(x, i):
            return x[i + 1] - x[i]
        const = [{'type': 'ineq', 'fun': func, 'args': (i, )}
                 for i in range(nl - 1)]
        res = minimize(prob.fitness, x0,
                       jac=prob.gradient,
                       constraints=const,
                       method='SLSQP',
                       bounds=prob.bounds,
                       options=options)

        forward = prob.fetch_forward(res.x)
        f_residual = forward.fitness
        f_reg = prob._fitness_regularization(res.x)
        f_full = f_residual + factor * f_reg
        results = dict(f_residual=f_residual, f_reg=f_reg, f_full=f_full)
        ind_data = file_data.split('.')[0]
        return ind_ai, ind_data, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inversion using mpi')
    parser.add_argument('-c', '--config', default='config_inv.yml')
    args = parser.parse_args()
    file_config = args.config

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        with open(file_config, 'r') as fp:
            config = yaml.safe_load(fp)

        dir_data = config['dir_data']
        data_collections = [
            x for x in os.listdir(dir_data) if x.endswith('.txt')
        ]

        process = InversionMultiple(range(1, size), config, data_collections)
        process.run()
        process.terminate_slaves()
    else:
        InversionOne().run()

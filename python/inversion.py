#!/usr/bin/env python
from objective_function import (ObjectiveFunctionDerivativeUsed)
import os
import numpy as np
import pathlib
import time
from scipy.optimize import minimize, LinearConstraint, SR1
import argparse
import shutil

from mpi_master_slave import Master, Slave
from mpi_master_slave import WorkQueue
from mpi4py import MPI

import yaml


class InversionMultiple(object):
    def __init__(self, slaves, config, file_data, num_model):
        self.config = config
        self.file_data = file_data
        self.num_model = num_model
        self.master = Master(slaves)
        self.njobs = len(slaves)
        self.work_queue = WorkQueue(self.master)

        self.file_logging = config['file_logging']
        if os.path.isfile(self.file_logging):
            os.remove(self.file_logging)
        fmt_head = '{:>10s}' + '{:>8s}'*4 + '{:>15s}'*2 + '{:>10s}' + '\n'
        with open(self.file_logging, 'a') as flog:
            flog.write(fmt_head.format('No.data', 'No.m0', 'niter', 'nfev', 'success',
                                       'f(m0)', 'f(m)', 'time(s)'))

        self.init_method = config.get('init_method', 'random')
        self.model_init = config['model_init']
        self.options = config.get('option_bfgs', dict())

    def terminate_slaves(self):
        self.master.terminate_slaves()

    def run(self):
        if self.njobs == 1:
            self.options['disp'] = True
        else:
            self.options['disp'] = False
        xs = self.create_init()

        num_data = len(self.file_data)
        num_model = self.num_model
        if num_model > num_data:
            outer_loop, inner_loop = self.file_data, xs
            sequence_data_x = True
        else:
            outer_loop, inner_loop = xs, self.file_data
            sequence_data_x = False

        for ind_o2, ol2 in enumerate(outer_loop):
            for ind_i2, il2 in enumerate(inner_loop):
                if sequence_data_x:
                    il, ol = il2, ol2
                    ind_i = ind_i2
                else:
                    il, ol = ol2, il2
                    ind_i = ind_o2
                self.work_queue.add_work(
                    data=(ind_i, self.config, ol, il, self.options))

            fmt_line = '{:>10s}' + '{:8d}'*4 + '{:15.5e}'*2 + '{:10d}' + '\n'

            while not self.work_queue.done():
                self.work_queue.do_work()

                for slave_return_data in self.work_queue.get_completed_work():
                    ind_mi, ind_data, res = slave_return_data
                    with open(self.file_logging, 'a') as flog:
                        flog.write(fmt_line.format(ind_data, ind_mi, res['niter'], res['nfev'],
                                                   res['success'], res['f0'],
                                                   res['fi'], res['time']))
                    dir_output = 'inversion/' + ind_data
                    pathlib.Path(dir_output).mkdir(parents=True, exist_ok=True)
                    np.savez(dir_output + '/' +
                             '{:d}.npz'.format(ind_mi), **res)
                time.sleep(0.03)

    def create_init(self):
        model_init = np.loadtxt(self.model_init)
        num_layer = model_init.shape[0]
        num_model = self.num_model
        list_para = []
        if num_model == 1:
            list_para.append(np.ones(num_layer) * 0.5)
        else:
            init_method = self.init_method
            if init_method == 'random':
                for i in range(num_model):
                    list_para.append(np.random.random(num_layer))
            elif init_method == 'ascend':
                d = 1.0 / (num_model - 1)
                for i in range(num_model):
                    list_para.append(i * d * np.ones(num_layer))
            else:
                raise ValueError('invalid init_method in config')
        return list_para


class InversionOne(Slave):
    def __init__(self):
        super(InversionOne, self).__init__()

    def do_work(self, data):
        ind_mi, config, file_data, x0, options = data
        nx = len(x0)

        def func(x, i):
            return x[i + 1] - x[i]
        const = [{'type': 'ineq', 'fun': func, 'args': (i, )}
                 for i in range(nx - 1)]

        t1 = time.time()
        prob = ObjectiveFunctionDerivativeUsed(config, file_data)

        res = minimize(prob.fitness, x0,
                       jac=prob.gradient,
                       constraints=const,
                       method='SLSQP',
                       bounds=prob.bounds,
                       options=options)

        t2 = time.time()
        dt_seconds = int(t2 - t1)
        f0 = prob.fitness(x0)
        m0 = prob._update_model(x0)
        mi = prob._update_model(res.x)
        ri = prob.fetch_forward(res.x).disp
        r0 = prob.fetch_forward(x0).disp
        results = dict(niter=res.nit, nfev=res.nfev, success=res.success,
                       fi=res.fun, f0=f0, m0=m0, mi=mi, time=dt_seconds,
                       ri=ri, r0=r0)
        ind_data = file_data.split('.')[0]
        return ind_mi, ind_data, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inversion using mpi')
    parser.add_argument('-c', '--config', default='config_inv.yml')
    parser.add_argument('--num_init', type=int, default=1,
                        help='number of initial model')
    args = parser.parse_args()
    file_config = args.config
    num_init = args.num_init

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        with open(file_config, 'r') as fp:
            config = yaml.safe_load(fp)
        dir_output = config['dir_output']
        if os.path.exists(dir_output):
            shutil.rmtree(dir_output)
        os.makedirs(dir_output)

        dir_data = config['dir_data']
        data_collections = [
            x for x in os.listdir(dir_data) if x.endswith('.txt')
        ]

        process = InversionMultiple(range(1, size), config, data_collections,
                                    num_init)
        process.run()
        process.terminate_slaves()
    else:
        InversionOne().run()

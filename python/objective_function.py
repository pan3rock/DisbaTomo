import numpy as np
import os
import time
from collections import deque
from empirical_relation import (brocher, gardner, user_defined)
from dpcxx import GradientEval
from disba import PhaseDispersion


class ObjectiveFunctionDerivativeFree:
    def __init__(self, config, file_data):
        self._init_config(config, file_data)
        self._init_memvar()
        self._init_model()
        self._init_data()
        self.icov_m = self._get_model_convariance_inv()

    def _init_config(self, config, file_data):
        self.weights_mode = config.get("weights_mode", None)
        self.norm_damping = config.get("norm_damping", 0.0)
        self.derivative_damping = config.get("derivative_damping", 0.0)
        self.empirical_relation = config.get("empirical_relation", None)
        self.epsilon = config.get("epsilon", 1.0e-8)
        self.dir_output = config.get("dir_output", "inversion")

        self.smooth = config['smooth']

        dir_data = config["dir_data"]
        self.file_data = os.path.join(dir_data, file_data)
        self.file_model_init = config["model_init"]
        self.half_width = config["init_half_width"]
        self.inv_half_width = config["inv_half_width"]
        self.wave_type = config.get("wave_type", "rayleigh")

    def _init_memvar(self):
        # count
        self.count_fitness = 0
        self.count_forward = 0
        self.count_iter = 0

        # timing
        self.time_start = time.time()

        # cache for roots
        self.forward_cache = deque(maxlen=10)

    def _init_data(self):
        data = np.loadtxt(self.file_data)
        self.data = data[(-data[:, 0]).argsort()]
        modes = set(self.data[:, 2].astype(int))
        if self.weights_mode:
            tmp = self.weights_mode.copy()
            self.weights_mode = dict()
            for mode in modes:
                if mode not in tmp.keys():
                    continue
                self.weights_mode[mode] = tmp[mode]
        else:
            for mode in modes:
                self.weights_mode[mode] = 1.0

    def _init_model(self):
        model_init = np.loadtxt(self.file_model_init)
        num_layer = model_init.shape[0]
        self.model_init = model_init
        self.z = model_init[:, 1]
        self.num_layer = num_layer

        vs0 = model_init[:, 3]
        self.vsmin = vs0 - self.half_width
        self.vsmax = vs0 + self.half_width
        num_para = num_layer

        self.num_para = num_para
        self.lb = np.zeros(num_para)
        self.ub = np.ones(num_para)

    def _update_model(self, x_in):
        x = x_in.flatten()
        vs = self.vsmin + (self.vsmax - self.vsmin) * x
        if self.empirical_relation == 'brocher':
            model = brocher(self.z, vs)
        elif self.empirical_relation == 'gardner':
            model = gardner(self.z, vs)
        elif self.empirical_relation == 'user-defined':
            model = user_defined(self.z, vs)
        else:
            model = self.model_init
            model[:, 3] = vs
        return model

    def fetch_forward(self, x):
        model = self._update_model(x)

        for forward_iter in reversed(self.forward_cache):
            if np.linalg.norm(forward_iter.model - model) < self.epsilon:
                forward = forward_iter
                break
        else:
            forward = Forward(model, self.data, self.wave_type)
            forward.compute(self.weights_mode)
            self.forward_cache.append(forward)
            self.count_forward += 1
        return forward

    def _get_smoothing_distance(self, z):
        smooth = self.smooth
        if z < smooth['zmin']:
            sd = smooth['dmin']
        elif z < smooth['zmax']:
            sd = (smooth['dmax'] - smooth['dmin']) / \
                (smooth['zmax'] - smooth['zmin']) * (z - smooth['zmin']) \
                + smooth['dmin']
        else:
            sd = smooth['dmax']
        return sd

    def _get_model_convariance_inv(self):
        z = self.z
        cov_m = np.zeros([self.num_layer, self.num_layer])
        for i in range(self.num_layer):
            for j in range(self.num_layer):
                sd = self._get_smoothing_distance((z[i] + z[j]) / 2.0)
                cov_m[i, j] = np.exp(-abs(z[i] - z[j]) / sd)
        return np.linalg.inv(cov_m)

    def _derivative_freg(self, x):
        model = self._update_model(x)
        vs = model[:, 3]
        vs = vs.reshape((-1, 1))

        nl = len(vs)
        matL = np.zeros((nl-2, nl))
        for i in range(nl - 2):
            matL[i, i] = -0.25
            matL[i, i+1] = 0.5
            matL[i, i+2] = -0.25

        lap = matL@vs
        ret = 0.5 * np.sum(lap ** 2) / (nl - 2)
        return ret

    def _fitness_regularization2(self, x):
        ret = 0.
        # norm damping ignored
        vs1 = self._update_model(x)[:, 3]
        vs0 = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        self.f_normreg = 0.5 * np.sum((vs1 - vs0)**2) / len(x)
        ret += self.norm_damping * self.f_normreg
        # derivative damping
        self.f_dervreg = self._derivative_freg(x)
        ret += self.derivative_damping * self.f_dervreg
        return ret

    def _fitness_regularization(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))
        ret = self.smooth['factor'] * \
            (vs - vs_ref).T @ self.icov_m @ (vs - vs_ref) / len(x)
        return ret[0, 0]

    def fitness(self, x):
        forward = self.fetch_forward(x)
        ret = forward.fitness + self._fitness_regularization(x)
        self.fitness_current = ret
        if self.count_fitness == 0:
            self.fitness_init = ret
        self.count_fitness += 1
        return ret


class ObjectiveFunctionDerivativeUsed(ObjectiveFunctionDerivativeFree):
    def __init__(self, config, file_data):
        self._init_config(config, file_data)
        self._init_memvar()
        self._init_model()
        self._init_data()
        self.icov_m = self._get_model_convariance_inv()

    def _init_model(self, model_init=None):
        if not model_init:
            model_init = np.loadtxt(self.file_model_init)
        num_layer = model_init.shape[0]
        self.model_init = model_init
        self.z = model_init[:, 1]
        self.num_layer = num_layer
        vs0 = model_init[:, 3]
        self.vsmin = vs0 - self.half_width
        self.vsmax = vs0 + self.half_width
        lb = (self.half_width - self.inv_half_width) / (2.0 * self.half_width)
        ub = (self.half_width + self.inv_half_width) / (2.0 * self.half_width)
        self.bounds = [(lb, ub), ] * num_layer
        self.x0 = np.ones(num_layer) * 0.5

    def _derivative_greg(self, x):
        model = self._update_model(x)
        vs = model[:, 3]
        vs = vs.reshape((-1, 1))

        nl = len(vs)
        matL = np.zeros((nl-2, nl))
        for i in range(nl - 2):
            matL[i, i] = -0.25
            matL[i, i+1] = 0.5
            matL[i, i+2] = -0.25

        ret = (matL @ vs).T @ matL / (nl - 2)
        return ret.ravel()

    def _gradient_regularization2(self, x):
        ret = 0.
        # norm damping ignored
        vs1 = self._update_model(x)[:, 3]
        vs0 = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        self.g_normreg = (vs1 - vs0) / len(x)
        ret += self.norm_damping * self.g_normreg
        # derivative damping
        self.g_dervreg = self._derivative_greg(x)
        ret += self.derivative_damping * self.g_dervreg
        return ret

    def _gradient_regularization(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))
        ret = self.smooth['factor'] * self.icov_m @ (vs - vs_ref) / len(x)
        return ret.flatten()

    def gradient(self, x):
        forward = self.fetch_forward(x)
        grad = forward.gradient
        grad += self._gradient_regularization(x)
        grad *= self.vsmax - self.vsmin
        return grad

    def callback(self, x, dir_output=None, ind_count=0, x0=None):
        if not dir_output:
            dir_output = self.dir_output
        if x0 is None:
            x0 = self.x0
        model = self._update_model(x)
        file_model = os.path.join(dir_output,
                                  'model{:d}.txt'.format(self.count_iter+1))
        np.savetxt(file_model, model, fmt="%5d%12.6f%12.4f%12.4f%12.4f")
        file_roots = os.path.join(dir_output,
                                  'roots{:d}.npy'.format(self.count_iter+1))
        roots = self.fetch_roots(x)
        np.save(file_roots, roots.value)

        # save the initial model and its corresponding dispersion curves
        if self.count_iter == 0:
            model = self._update_model(x0)
            file_model = os.path.join(dir_output, 'model0.txt')
            np.savetxt(file_model, model, fmt="%5d%12.6f%12.4f%12.4f%12.4f")
            file_roots = os.path.join(dir_output, 'roots0.npy')
            roots = self.fetch_roots(x0)
            np.save(file_roots, roots.value)

        self.count_iter += 1


class Forward:
    def __init__(self, model, data, wave_type):
        self.model = model
        self.nl = model.shape[0]
        self.data = data
        self.wave_type = wave_type
        self.gradEval = GradientEval(model, wave_type)

        z = model[:, 1]
        tn = np.diff(z)
        tn = np.append(tn, [0, ])
        rho = model[:, 2]
        vs = model[:, 3]
        vp = model[:, 4]
        self.pd = PhaseDispersion(tn, vp, vs, rho)

    def compute(self, weights_mode):
        self.fitness = 0.0
        self.gradient = np.zeros(self.nl)
        self.disp = dict()
        for mode, weight in weights_mode.items():
            disp = self.data[self.data[:, 2].astype(int) == mode]
            periods = 1.0 / disp[:, 0]
            cp = self.pd(periods, mode=mode, wave=self.wave_type)
            self.disp[mode] = np.vstack(
                [1.0 / cp.period[::-1], cp.velocity[::-1]]).T
            count_1mode = len(cp.period)
            f_1mode = 0.0
            grad_1mode = np.zeros(self.nl)
            for p, c in zip(cp.period, cp.velocity):
                ind = np.argmin(np.abs(periods - p))
                c_obs = disp[ind, 1]
                f_1mode += (c - c_obs) ** 2
                grad_1mode += 2.0 * (c - c_obs) * \
                    self.gradEval.compute(1.0 / p, c)
            if count_1mode > 0:
                f_1mode *= weight / count_1mode
                grad_1mode *= weight / count_1mode
            self.fitness += f_1mode
            self.gradient += grad_1mode

        self.fitness /= len(weights_mode)
        self.gradient /= len(weights_mode)

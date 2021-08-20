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

        self.reg_method = config.get("reg_method", "exp")
        if self.reg_method == "exp":
            self._fitness_regularization = self._freg_exp
        elif self.reg_method == "tr0":
            self._fitness_regularization = self._freg_tr0
        elif self.reg_method == "tr1":
            self._fitness_regularization = self._freg_tr1
        elif self.reg_method == "tr2":
            self._fitness_regularization = self._freg_tr2

    def _init_memvar(self):
        # count
        self.count_fitness = 0
        self.count_forward = 0

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
            model = self.model_init.copy()
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

    def _freg_tr0(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        return np.sum((vs - vs_ref) ** 2) / len(x)

    def _freg_tr1(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))
        nl = len(vs)
        matL = np.zeros((nl - 1, nl))
        for i in range(nl - 1):
            matL[i, i] = -0.5
            matL[i, i+1] = 0.5

        lap = matL@ (vs - vs_ref)
        ret = 0.5 * np.sum(lap ** 2) / (nl - 1)
        return ret

    def _freg_tr2(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))
        nl = len(vs)
        matL = np.zeros((nl - 2, nl))
        for i in range(nl - 2):
            matL[i, i] = -0.25
            matL[i, i+1] = 0.5
            matL[i, i+2] = -0.25

        lap = matL@ (vs - vs_ref)
        ret = 0.5 * np.sum(lap ** 2) / (nl - 2)
        return ret

    def _freg_exp(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))
        ret = (vs - vs_ref).T @ self.icov_m @ (vs - vs_ref) / len(x)
        return ret[0, 0]

    def fitness(self, x):
        forward = self.fetch_forward(x)
        ret = forward.fitness + \
            self.smooth['factor'] / forward.mean_dstd * \
            self._fitness_regularization(x)
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
        if self.reg_method == "exp":
            self._gradient_regularization = self._greg_exp
        elif self.reg_method == "tr0":
            self._gradient_regularization = self._greg_tr0
        elif self.reg_method == "tr1":
            self._gradient_regularization = self._greg_tr1
        elif self.reg_method == "tr2":
            self._gradient_regularization = self._greg_tr2

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

    def _greg_tr0(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        return vs - vs_ref

    def _greg_tr1(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))

        nl = len(vs)
        matL = np.zeros((nl - 1, nl))
        for i in range(nl - 1):
            matL[i, i] = -0.5
            matL[i, i + 1] = 0.5

        ret = (matL @ (vs - vs_ref)).T @ matL / (nl - 1)
        return ret.ravel()

    def _greg_tr2(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))

        nl = len(vs)
        matL = np.zeros((nl - 2, nl))
        for i in range(nl - 2):
            matL[i, i] = -0.25
            matL[i, i + 1] = 0.5
            matL[i, i + 2] = -0.25

        ret = (matL @ (vs - vs_ref)).T @ matL / (nl - 2)
        return ret.ravel()

    def _greg_exp(self, x):
        vs = self._update_model(x)[:, 3]
        vs_ref = self._update_model(np.ones_like(x) * 0.5)[:, 3]
        vs = vs.reshape((-1, 1))
        vs_ref = vs_ref.reshape((-1, 1))
        ret = self.icov_m @ (vs - vs_ref) / len(x)
        return ret.flatten()

    def gradient(self, x):
        forward = self.fetch_forward(x)
        grad = forward.gradient
        grad += self.smooth['factor'] / forward.mean_dstd * \
            self._gradient_regularization(x)
        grad *= self.vsmax - self.vsmin
        return grad


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
        self.mean_dstd = []
        for mode, weight in weights_mode.items():
            disp = self.data[self.data[:, 2].astype(int) == mode]
            if self.data.shape[1] == 4:
                dstd = disp[:, 3]
            else:
                dstd = np.ones(disp.shape[0])
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
                f_1mode += (c - c_obs) ** 2 / dstd[ind] ** 2
                grad_1mode += 2.0 * (c - c_obs) * \
                    self.gradEval.compute(1.0 / p, c) / dstd[ind] ** 2
                self.mean_dstd.append(dstd[ind])
            if count_1mode > 0:
                f_1mode *= weight / count_1mode
                grad_1mode *= weight / count_1mode
            self.fitness += f_1mode
            self.gradient += grad_1mode

        self.mean_dstd = np.mean(self.mean_dstd) ** 2
        self.fitness /= len(weights_mode)
        self.gradient /= len(weights_mode)

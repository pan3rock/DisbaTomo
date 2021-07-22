import numpy as np


def brocher(z, vs):
    model = np.zeros([len(z), 5])
    for i in range(len(vs)):
        vp = (0.9409 + 2.0947 * vs[i] - 0.8206 * vs[i]**2 +
              0.2683 * vs[i]**3 - 0.0251 * vs[i]**4)
        rho = (1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3
               - 0.0043 * vp**4 + 0.000106 * vp**5)
        model[i, 2] = rho
        model[i, 4] = vp
    model[:, 0] = np.arange(len(z)) + 1.0
    model[:, 1] = z
    model[:, 3] = vs
    return model


def gardner(z, vs):
    model = np.zeros([len(z), 5])

    vp_vs_ratio = 1.7321
    vp = vp_vs_ratio * vs
    rho = 1.741 * vp ** 0.25
    model[:, 0] = np.arange(len(z)) + 1.0
    model[:, 1] = z
    model[:, 2] = rho
    model[:, 3] = vs
    model[:, 4] = vp
    return model


def user_defined(z, vs):
    pass

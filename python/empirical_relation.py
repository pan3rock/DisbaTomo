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

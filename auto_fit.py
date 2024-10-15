import numpy
import scipy.optimize as opt
import autograd.numpy as np
from autograd import elementwise_grad as egrad


def nll(residuals, sigma2):
    return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + (residuals ** 2) / sigma2)


def residuals_and_variance(func, params, x, y):
    residuals = y - func(x, *params)
    sigma2 = np.var(residuals)
    return residuals, sigma2


def objective(params, func, x, y, lamda):
    func_grad = egrad(func, 0)
    res, var2 = residuals_and_variance(func, params, x, y)
    res_grad, var2_grad = residuals_and_variance(func_grad, params, x, np.gradient(y, x))
    return nll(res, var2) - 1 * lamda * nll(res_grad, var2_grad)


def fit_with_derivative(func, x, y, p0, lamda=10 ** -4, max_iter=1000):
    options = {'disp': True, 'return_all': True, 'maxiter': max_iter}

    result = opt.minimize(
        objective,
        p0,
        args=(func, x, y, lamda),
        method='BFGS',
        options=options
    )

    if isinstance(result.hess_inv, np.ndarray):
        hessian_inv = result.hess_inv
    else:
        hessian_inv = result.hess_inv.todense()

    parameter_errors = np.sqrt(np.diag(hessian_inv))

    return result.x, parameter_errors
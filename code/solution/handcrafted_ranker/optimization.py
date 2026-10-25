"""Training-only block L2; inference remains a single NumPy linear scorer."""
import numpy as np
from scipy.optimize import minimize

from solution.group_ranker.features import FEATURE_DIM, standardize
from solution.group_ranker.train_linear import (
    fit_linear as fit_uniform_linear, fit_standardizer, objective as uniform_objective,
)


def penalty_vector(l2, handcrafted_multiplier=1.0):
    if not np.isfinite(l2) or l2 <= 0 or not np.isfinite(handcrafted_multiplier) or handcrafted_multiplier <= 0:
        raise ValueError("L2 and handcrafted multiplier must be finite and positive")
    penalty = np.full(FEATURE_DIM, l2, dtype=np.float64)
    penalty[2064:2074] *= handcrafted_multiplier
    return penalty


def block_objective(weight, x, mask, labels, penalty):
    loss, gradient = uniform_objective(weight, x, mask, labels, 0.0)
    return loss + np.dot(penalty, weight * weight) / 2, gradient + penalty * weight


def fit_linear(features, mask, labels, l2, maxiter, handcrafted_multiplier=1.0):
    penalty = penalty_vector(l2, handcrafted_multiplier)
    if handcrafted_multiplier == 1.0:
        # Preserve old weights and numerical behavior exactly.
        return fit_uniform_linear(features, mask, labels, l2, maxiter)
    model = fit_standardizer(features, mask)
    x = standardize(features, **model).astype(np.float64).reshape(-1, FEATURE_DIM)
    result = minimize(
        block_objective, np.zeros(FEATURE_DIM),
        args=(x, mask, labels, penalty), method="L-BFGS-B", jac=True,
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("Nonfinite block-regularized optimizer result")
    model["weight"] = result.x.astype(np.float32)
    return model, {"converged": bool(result.success), "iterations": int(result.nit),
                   "objective": float(result.fun), "message": str(result.message)}

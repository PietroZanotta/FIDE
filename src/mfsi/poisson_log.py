"""Unfloored weighted-Poisson solvers driven by log cell masses.

The primary solver preserves the existing arithmetic-face finite-volume
discretization and no-flux boundary.  It equilibrates every equation by its
physical diagonal before coefficients are exponentiated.  If even those local
ratios exceed float64's representable range, the solve is rejected rather than
silently deleting a positive conductance.

The divided-log-density solver is deliberately independent: it discretizes
``-(Delta psi + grad(log q) dot grad psi) = -h`` directly, using ``log q`` for
the drift.  Every candidate is still audited against the original unfloored
finite-volume equation in log arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, spilu, splu


_LOG_FLOAT64_TINY = math.log(np.nextafter(0.0, 1.0))


@dataclass(frozen=True)
class LogPoissonConfig:
    dx: float
    iterative_relative_tolerance: float = 1.0e-8
    physical_relative_tolerance: float = 1.0e-6
    gauge_absolute_tolerance: float = 1.0e-6
    maximum_iterations: int = 1200
    gmres_restart: int = 80
    ilu_drop_tolerance: float = 1.0e-5
    ilu_fill_factor: float = 8.0
    direct_maximum_cells: int = 40_000
    iterative_solver: str = "bicgstab"


def log_face_conductances(
    log_q_mass: np.ndarray, dx: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arithmetic-face log conductances and log physical diagonals."""
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    if log_q.ndim != 2 or not np.isfinite(log_q).all():
        raise ValueError("log_q_mass must be a finite two-dimensional array")
    log_geometry = -2.0 * math.log(float(dx))
    log_x = np.logaddexp(log_q[:, :-1], log_q[:, 1:]) - math.log(2.0) + log_geometry
    log_y = np.logaddexp(log_q[:-1, :], log_q[1:, :]) - math.log(2.0) + log_geometry
    log_diagonal = np.full_like(log_q, -np.inf)
    log_diagonal[:, :-1] = np.logaddexp(log_diagonal[:, :-1], log_x)
    log_diagonal[:, 1:] = np.logaddexp(log_diagonal[:, 1:], log_x)
    log_diagonal[:-1, :] = np.logaddexp(log_diagonal[:-1, :], log_y)
    log_diagonal[1:, :] = np.logaddexp(log_diagonal[1:, :], log_y)
    return log_x, log_y, log_diagonal


def _weighted_mean(log_q_mass: np.ndarray, values: np.ndarray) -> float:
    log_abs, sign = logsumexp(
        np.asarray(log_q_mass).ravel(),
        b=np.asarray(values, dtype=np.float64).ravel(),
        return_sign=True,
    )
    normalization = float(logsumexp(np.asarray(log_q_mass).ravel()))
    if sign == 0.0:
        return 0.0
    return float(sign * math.exp(log_abs - normalization))


def _log_l2(log_absolute_values: np.ndarray) -> float:
    finite = np.isfinite(log_absolute_values)
    if not np.any(finite):
        return -math.inf
    return float(0.5 * logsumexp(2.0 * log_absolute_values[finite]))


def physical_log_conductance_diagnostics(
    log_q_mass: np.ndarray,
    h: np.ndarray,
    potential: np.ndarray,
    dx: float,
) -> dict[str, float]:
    """Audit the original unfloored finite-volume equation in log arithmetic."""
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    forcing = np.asarray(h, dtype=np.float64)
    psi = np.asarray(potential, dtype=np.float64)
    if forcing.shape != log_q.shape or psi.shape != log_q.shape:
        raise ValueError("log_q_mass, h, and potential must have the same shape")
    log_x, log_y, log_diagonal = log_face_conductances(log_q, dx)

    # Each row has at most five signed terms.  Signed log-sum-exp evaluates the
    # original physical residual even when no single global or row scaling can
    # represent every conductance simultaneously.
    term_logs = np.full((5, *psi.shape), -np.inf, dtype=np.float64)
    term_signs = np.zeros((5, *psi.shape), dtype=np.float64)
    difference_x = psi[:, :-1] - psi[:, 1:]
    difference_y = psi[:-1, :] - psi[1:, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_terms = log_x + np.where(
            difference_x != 0.0, np.log(np.abs(difference_x)), -np.inf
        )
        y_terms = log_y + np.where(
            difference_y != 0.0, np.log(np.abs(difference_y)), -np.inf
        )
        rhs_terms = log_q + np.where(
            forcing != 0.0, np.log(np.abs(forcing)), -np.inf
        )
    term_logs[0, :, :-1] = x_terms
    term_signs[0, :, :-1] = np.sign(difference_x)
    term_logs[1, :, 1:] = x_terms
    term_signs[1, :, 1:] = -np.sign(difference_x)
    term_logs[2, :-1, :] = y_terms
    term_signs[2, :-1, :] = np.sign(difference_y)
    term_logs[3, 1:, :] = y_terms
    term_signs[3, 1:, :] = -np.sign(difference_y)
    term_logs[4] = rhs_terms
    term_signs[4] = np.sign(forcing)
    log_abs_residual, residual_sign = logsumexp(
        term_logs, b=term_signs, axis=0, return_sign=True
    )
    log_abs_residual = np.where(residual_sign != 0.0, log_abs_residual, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_abs_rhs = np.where(
            forcing != 0.0,
            np.log(np.abs(forcing)) + log_q,
            -np.inf,
        )
    log_residual_norm = _log_l2(log_abs_residual.ravel())
    log_rhs_norm = _log_l2(log_abs_rhs.ravel())
    relative = (
        math.exp(log_residual_norm - log_rhs_norm)
        if np.isfinite(log_residual_norm) and np.isfinite(log_rhs_norm)
        else (0.0 if log_residual_norm == -math.inf else math.inf)
    )

    with np.errstate(divide="ignore"):
        log_energy_x = np.where(
            difference_x != 0.0, log_x + 2.0 * np.log(np.abs(difference_x)), -np.inf
        )
        log_energy_y = np.where(
            difference_y != 0.0, log_y + 2.0 * np.log(np.abs(difference_y)), -np.inf
        )
    log_action = float(logsumexp(np.r_[log_energy_x.ravel(), log_energy_y.ravel()]))
    action = math.exp(log_action) if log_action < math.log(np.finfo(float).max) else math.inf
    absolute = (
        math.exp(log_residual_norm)
        if log_residual_norm < math.log(np.finfo(float).max) else math.inf
    )
    return {
        "physical_relative_residual": relative,
        "physical_absolute_residual": absolute,
        "physical_log10_absolute_residual": log_residual_norm / math.log(10.0),
        "weighted_mean_potential": _weighted_mean(log_q, psi),
        "action": action,
        "log_action": log_action,
    }


def _local_dynamic_range(
    log_x: np.ndarray, log_y: np.ndarray, log_diagonal: np.ndarray, log_q: np.ndarray
) -> dict[str, Any]:
    directed = np.r_[
        (log_x - log_diagonal[:, :-1]).ravel(),
        (log_x - log_diagonal[:, 1:]).ravel(),
        (log_y - log_diagonal[:-1, :]).ravel(),
        (log_y - log_diagonal[1:, :]).ravel(),
        (log_q - log_diagonal).ravel(),
    ]
    unrepresentable = directed < _LOG_FLOAT64_TINY
    minimum = float(np.min(directed))
    return {
        "minimum_equilibrated_log_coefficient": minimum,
        "maximum_local_log_dynamic_range": -minimum,
        "unrepresentable_equilibrated_coefficient_count": int(np.sum(unrepresentable)),
        "unrepresentable_equilibrated_coefficient_fraction": float(np.mean(unrepresentable)),
        "log10_condition_proxy": -minimum / math.log(10.0),
    }


def _row_equilibrated_matrix(
    log_x: np.ndarray, log_y: np.ndarray, log_diagonal: np.ndarray
) -> sparse.csr_matrix:
    ny, nx = log_diagonal.shape
    count = nx * ny
    indices = np.arange(count).reshape((ny, nx))
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for log_edge, left, right, left_diag, right_diag in (
        (log_x, indices[:, :-1], indices[:, 1:], log_diagonal[:, :-1], log_diagonal[:, 1:]),
        (log_y, indices[:-1, :], indices[1:, :], log_diagonal[:-1, :], log_diagonal[1:, :]),
    ):
        rows.extend((left.ravel(), right.ravel()))
        columns.extend((right.ravel(), left.ravel()))
        values.extend((
            -np.exp((log_edge - left_diag).ravel()),
            -np.exp((log_edge - right_diag).ravel()),
        ))
    rows.append(np.arange(count))
    columns.append(np.arange(count))
    values.append(np.ones(count))
    return sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(count, count),
    ).tocsr()


def _pin_and_solve(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    pin: int,
    cfg: LogPoissonConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    system = matrix.tolil(copy=True)
    system.rows[pin] = [pin]
    system.data[pin] = [1.0]
    vector = np.asarray(rhs, dtype=np.float64).copy()
    vector[pin] = 0.0
    system = system.tocsr()
    preconditioner = "none"
    iterations = 0
    direct = False
    error = ""
    try:
        if system.shape[0] <= int(cfg.direct_maximum_cells):
            factor = splu(system.tocsc())
            solution = factor.solve(vector)
            info = 0
            direct = True
            preconditioner = "SuperLU sparse direct"
        else:
            ilu = spilu(
                system.tocsc(),
                drop_tol=float(cfg.ilu_drop_tolerance),
                fill_factor=float(cfg.ilu_fill_factor),
            )
            preconditioner = (
                f"SuperLU ILU(drop={cfg.ilu_drop_tolerance:g},fill={cfg.ilu_fill_factor:g})"
            )
            operator = LinearOperator(system.shape, matvec=ilu.solve, dtype=np.float64)

            def callback(_: Any) -> None:
                nonlocal iterations
                iterations += 1

            if cfg.iterative_solver == "bicgstab":
                solution, info = bicgstab(
                    system,
                    vector,
                    M=operator,
                    rtol=float(cfg.iterative_relative_tolerance),
                    atol=0.0,
                    maxiter=int(cfg.maximum_iterations),
                    callback=callback,
                )
            elif cfg.iterative_solver == "gmres":
                solution, info = gmres(
                    system,
                    vector,
                    M=operator,
                    rtol=float(cfg.iterative_relative_tolerance),
                    atol=0.0,
                    restart=int(cfg.gmres_restart),
                    maxiter=max(1, int(math.ceil(cfg.maximum_iterations / cfg.gmres_restart))),
                    callback=callback,
                    callback_type="pr_norm",
                )
            else:
                raise ValueError(f"unknown iterative solver {cfg.iterative_solver!r}")
    except Exception as exc:
        solution = np.full(len(vector), np.nan)
        info = -1
        error = f"{type(exc).__name__}: {exc}"
    if np.isfinite(solution).all():
        scaled_relative = float(
            np.linalg.norm(system @ solution - vector) / max(np.linalg.norm(vector), 1e-300)
        )
    else:
        scaled_relative = math.inf
    return solution, {
        "linear_solver_info": int(info),
        "iteration_count": int(iterations),
        "preconditioner": preconditioner,
        "direct_solve": direct,
        "scaled_relative_residual": scaled_relative,
        "solver_error": error,
    }


def solve_log_conductance_poisson(
    log_q_mass: np.ndarray, h: np.ndarray, cfg: LogPoissonConfig
) -> dict[str, Any]:
    """Solve the unchanged FV equation after exact local log equilibration."""
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    forcing = np.asarray(h, dtype=np.float64)
    log_x, log_y, log_diagonal = log_face_conductances(log_q, cfg.dx)
    dynamic = _local_dynamic_range(log_x, log_y, log_diagonal, log_q)
    base: dict[str, Any] = {
        "formulation": "equilibrated_log_conductance_finite_volume",
        "solver": "row-equilibrated sparse solve",
        "operator_floor": 0.0,
        "density_modified": False,
        **dynamic,
    }
    if dynamic["unrepresentable_equilibrated_coefficient_count"]:
        return {
            **base,
            "converged": False,
            "physical_residual_valid": False,
            "potential": None,
            "action": math.nan,
            "physical_relative_residual": math.inf,
            "physical_absolute_residual": math.inf,
            "physical_log10_absolute_residual": math.inf,
            "weighted_mean_potential": math.nan,
            "iteration_count": 0,
            "preconditioner": "not attempted",
            "scaled_relative_residual": math.inf,
            "solver_error": (
                "positive locally equilibrated conductances/RHS exceed float64 exponent range; "
                "solve rejected rather than thresholding them"
            ),
        }
    matrix = _row_equilibrated_matrix(log_x, log_y, log_diagonal)
    rhs = (-np.exp(log_q - log_diagonal) * forcing).ravel()
    pin = int(np.argmax(log_q))
    solution, linear = _pin_and_solve(matrix, rhs, pin, cfg)
    if np.isfinite(solution).all():
        solution -= _weighted_mean(log_q, solution)
        potential = solution.reshape(log_q.shape)
        physical = physical_log_conductance_diagnostics(log_q, forcing, potential, cfg.dx)
    else:
        potential = None
        physical = {
            "physical_relative_residual": math.inf,
            "physical_absolute_residual": math.inf,
            "physical_log10_absolute_residual": math.inf,
            "weighted_mean_potential": math.nan,
            "action": math.nan,
            "log_action": math.nan,
        }
    converged = bool(
        linear["linear_solver_info"] == 0
        and linear["scaled_relative_residual"] <= 10.0 * cfg.iterative_relative_tolerance
    )
    physical_valid = bool(
        converged
        and physical["physical_relative_residual"] <= cfg.physical_relative_tolerance
        and abs(physical["weighted_mean_potential"]) <= cfg.gauge_absolute_tolerance
    )
    return {
        **base,
        **linear,
        **physical,
        "converged": converged,
        "physical_residual_valid": physical_valid,
        "potential": potential,
    }


def _divided_log_density_matrix(
    log_q: np.ndarray, dx: float
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Central finite differences for ``-Delta-grad(log q).grad`` with no flux."""
    ny, nx = log_q.shape
    count = nx * ny
    indices = np.arange(count).reshape((ny, nx))
    grad_y, grad_x = np.gradient(log_q, dx, dx, edge_order=2)
    # The physical boundary condition is normal derivative zero.  Normal drift
    # therefore contributes no boundary-face flux.
    grad_x[:, 0] = 0.0
    grad_x[:, -1] = 0.0
    grad_y[0, :] = 0.0
    grad_y[-1, :] = 0.0
    inverse_dx2 = 1.0 / (dx * dx)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    diagonal = np.zeros_like(log_q)

    left = indices[:, :-1]
    right = indices[:, 1:]
    rows.extend((left.ravel(), right.ravel()))
    columns.extend((right.ravel(), left.ravel()))
    values.extend((
        (-inverse_dx2 - grad_x[:, :-1] / (2.0 * dx)).ravel(),
        (-inverse_dx2 + grad_x[:, 1:] / (2.0 * dx)).ravel(),
    ))
    diagonal[:, :-1] += inverse_dx2
    diagonal[:, 1:] += inverse_dx2

    lower = indices[:-1, :]
    upper = indices[1:, :]
    rows.extend((lower.ravel(), upper.ravel()))
    columns.extend((upper.ravel(), lower.ravel()))
    values.extend((
        (-inverse_dx2 - grad_y[:-1, :] / (2.0 * dx)).ravel(),
        (-inverse_dx2 + grad_y[1:, :] / (2.0 * dx)).ravel(),
    ))
    diagonal[:-1, :] += inverse_dx2
    diagonal[1:, :] += inverse_dx2

    rows.append(np.arange(count))
    columns.append(np.arange(count))
    values.append(diagonal.ravel())
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(count, count),
    ).tocsr()
    row_scale = np.asarray(np.abs(matrix).max(axis=1).toarray()).ravel()
    return (
        sparse.diags(1.0 / np.maximum(row_scale, np.finfo(float).tiny)) @ matrix,
        row_scale,
    )


def solve_divided_log_density_poisson(
    log_q_mass: np.ndarray, h: np.ndarray, cfg: LogPoissonConfig
) -> dict[str, Any]:
    """Solve the independent divided-by-q/log-density finite difference form."""
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    forcing = np.asarray(h, dtype=np.float64)
    matrix, row_scale = _divided_log_density_matrix(log_q, cfg.dx)
    rhs = -forcing.ravel() / row_scale
    pin = int(np.argmax(log_q))
    solution, linear = _pin_and_solve(matrix, rhs, pin, cfg)
    if np.isfinite(solution).all():
        solution -= _weighted_mean(log_q, solution)
        potential = solution.reshape(log_q.shape)
        physical = physical_log_conductance_diagnostics(log_q, forcing, potential, cfg.dx)
    else:
        potential = None
        physical = {
            "physical_relative_residual": math.inf,
            "physical_absolute_residual": math.inf,
            "physical_log10_absolute_residual": math.inf,
            "weighted_mean_potential": math.nan,
            "action": math.nan,
            "log_action": math.nan,
        }
    converged = bool(
        linear["linear_solver_info"] == 0
        and linear["scaled_relative_residual"] <= 10.0 * cfg.iterative_relative_tolerance
    )
    physical_valid = bool(
        converged
        and physical["physical_relative_residual"] <= cfg.physical_relative_tolerance
        and abs(physical["weighted_mean_potential"]) <= cfg.gauge_absolute_tolerance
    )
    return {
        "formulation": "divided_log_density_central_difference",
        "solver": "row-equilibrated sparse solve",
        "operator_floor": 0.0,
        "density_modified": False,
        "minimum_equilibrated_log_coefficient": math.nan,
        "maximum_local_log_dynamic_range": float(np.max(log_q) - np.min(log_q)),
        "unrepresentable_equilibrated_coefficient_count": 0,
        "unrepresentable_equilibrated_coefficient_fraction": 0.0,
        "log10_condition_proxy": float(np.max(log_q) - np.min(log_q)) / math.log(10.0),
        **linear,
        **physical,
        "converged": converged,
        "physical_residual_valid": physical_valid,
        "potential": potential,
    }


def solve_cosine_ritz_reference(
    log_q_mass: np.ndarray,
    h: np.ndarray,
    dx: float,
    bounds: tuple[float, float, float, float],
    maximum_mode: int = 5,
) -> dict[str, Any]:
    """Small independent Neumann cosine-Ritz variational cross-check."""
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    forcing = np.asarray(h, dtype=np.float64)
    ny, nx = log_q.shape
    xmin, xmax, ymin, ymax = (float(value) for value in bounds)
    x = xmin + (np.arange(nx) + 0.5) * dx
    y = ymin + (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    modes = [
        (kx, ky)
        for kx in range(maximum_mode + 1)
        for ky in range(maximum_mode + 1)
        if (kx, ky) != (0, 0)
    ]
    basis = []
    gradient_x = []
    gradient_y = []
    for kx, ky in modes:
        x_phase = kx * math.pi * (xx - xmin) / (xmax - xmin)
        y_phase = ky * math.pi * (yy - ymin) / (ymax - ymin)
        basis.append(np.cos(x_phase) * np.cos(y_phase))
        gradient_x.append(
            -(kx * math.pi / (xmax - xmin)) * np.sin(x_phase) * np.cos(y_phase)
        )
        gradient_y.append(
            -(ky * math.pi / (ymax - ymin)) * np.cos(x_phase) * np.sin(y_phase)
        )
    values = np.stack(basis, axis=-1).reshape((-1, len(modes)))
    grad_x_values = np.stack(gradient_x, axis=-1).reshape((-1, len(modes)))
    grad_y_values = np.stack(gradient_y, axis=-1).reshape((-1, len(modes)))
    weights = np.exp(log_q.ravel() - logsumexp(log_q.ravel()))
    sqrt_weight = np.sqrt(weights)[:, None]
    weighted_x = sqrt_weight * grad_x_values
    weighted_y = sqrt_weight * grad_y_values
    gram = weighted_x.T @ weighted_x + weighted_y.T @ weighted_y
    linear = values.T @ (weights * forcing.ravel())
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    threshold = 1.0e-12 * max(float(eigenvalues[-1]), 1e-300)
    retained = eigenvalues > threshold
    coefficients = np.zeros(len(modes))
    coefficients[retained] = -(
        eigenvectors[:, retained].T @ linear
    ) / eigenvalues[retained]
    coefficients = eigenvectors @ coefficients
    potential = (values @ coefficients).reshape(log_q.shape)
    potential -= _weighted_mean(log_q, potential)
    physical = physical_log_conductance_diagnostics(log_q, forcing, potential, dx)
    return {
        "formulation": f"cosine_ritz_variational_mode_{maximum_mode}",
        "solver": "dense eigensolved Ritz normal equations",
        "preconditioner": "none",
        "operator_floor": 0.0,
        "density_modified": False,
        "converged": bool(np.isfinite(coefficients).all()),
        "physical_residual_valid": bool(
            physical["physical_relative_residual"] <= 1.0e-6
        ),
        "iteration_count": 1,
        "scaled_relative_residual": float(
            np.linalg.norm(gram @ coefficients + linear) / max(np.linalg.norm(linear), 1e-300)
        ),
        "condition_proxy": float(eigenvalues[-1] / max(eigenvalues[retained][0], 1e-300))
        if np.any(retained) else math.inf,
        "retained_rank": int(np.sum(retained)),
        "potential": potential,
        **physical,
    }

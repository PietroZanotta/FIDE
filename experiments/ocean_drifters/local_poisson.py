"""Float64-only local Poisson diagnostics for the ocean repair pilot.

This module is intentionally ocean-local.  The conservative solver keeps the
existing arithmetic-face finite-volume equation and applies only positive row
scaling.  The score solver is an independent centered discretization of
``Delta(psi) + grad(log(q)).grad(psi) = h``.  Neither path floors the density,
regularizes the operator, or drops a representable positive face coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from scipy.sparse.linalg import LinearOperator, gmres, spilu


_LOG_SMALLEST_SUBNORMAL = math.log(np.nextafter(0.0, 1.0))


@dataclass(frozen=True)
class LocalPoissonConfig:
    dx: float
    relative_tolerance: float = 1.0e-9
    maximum_iterations: int = 2000
    restart: int = 100
    ilu_drop_tolerance: float = 1.0e-5
    ilu_fill_factor: float = 10.0


def _validate_grid(log_q_mass: np.ndarray, forcing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    h = np.asarray(forcing, dtype=np.float64)
    if log_q.ndim != 2 or h.shape != log_q.shape:
        raise ValueError("log_q_mass and forcing must have the same two-dimensional shape")
    if min(log_q.shape) < 2 or not np.isfinite(log_q).all() or not np.isfinite(h).all():
        raise ValueError("local Poisson inputs must be finite on a grid of at least 2x2")
    return log_q, h


def arithmetic_face_log_conductances(
    log_q_mass: np.ndarray, dx: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return logs of the unchanged arithmetic-face FV conductances and diagonal."""
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    log_geometry = -2.0 * math.log(float(dx))
    log_x = np.logaddexp(log_q[:, :-1], log_q[:, 1:]) - math.log(2.0) + log_geometry
    log_y = np.logaddexp(log_q[:-1, :], log_q[1:, :]) - math.log(2.0) + log_geometry
    diagonal = np.full_like(log_q, -np.inf)
    diagonal[:, :-1] = np.logaddexp(diagonal[:, :-1], log_x)
    diagonal[:, 1:] = np.logaddexp(diagonal[:, 1:], log_x)
    diagonal[:-1, :] = np.logaddexp(diagonal[:-1, :], log_y)
    diagonal[1:, :] = np.logaddexp(diagonal[1:, :], log_y)
    return log_x, log_y, diagonal


def _signed_weighted_mean(log_weights: np.ndarray, values: np.ndarray) -> float:
    log_abs, sign = logsumexp(
        np.asarray(log_weights).ravel(),
        b=np.asarray(values, dtype=np.float64).ravel(),
        return_sign=True,
    )
    if sign == 0.0:
        return 0.0
    return float(sign * math.exp(log_abs - logsumexp(np.asarray(log_weights).ravel())))


def stable_action_diagnostics(
    log_q_mass: np.ndarray,
    forcing: np.ndarray,
    potential: np.ndarray,
    dx: float,
) -> dict[str, float]:
    """Evaluate FV Dirichlet energy, weak identity, and gauge in log arithmetic."""
    log_q, h = _validate_grid(log_q_mass, forcing)
    psi = np.asarray(potential, dtype=np.float64)
    if psi.shape != log_q.shape or not np.isfinite(psi).all():
        raise ValueError("potential must be finite and have the density grid shape")
    log_x, log_y, _ = arithmetic_face_log_conductances(log_q, dx)
    difference_x = psi[:, :-1] - psi[:, 1:]
    difference_y = psi[:-1, :] - psi[1:, :]
    with np.errstate(divide="ignore"):
        energy_logs = np.r_[
            (log_x + np.where(difference_x != 0.0, 2.0 * np.log(np.abs(difference_x)), -np.inf)).ravel(),
            (log_y + np.where(difference_y != 0.0, 2.0 * np.log(np.abs(difference_y)), -np.inf)).ravel(),
        ]
    log_action = float(logsumexp(energy_logs) - logsumexp(log_q.ravel()))
    action = float(math.exp(log_action)) if log_action < math.log(np.finfo(float).max) else math.inf
    weak_action = -_signed_weighted_mean(log_q, h * psi)
    identity_absolute = abs(action - weak_action)
    identity_relative = identity_absolute / max(abs(action), abs(weak_action), np.finfo(float).tiny)
    return {
        "action": action,
        "log_action": log_action,
        "weak_action": weak_action,
        "action_identity_absolute_error": identity_absolute,
        "action_identity_relative_error": identity_relative,
        "weighted_mean_potential": _signed_weighted_mean(log_q, psi),
    }


def _conservative_residual(
    log_q: np.ndarray, h: np.ndarray, psi: np.ndarray, dx: float
) -> dict[str, float]:
    """Evaluate every original unscaled FV row by signed log-sum-exp."""
    log_x, log_y, _ = arithmetic_face_log_conductances(log_q, dx)
    logs = np.full((5, *log_q.shape), -np.inf, dtype=np.float64)
    signs = np.zeros_like(logs)
    dx_psi = psi[:, :-1] - psi[:, 1:]
    dy_psi = psi[:-1, :] - psi[1:, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_log = log_x + np.where(dx_psi != 0.0, np.log(np.abs(dx_psi)), -np.inf)
        y_log = log_y + np.where(dy_psi != 0.0, np.log(np.abs(dy_psi)), -np.inf)
        rhs_log = log_q + np.where(h != 0.0, np.log(np.abs(h)), -np.inf)
    logs[0, :, :-1], signs[0, :, :-1] = x_log, np.sign(dx_psi)
    logs[1, :, 1:], signs[1, :, 1:] = x_log, -np.sign(dx_psi)
    logs[2, :-1, :], signs[2, :-1, :] = y_log, np.sign(dy_psi)
    logs[3, 1:, :], signs[3, 1:, :] = y_log, -np.sign(dy_psi)
    logs[4], signs[4] = rhs_log, np.sign(h)
    residual_log, residual_sign = logsumexp(logs, b=signs, axis=0, return_sign=True)
    residual_log = np.where(residual_sign == 0.0, -np.inf, residual_log)
    rhs_logs = rhs_log.ravel()
    finite_residual = np.isfinite(residual_log.ravel())
    finite_rhs = np.isfinite(rhs_logs)
    log_residual_norm = (
        float(0.5 * logsumexp(2.0 * residual_log.ravel()[finite_residual]))
        if np.any(finite_residual) else -math.inf
    )
    log_rhs_norm = (
        float(0.5 * logsumexp(2.0 * rhs_logs[finite_rhs]))
        if np.any(finite_rhs) else -math.inf
    )
    relative = (
        math.exp(log_residual_norm - log_rhs_norm)
        if np.isfinite(log_residual_norm) and np.isfinite(log_rhs_norm)
        else (0.0 if log_residual_norm == -math.inf else math.inf)
    )
    return {
        "original_relative_residual": float(relative),
        "original_log10_absolute_residual": float(log_residual_norm / math.log(10.0)),
    }


def _row_scaled_conservative_system(
    log_q: np.ndarray, h: np.ndarray, dx: float
) -> tuple[sparse.csr_matrix | None, np.ndarray | None, dict[str, Any]]:
    log_x, log_y, log_diagonal = arithmetic_face_log_conductances(log_q, dx)
    with np.errstate(divide="ignore"):
        log_rhs_abs = log_q + np.where(h != 0.0, np.log(np.abs(h)), -np.inf)
    row_exponent = np.maximum(log_diagonal, log_rhs_abs)
    ny, nx = log_q.shape
    indices = np.arange(nx * ny).reshape((ny, nx))
    directed_exponents = np.r_[
        (log_x - row_exponent[:, :-1]).ravel(),
        (log_x - row_exponent[:, 1:]).ravel(),
        (log_y - row_exponent[:-1, :]).ravel(),
        (log_y - row_exponent[1:, :]).ravel(),
        (log_diagonal - row_exponent).ravel(),
    ]
    rhs_exponents = log_rhs_abs - row_exponent
    face_underflow = directed_exponents < _LOG_SMALLEST_SUBNORMAL
    rhs_underflow = np.isfinite(log_rhs_abs) & (rhs_exponents < _LOG_SMALLEST_SUBNORMAL)
    diagnostics = {
        "minimum_log_face_conductance": float(min(np.min(log_x), np.min(log_y))),
        "maximum_log_face_conductance": float(max(np.max(log_x), np.max(log_y))),
        "log_conductance_range": float(max(np.max(log_x), np.max(log_y)) - min(np.min(log_x), np.min(log_y))),
        "minimum_scaled_log_coefficient": float(np.min(directed_exponents)),
        "genuine_scaled_conductance_underflow_count": int(np.sum(face_underflow)),
        "genuine_scaled_rhs_underflow_count": int(np.sum(rhs_underflow)),
        "row_scaling_exactly_representable": bool(not np.any(face_underflow) and not np.any(rhs_underflow)),
    }
    if not diagnostics["row_scaling_exactly_representable"]:
        return None, None, diagnostics

    rows: list[np.ndarray] = [np.arange(nx * ny)]
    columns: list[np.ndarray] = [np.arange(nx * ny)]
    values: list[np.ndarray] = [np.exp((log_diagonal - row_exponent).ravel())]
    for edge, left, right in (
        (log_x, indices[:, :-1], indices[:, 1:]),
        (log_y, indices[:-1, :], indices[1:, :]),
    ):
        rows.extend((left.ravel(), right.ravel()))
        columns.extend((right.ravel(), left.ravel()))
        values.extend((
            -np.exp(edge.ravel() - row_exponent.ravel()[left.ravel()]),
            -np.exp(edge.ravel() - row_exponent.ravel()[right.ravel()]),
        ))
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(nx * ny, nx * ny),
    ).tocsr()
    rhs = -np.exp(log_q - row_exponent) * h
    return matrix, rhs.ravel(), diagnostics


def _pin(matrix: sparse.csr_matrix, rhs: np.ndarray, pin: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    pinned = matrix.tolil(copy=True)
    pinned.rows[pin] = [pin]
    pinned.data[pin] = [1.0]
    vector = np.asarray(rhs, dtype=np.float64).copy()
    vector[pin] = 0.0
    return pinned.tocsr(), vector


def _gmres_solve(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    pin: int,
    config: LocalPoissonConfig,
    preconditioner_matrix: sparse.csr_matrix | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    system, vector = _pin(matrix, rhs, pin)
    preconditioner_name = "diagonal"
    diagonal = np.maximum(np.abs(system.diagonal()), np.finfo(float).tiny)
    preconditioner: LinearOperator = LinearOperator(
        system.shape, matvec=lambda value: value / diagonal, dtype=np.float64
    )
    candidate = system if preconditioner_matrix is None else _pin(
        preconditioner_matrix, np.zeros_like(rhs), pin
    )[0]
    try:
        ilu = spilu(
            candidate.tocsc(),
            drop_tol=float(config.ilu_drop_tolerance),
            fill_factor=float(config.ilu_fill_factor),
        )
        preconditioner = LinearOperator(system.shape, matvec=ilu.solve, dtype=np.float64)
        preconditioner_name = "SuperLU ILU"
    except Exception as exc:  # the diagonal fallback does not alter the equation
        preconditioner_name = f"diagonal (ILU unavailable: {type(exc).__name__})"
    iterations = 0
    history: list[float] = []

    def callback(residual: float) -> None:
        nonlocal iterations
        iterations += 1
        history.append(float(residual))

    started = time.perf_counter()
    try:
        solution, info = gmres(
            system,
            vector,
            M=preconditioner,
            rtol=float(config.relative_tolerance),
            atol=0.0,
            restart=int(config.restart),
            maxiter=max(1, int(math.ceil(config.maximum_iterations / config.restart))),
            callback=callback,
            callback_type="pr_norm",
        )
        error = ""
    except Exception as exc:
        solution = np.full_like(vector, np.nan)
        info = -1
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    if np.isfinite(solution).all():
        residual = matrix @ solution - rhs
        scaled_relative = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
    else:
        scaled_relative = math.inf
    return solution, {
        "linear_solver": "GMRES",
        "linear_solver_info": int(info),
        "iteration_count": iterations,
        "preconditioner": preconditioner_name,
        "scaled_relative_residual": scaled_relative,
        "last_preconditioned_residual": history[-1] if history else math.nan,
        "solve_seconds": elapsed,
        "solver_error": error,
    }


def solve_log_row_scaled_fv(
    log_q_mass: np.ndarray, forcing: np.ndarray, config: LocalPoissonConfig
) -> dict[str, Any]:
    """Solve the unchanged conservative FV equations after log-domain row scaling."""
    log_q, h = _validate_grid(log_q_mass, forcing)
    matrix, rhs, representation = _row_scaled_conservative_system(log_q, h, config.dx)
    base: dict[str, Any] = {
        "formulation": "log_row_scaled_conservative_fv",
        "density_modified": False,
        "operator_regularized": False,
        **representation,
    }
    if matrix is None or rhs is None:
        return {
            **base,
            "converged": False,
            "scientifically_valid": False,
            "potential": None,
            "action": math.nan,
            "weak_action": math.nan,
            "action_identity_relative_error": math.inf,
            "original_relative_residual": math.inf,
            "scaled_relative_residual": math.inf,
            "iteration_count": 0,
            "solve_seconds": 0.0,
            "solver_error": "genuine float64 underflow remains after exact relative row scaling",
        }
    pin = int(np.argmax(log_q))
    solution, linear = _gmres_solve(matrix, rhs, pin, config)
    if not np.isfinite(solution).all():
        return {**base, **linear, "converged": False, "scientifically_valid": False, "potential": None}
    potential = solution.reshape(log_q.shape)
    potential -= _signed_weighted_mean(log_q, potential)
    scaled_relative = float(
        np.linalg.norm(matrix @ potential.ravel() - rhs)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    action = stable_action_diagnostics(log_q, h, potential, config.dx)
    original = _conservative_residual(log_q, h, potential, config.dx)
    converged = bool(linear["linear_solver_info"] == 0 and scaled_relative <= 10.0 * config.relative_tolerance)
    return {
        **base,
        **linear,
        **action,
        **original,
        "scaled_relative_residual": scaled_relative,
        "converged": converged,
        "scientifically_valid": converged,
        "potential": potential,
    }


def _score_matrix(
    score_x: np.ndarray, score_y: np.ndarray, dx: float
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, float]]:
    """Centered cell-based ``-Delta - score.grad`` with reflected Neumann ghosts."""
    sx = np.asarray(score_x, dtype=np.float64)
    sy = np.asarray(score_y, dtype=np.float64)
    if sx.ndim != 2 or sy.shape != sx.shape or not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise ValueError("score components must be finite grids of equal shape")
    ny, nx = sx.shape
    index = np.arange(nx * ny).reshape((ny, nx))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lap_values: list[float] = []
    inv_dx2 = 1.0 / float(dx) ** 2
    inv_2dx = 0.5 / float(dx)
    for iy in range(ny):
        for ix in range(nx):
            row = int(index[iy, ix])
            coefficients: dict[int, float] = {row: 0.0}
            lap_coefficients: dict[int, float] = {row: 0.0}

            def add(column: int, value: float, lap_value: float = 0.0) -> None:
                coefficients[column] = coefficients.get(column, 0.0) + value
                lap_coefficients[column] = lap_coefficients.get(column, 0.0) + lap_value

            # Cell-centered finite-volume Laplacian with a zero boundary-face
            # flux.  Reflected ghost values supply the centered first derivative.
            if ix > 0:
                add(int(index[iy, ix - 1]), -inv_dx2, -inv_dx2)
                add(row, inv_dx2, inv_dx2)
            if ix + 1 < nx:
                add(int(index[iy, ix + 1]), -inv_dx2, -inv_dx2)
                add(row, inv_dx2, inv_dx2)
            if 0 < ix < nx - 1:
                add(int(index[iy, ix - 1]), sx[iy, ix] * inv_2dx)
                add(int(index[iy, ix + 1]), -sx[iy, ix] * inv_2dx)
            elif ix == 0:
                add(row, sx[iy, ix] * inv_2dx)
                add(int(index[iy, 1]), -sx[iy, ix] * inv_2dx)
            else:
                add(int(index[iy, ix - 1]), sx[iy, ix] * inv_2dx)
                add(row, -sx[iy, ix] * inv_2dx)
            if iy > 0:
                add(int(index[iy - 1, ix]), -inv_dx2, -inv_dx2)
                add(row, inv_dx2, inv_dx2)
            if iy + 1 < ny:
                add(int(index[iy + 1, ix]), -inv_dx2, -inv_dx2)
                add(row, inv_dx2, inv_dx2)
            if 0 < iy < ny - 1:
                add(int(index[iy - 1, ix]), sy[iy, ix] * inv_2dx)
                add(int(index[iy + 1, ix]), -sy[iy, ix] * inv_2dx)
            elif iy == 0:
                add(row, sy[iy, ix] * inv_2dx)
                add(int(index[1, ix]), -sy[iy, ix] * inv_2dx)
            else:
                add(int(index[iy - 1, ix]), sy[iy, ix] * inv_2dx)
                add(row, -sy[iy, ix] * inv_2dx)
            for column in coefficients:
                rows.append(row)
                columns.append(column)
                values.append(coefficients[column])
                lap_values.append(lap_coefficients.get(column, 0.0))
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(nx * ny, nx * ny)).tocsr()
    laplacian = sparse.coo_matrix((lap_values, (rows, columns)), shape=matrix.shape).tocsr()
    magnitude = np.sqrt(sx * sx + sy * sy)
    peclet = 0.5 * float(dx) * magnitude
    return matrix, laplacian, {
        "score_magnitude_minimum": float(np.min(magnitude)),
        "score_magnitude_median": float(np.median(magnitude)),
        "score_magnitude_p95": float(np.percentile(magnitude, 95.0)),
        "score_magnitude_maximum": float(np.max(magnitude)),
        "centered_cell_peclet_maximum": float(np.max(peclet)),
        "centered_cell_peclet_above_one_fraction": float(np.mean(peclet > 1.0)),
    }


def solve_score_form(
    log_q_mass: np.ndarray,
    forcing: np.ndarray,
    score_x: np.ndarray,
    score_y: np.ndarray,
    config: LocalPoissonConfig,
) -> dict[str, Any]:
    """Solve the unregularized score-form PDE with centered differences and GMRES."""
    log_q, h = _validate_grid(log_q_mass, forcing)
    matrix, laplacian, score_stats = _score_matrix(score_x, score_y, config.dx)
    rhs = -h.ravel()
    row_scale = np.maximum(
        np.asarray(np.abs(matrix).max(axis=1).toarray()).ravel(),
        np.abs(rhs),
    )
    if np.any(~np.isfinite(row_scale)) or np.any(row_scale <= 0.0):
        raise ValueError("score-form row scaling is nonfinite or zero")
    inverse_scale = sparse.diags(1.0 / row_scale)
    scaled_matrix = (inverse_scale @ matrix).tocsr()
    scaled_laplacian = (inverse_scale @ laplacian).tocsr()
    scaled_rhs = rhs / row_scale
    pin = int(np.argmax(log_q))
    advection_dominated = score_stats["centered_cell_peclet_above_one_fraction"] > 0.1
    first_preconditioner = None if advection_dominated else scaled_laplacian
    solution, linear = _gmres_solve(
        scaled_matrix,
        scaled_rhs,
        pin,
        config,
        preconditioner_matrix=first_preconditioner,
    )
    if advection_dominated:
        linear["preconditioner"] += " of centered score operator"
        linear["preconditioner_selected_from_peclet_diagnostic"] = True
    else:
        linear["preconditioner_selected_from_peclet_diagnostic"] = False
    # A Laplacian ILU is reusable and is the preferred independent
    # preconditioner.  In very high-score regimes it can be a poor approximation;
    # an ILU of the actual centered matrix is then a legitimate algebraic retry
    # (preconditioning only, with the discrete equation unchanged).
    needs_retry = linear["linear_solver_info"] != 0 or (
        not advection_dominated
        and linear["scaled_relative_residual"] > 10.0 * config.relative_tolerance
    )
    if needs_retry:
        retry_preconditioner = scaled_laplacian if advection_dominated else None
        retry_solution, retry = _gmres_solve(
            scaled_matrix,
            scaled_rhs,
            pin,
            config,
            preconditioner_matrix=retry_preconditioner,
        )
        if retry["scaled_relative_residual"] < linear["scaled_relative_residual"]:
            if not advection_dominated:
                retry["preconditioner"] += " of centered score operator"
            retry["preconditioner_selected_from_peclet_diagnostic"] = advection_dominated
            retry["solve_seconds"] += linear["solve_seconds"]
            retry["preconditioner_retry_used"] = True
            solution, linear = retry_solution, retry
        else:
            linear["solve_seconds"] += retry["solve_seconds"]
            linear["preconditioner_retry_used"] = True
    else:
        linear["preconditioner_retry_used"] = False
    base: dict[str, Any] = {
        "formulation": "analytic_score_centered_difference",
        "density_modified": False,
        "operator_regularized": False,
        **score_stats,
        **linear,
    }
    if not np.isfinite(solution).all():
        return {**base, "converged": False, "scientifically_valid": False, "potential": None}
    potential = solution.reshape(log_q.shape)
    potential -= _signed_weighted_mean(log_q, potential)
    raw_residual = matrix @ potential.ravel() - rhs
    score_relative = float(np.linalg.norm(raw_residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
    scaled_relative = float(
        np.linalg.norm(scaled_matrix @ potential.ravel() - scaled_rhs)
        / max(np.linalg.norm(scaled_rhs), np.finfo(float).tiny)
    )
    action = stable_action_diagnostics(log_q, h, potential, config.dx)
    original = _conservative_residual(log_q, h, potential, config.dx)
    converged = bool(linear["linear_solver_info"] == 0 and scaled_relative <= 10.0 * config.relative_tolerance)
    return {
        **base,
        **action,
        **original,
        "score_form_relative_residual": score_relative,
        "scaled_relative_residual": scaled_relative,
        "converged": converged,
        "scientifically_valid": converged,
        "potential": potential,
    }


def gaussian_kde_log_density_and_score(
    points: np.ndarray,
    atoms: np.ndarray,
    bandwidth: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable Gaussian-mixture log density and score via normalized responsibilities."""
    x = np.asarray(points, dtype=np.float64)
    centers = np.asarray(atoms, dtype=np.float64)
    covariance = np.asarray(bandwidth, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("points and atoms must have shape [N,2]")
    precision = np.linalg.inv(covariance)
    log_normalization = (
        -math.log(2.0 * math.pi)
        - 0.5 * float(np.linalg.slogdet(covariance)[1])
        - math.log(len(centers))
    )
    log_density = np.empty(len(x), dtype=np.float64)
    score = np.empty_like(x)
    for start in range(0, len(x), int(chunk_size)):
        local = x[start : start + int(chunk_size)]
        delta = local[:, None, :] - centers[None, :, :]
        logits = -0.5 * np.einsum("nki,ij,nkj->nk", delta, precision, delta)
        normalizer = logsumexp(logits, axis=1)
        responsibilities = np.exp(logits - normalizer[:, None])
        component_scores = -np.einsum("ij,nkj->nki", precision, delta)
        score[start : start + len(local)] = np.einsum(
            "nk,nki->ni", responsibilities, component_scores
        )
        log_density[start : start + len(local)] = normalizer + log_normalization
    return log_density, score


def transported_projected_log_density_and_score(
    points: np.ndarray,
    *,
    flow: Any,
    time_value: float,
    backward_steps: int,
    atoms: np.ndarray,
    bandwidth: np.ndarray,
    bounds: np.ndarray,
    sensor_centers: np.ndarray,
    sensor_sigma: float,
    multiplier: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the actual transported-KDE projected law stably.

    The KDE is evaluated as a log-sum-exp, so autodiff differentiates stable
    mixture responsibilities rather than an underflowed density quotient.  The
    transport correction and physical/latent Jacobians exactly match the ocean
    reference-density implementation; the Gaussian sensor tilt is then added
    analytically inside the differentiated log law.  Normalization constants do
    not affect the returned score.
    """
    import jax
    import jax.numpy as jnp
    from mfsi.reference_density import (
        backward_latent_with_log_density_correction,
        logistic_log_abs_det_jacobian,
    )

    x = np.asarray(points, dtype=np.float64)
    atom_array = jnp.asarray(atoms, dtype=jnp.float64)
    precision = jnp.asarray(np.linalg.inv(np.asarray(bandwidth, dtype=np.float64)))
    bound_array = jnp.asarray(bounds, dtype=jnp.float64)
    centers = jnp.asarray(sensor_centers, dtype=jnp.float64)
    lam = jnp.asarray(multiplier, dtype=jnp.float64)
    sigma2 = float(sensor_sigma) ** 2
    time = jnp.asarray(float(time_value), dtype=jnp.float64)
    log_normalization = (
        -math.log(2.0 * math.pi)
        - 0.5 * float(np.linalg.slogdet(np.asarray(bandwidth, dtype=np.float64))[1])
        - math.log(len(atoms))
    )

    def single_log_law(point: Any) -> Any:
        z_t = flow.to_latent(point)
        if float(time_value) == 0.0:
            z_0 = z_t
            correction = jnp.asarray(0.0, dtype=jnp.float64)
        else:
            z_0, correction = backward_latent_with_log_density_correction(
                flow.params, z_t, time, steps=int(backward_steps)
            )
        initial_point = flow.to_physical(z_0)
        delta = initial_point[None, :] - atom_array
        logits = -0.5 * jnp.einsum("ki,ij,kj->k", delta, precision, delta)
        reference_log_density = (
            jax.scipy.special.logsumexp(logits)
            + log_normalization
            + logistic_log_abs_det_jacobian(z_0, bound_array)
            + correction
            - logistic_log_abs_det_jacobian(z_t, bound_array)
        )
        sensor_delta = point[None, :] - centers
        features = jnp.exp(-0.5 * jnp.sum(sensor_delta * sensor_delta, axis=1) / sigma2)
        return reference_log_density + jnp.dot(features, lam)

    value_and_score = jax.jit(jax.vmap(jax.value_and_grad(single_log_law)))
    log_density = np.empty(len(x), dtype=np.float64)
    score = np.empty_like(x)
    for start in range(0, len(x), int(chunk_size)):
        local = jnp.asarray(x[start : start + int(chunk_size)], dtype=jnp.float64)
        local_log, local_score = value_and_score(local)
        log_density[start : start + len(local)] = np.asarray(local_log)
        score[start : start + len(local)] = np.asarray(local_score)
    if not np.isfinite(log_density).all() or not np.isfinite(score).all():
        raise FloatingPointError("transported projected-law score evaluation produced nonfinite values")
    return log_density, score

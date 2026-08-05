"""Deterministic fixtures for the methodology's S1--S3 smoke problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import wrap_positions
from .observables import PairBasis, ensemble_pair_moments


def build_smoke_problem_instances(
    seed: int = 20260805,
    dtype_name: str = "float64",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build reusable inputs and targets for S1, S2, and S3.

    The fixtures contain no solver outputs.  They are stable inputs against
    which every future Tesseract implementation can be tested.
    """
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("dtype_name must be float32 or float64")
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    box = jnp.asarray([1.0, 1.0], dtype=dtype)
    key = jax.random.PRNGKey(seed)

    # S1: M=1, N=2 and initial separation strictly below r0.
    s1_coordinates = jnp.asarray([[[0.46, 0.50], [0.54, 0.50]]], dtype=dtype)
    s1_r0 = jnp.asarray(0.16, dtype=dtype)
    s1_kappa = jnp.asarray(40.0, dtype=dtype)
    s1_prox_strength = jnp.asarray(0.05, dtype=dtype)

    # S2: M=8, N=4, one smooth pair observable and a nearby perturbed ensemble.
    key, ref_key, perturb_key = jax.random.split(key, 3)
    s2_reference = jax.random.uniform(ref_key, (8, 4, 2), dtype=dtype) * box
    s2_basis = PairBasis.uniform(1, r_min=0.22, r_max=0.22, width=0.10, dtype=dtype)
    s2_target = ensemble_pair_moments(s2_reference, box, s2_basis)
    s2_perturbation = 0.025 * jax.random.normal(
        perturb_key, s2_reference.shape, dtype=dtype
    )
    # Remove replica-wise global translations so the perturbation genuinely
    # changes pair geometry while retaining an easy translation-equivariance check.
    s2_perturbation = s2_perturbation - jnp.mean(s2_perturbation, axis=1, keepdims=True)
    s2_relaxed_input = wrap_positions(s2_reference + s2_perturbation, box)

    # S3: scalar generator G_a(Z,c)=wrap(X_base+a Z), with a known a_star.
    key, base_key, latent_key = jax.random.split(key, 3)
    s3_base = jax.random.uniform(base_key, (8, 4, 2), dtype=dtype) * box
    s3_latent = jax.random.normal(latent_key, s3_base.shape, dtype=dtype)
    s3_latent = s3_latent - jnp.mean(s3_latent, axis=1, keepdims=True)
    s3_latent = 0.08 * s3_latent / jnp.maximum(
        jnp.std(s3_latent), jnp.asarray(1e-8, dtype=dtype)
    )
    s3_a_star = jnp.asarray(0.65, dtype=dtype)
    s3_a_initial = jnp.asarray(0.20, dtype=dtype)
    s3_basis = PairBasis.uniform(4, r_min=0.08, r_max=0.55, width=0.12, dtype=dtype)
    s3_coordinates_star = wrap_positions(s3_base + s3_a_star * s3_latent, box)
    s3_target = ensemble_pair_moments(s3_coordinates_star, box, s3_basis)

    arrays = {
        "box": np.asarray(box),
        "s1_coordinates": np.asarray(s1_coordinates),
        "s1_r0": np.asarray(s1_r0),
        "s1_kappa": np.asarray(s1_kappa),
        "s1_prox_strength": np.asarray(s1_prox_strength),
        "s2_reference_coordinates": np.asarray(s2_reference),
        "s2_relaxed_coordinates": np.asarray(s2_relaxed_input),
        "s2_target_moments": np.asarray(s2_target),
        "s2_basis_centers": np.asarray(s2_basis.centers),
        "s2_basis_widths": np.asarray(s2_basis.widths),
        "s3_base_coordinates": np.asarray(s3_base),
        "s3_latent_displacements": np.asarray(s3_latent),
        "s3_a_star": np.asarray(s3_a_star),
        "s3_a_initial": np.asarray(s3_a_initial),
        "s3_target_moments": np.asarray(s3_target),
        "s3_basis_centers": np.asarray(s3_basis.centers),
        "s3_basis_widths": np.asarray(s3_basis.widths),
    }
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "dtype": dtype_name,
        "jax_version": jax.__version__,
        "instances": {
            "S1": {
                "description": "two-particle proximal relaxation",
                "M": 1,
                "N": 2,
                "acceptance": [
                    "increase periodic separation",
                    "reduce smooth repulsive energy",
                    "retain center of mass up to periodic gauge",
                    "smooth final-separation sensitivity",
                ],
            },
            "S2": {
                "description": "one-moment ensemble projection",
                "M": 8,
                "N": 4,
                "R": 1,
                "target_is_constructively_feasible": True,
            },
            "S3": {
                "description": "scalar generator recovery through both Tesseracts",
                "M": 8,
                "N": 4,
                "R": 4,
                "generator": "wrap(X_base + a * Z)",
                "a_star": float(s3_a_star),
                "a_initial": float(s3_a_initial),
            },
        },
    }
    return arrays, metadata


def save_smoke_problem_instances(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_path: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    np.savez_compressed(output_path, **arrays)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path, metadata_path

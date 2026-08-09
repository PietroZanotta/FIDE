#!/usr/bin/env python3
"""Parity tests for the two Tesseract projects.

Always tests the pure-JAX kernels embedded in each ``tesseract_api.py`` against
``mfsi_components.py``. If the Pasteur/ISI Labs ``tesseract`` CLI, Docker, and
the two built images are available, it additionally invokes the actual
containerized Tesseracts with ``tesseract run``. No Tesseract Python SDK is used.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mfsi_components as m

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "results" / "learned_mfsi_example_a.npz"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def make_payloads(model):
    key = jax.random.PRNGKey(991)
    t = jnp.asarray(0.43)
    x, _ = m.sample_reference_bridge(key, t, 384, 0.8)
    ref_flat = m.flatten_mlp(model.reference_params)
    potential_flat = m.flatten_mlp(model.potential_params)
    u = m.reference_velocity_net(model.reference_params, t, x)
    ph = m.phi(x)
    jpu = m.jphi_times_velocity(x, u)
    x2 = x[:, None]
    u2 = u[:, None]
    return (
        {"x": x2, "t": t, "velocity_params": ref_flat},
        {
            "x": x2,
            "t": t,
            "velocity": u2,
            "phi_values": ph,
            "jphi_u": jpu,
            "target": m.TARGET,
            "log_base_weights": jnp.zeros_like(x),
            "potential_params": potential_flat,
        },
    )


def pure_jax_parity(model):
    ref_api = load_module("ref_tess_api", ROOT / "tesseracts/reference_transport/tesseract_api.py")
    fib_api = load_module("fib_tess_api", ROOT / "tesseracts/moment_fiber_realizer/tesseract_api.py")
    p1, p2 = make_payloads(model)

    t1 = ref_api.apply_jax(p1)
    u_root = m.reference_velocity_net(model.reference_params, p1["t"], p1["x"][:, 0])[:, None]

    t2 = fib_api.apply_jax(p2)
    root_fib = m.empirical_fiber_state(
        p2["x"][:, 0], p2["velocity"][:, 0], p2["target"],
        log_base_weights=p2["log_base_weights"],
        ph=p2["phi_values"], jphi_u=p2["jphi_u"],
    )
    root_corr = m.learned_correction(model.potential_params, p2["t"], p2["x"][:, 0])[:, None]

    diffs = {
        "t1_velocity": maxdiff(t1["velocity"], u_root),
        "t2_lambda": maxdiff(t2["lambda_value"], root_fib.lambda_),
        "t2_weights": maxdiff(t2["projected_weights"], root_fib.projected_weights),
        "t2_lambda_dot": maxdiff(t2["lambda_dot"], root_fib.lambda_dot),
        "t2_forcing": maxdiff(t2["forcing"], root_fib.forcing),
        "t2_correction": maxdiff(t2["correction"], root_corr),
        "t2_velocity": maxdiff(t2["velocity"], p2["velocity"] + root_corr),
    }

    # Derivative parity for a scalar functional through each component.
    f_root = lambda tt: jnp.mean(m.reference_velocity_net(model.reference_params, tt, p1["x"][:, 0]) ** 2)
    f_tess = lambda tt: jnp.mean(ref_api.apply_jax({**p1, "t": tt})["velocity"] ** 2)
    diffs["t1_dt_gradient"] = abs(float(jax.grad(f_root)(p1["t"]) - jax.grad(f_tess)(p1["t"])))

    def loss_root(ph):
        fib = m.empirical_fiber_state(
            p2["x"][:, 0], p2["velocity"][:, 0], p2["target"],
            log_base_weights=p2["log_base_weights"], ph=ph, jphi_u=p2["jphi_u"],
        )
        return 0.37 * jnp.sum(fib.lambda_ ** 2) + jnp.mean(fib.forcing ** 2)

    def loss_tess(ph):
        o = fib_api.apply_jax({**p2, "phi_values": ph})
        return 0.37 * jnp.sum(o["lambda_value"] ** 2) + jnp.mean(o["forcing"] ** 2)

    gr = jax.grad(loss_root)(p2["phi_values"])
    gt = jax.grad(loss_tess)(p2["phi_values"])
    diffs["t2_phi_vjp"] = maxdiff(gr, gt)

    # The Tesseract calibration uses an implicit custom JVP so forward-mode is
    # available to the JAX recipe as well as reverse-mode.
    dph = jnp.sin(p2["phi_values"])
    root_jvp = jax.jvp(
        lambda z: m._calibrate_empirical_primal(p2["log_base_weights"], z, p2["target"]),
        (p2["phi_values"],), (dph,),
    )[1]
    tess_jvp = jax.jvp(
        lambda z: fib_api.apply_jax({**p2, "phi_values": z})["lambda_value"],
        (p2["phi_values"],), (dph,),
    )[1]
    diffs["t2_phi_jvp"] = maxdiff(root_jvp, tess_jvp)

    assert max(diffs.values()) < 2e-9, diffs
    return diffs, ref_api, fib_api, p1, p2


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "__array__"):
        return np.asarray(x).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _decode_tesseract_json(x):
    if isinstance(x, dict) and x.get("object_type") == "array":
        data = x["data"]
        if data.get("encoding") != "json":
            raise ValueError(f"expected JSON array encoding, got {data.get('encoding')!r}")
        return np.asarray(data["buffer"], dtype=x["dtype"]).reshape(x["shape"])
    if isinstance(x, dict):
        return {k: _decode_tesseract_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode_tesseract_json(v) for v in x]
    return x


def _run_tesseract_cli(image, endpoint, payload):
    # Model parameter payloads are larger than the host's argv limit. Tesseract
    # accepts ``@file`` payloads and mounts the referenced file into the
    # short-lived container.
    with tempfile.TemporaryDirectory(prefix="mfsi_tesseract_") as tmp:
        payload_path = Path(tmp) / "payload.json"
        payload_path.write_text(json.dumps(_jsonable(payload)))
        proc = subprocess.run(
            ["tesseract", "run", image, endpoint, f"@{payload_path}"],
            check=True, text=True, capture_output=True,
        )
    # Runtime output is JSON on stdout. Be tolerant of informational lines.
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return _decode_tesseract_json(json.loads(line))
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"no JSON result from tesseract run {image} {endpoint}: {proc.stdout}")


def runtime_tests(ref_api, fib_api, p1, p2):
    result = {"available": False, "reason": None, "transport": "tesseract CLI / Docker"}
    if shutil.which("tesseract") is None:
        result["reason"] = "Pasteur/ISI Labs tesseract CLI not installed"
        return result
    if shutil.which("docker") is None:
        result["reason"] = "Docker not installed"
        return result
    if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        result["reason"] = "Docker daemon not reachable"
        return result
    images = ("mfsi-reference-transport:latest", "mfsi-moment-fiber-realizer:latest")
    missing = [img for img in images if subprocess.run(
        ["docker", "image", "inspect", img], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode != 0]
    if missing:
        result["reason"] = f"Tesseract images not built: {missing}; run ./scripts/build_tesseracts.sh"
        return result

    result["available"] = True
    np1 = {k: np.asarray(v) for k, v in p1.items()}
    np2 = {k: np.asarray(v) for k, v in p2.items()}
    out1 = _run_tesseract_cli(images[0], "apply", {"inputs": np1})
    out2 = _run_tesseract_cli(images[1], "apply", {"inputs": np2})
    direct1 = ref_api.apply_payload(p1)
    direct2 = fib_api.apply_payload(p2)
    runtime_diffs = {
        "reference_velocity": maxdiff(out1["velocity"], direct1["velocity"]),
        "fiber_weights": maxdiff(out2["projected_weights"], direct2["projected_weights"]),
        "fiber_velocity": maxdiff(out2["velocity"], direct2["velocity"]),
    }
    assert max(runtime_diffs.values()) < 2e-9, runtime_diffs
    result["runtime_diffs"] = runtime_diffs
    result["images"] = list(images)
    return result


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Run validate_pipeline.py first to create {MODEL}")
    model = m.load_learned_model(MODEL)
    diffs, ref_api, fib_api, p1, p2 = pure_jax_parity(model)
    runtime = runtime_tests(ref_api, fib_api, p1, p2)
    result = {
        "pure_jax_kernel_parity": diffs,
        "tesseract_runtime": runtime,
        "configs_present": {
            name: all((ROOT / f"tesseracts/{name}/{f}").exists() for f in (
                "tesseract_api.py", "tesseract_config.yaml", "tesseract_requirements.txt"
            ))
            for name in ("reference_transport", "moment_fiber_realizer")
        },
    }
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "tesseract_validation.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

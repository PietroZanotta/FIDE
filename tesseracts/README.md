# Homometric scientific Tesseracts

The homometric experiment uses the equation-identical wrappers in
`scientific_relaxation/` and `scientific_projection/`. Build both with:

```bash
./scripts/build_scientific_tesseracts.sh
tesseract serve --port 8001 manybody-scientific-relaxation
tesseract serve --port 8002 manybody-scientific-projection
```

For a host-side equivalence and gradient smoke test (no Docker), run:

```bash
PYTHONPATH=src python scripts/run_tesseract_backend_smoke.py
```

The experiment defaults to `solver_backend.kind: local_jax`. To use served
Tesseracts, change that field to `tesseract` in the selected
`configs/homometric_ablation_*.yaml`; the default endpoints are
`http://127.0.0.1:8001` and `http://127.0.0.1:8002`. They can be overridden with
`MBC_RELAXATION_TESSERACT_URL` and `MBC_PROJECTION_TESSERACT_URL`. Setting
`MBC_SOLVER_BACKEND=tesseract` selects Tesseract without editing YAML.
For an in-process debugging run, also set
`MBC_TESSERACT_TRANSPORT=local_api`; scientific runs should use served URLs.

The archived standalone Tesseracts used different solver equations and are
not part of this scientific comparison backend.

# Tesseract boundaries

The host process owns JAX model training and calls the two coarse solver components through
`tesseract-jax`. Each component will be initialized with `tesseract init --recipe jax` so its
API includes `abstract_eval` and `vector_jacobian_product` from the start.

The simulator and observable package under `src/manybody_completion/` is deliberately independent
of Tesseract. The physical-relaxation and moment-projection containers can vendor the same formulas
or consume a small solver-only package without pulling CUDA into their images.

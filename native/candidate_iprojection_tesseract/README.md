# Candidate-batched I-projection Tesseract

This independent optional Tesseract evaluates candidate-specific empirical
I-projection trajectories in one C++/OpenMP call. Its inputs have shapes
`phi=[candidate,time,particle,moment]`, `targets=[candidate,time,moment]`, and
`log_base_weights=[time,particle]`. Each candidate retains its own multiplier
warm start across physical time.

The original `native/iprojection_tesseract` is not modified or linked by this
component.

```bash
.venv/bin/cmake \
  -S native/candidate_iprojection_tesseract \
  -B native/candidate_iprojection_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/candidate_iprojection_tesseract/build -j "$(nproc)"
```

The implementation uses float64 without `-ffast-math`. Its VJP and JVP use
implicit moment-covariance solves and do not differentiate through Newton
iterations.

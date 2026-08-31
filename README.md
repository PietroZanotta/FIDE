# FIDE: Fiber-Informed Differentiable Experimental Design

> “We have no need of other worlds. We need mirrors.”
>
> — **Stanisław Lem, *Solaris***

This project was ideated and evaluated by [Pietro Zanotta](https://github.com/PietroZanotta) as part of the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/) for **Track 1: Inverse Design & Shape Optimization**.

### Contact

- Pietro Zanotta: pzanott1@jhu.edu

Our work, **Fiber-Informed Differentiable Experimental Design (FIDE)**, asks how to design measurements when experiments reveal only **aggregate information** about an evolving population. Among measurement systems that are already good enough for the scientific task, FIDE favors those whose implied full population dynamics remain most compatible with a shared frozen reference model.

This lets us use trusted endpoint information and trusted observable responses without requiring the complete intermediate microscopic dynamics to be known or trusted.

> **Naming convention.** Throughout this README, **FIDE** refers to the complete proposed framework. In experiments and figures, we call the FIDE-selected design **Full**, because it is selected using the **Full action**, our law-level transportability criterion. We therefore use **FIDE** and **Full** interchangeably when referring to the proposed design method; **Full action** refers specifically to its transportability objective. `Law` and `Tangent` denote the corresponding comparison methods.


## Key Features

- **Designed for aggregate observations.**  
  FIDE works with measurements such as sensor intensities, moments, projections, occupancies, or other population-level summaries, without assuming access to the full microscopic state distribution.

- **Reconstructs a law without pretending to recover ground truth.**  
  Aggregate measurements define a *moment fiber*: a family of distributions compatible with what was observed. FIDE selects a canonical member of this family by information projection onto a common reference law.

- **Provides a canonical law-level completion.**  
  Under the regularity conditions of the theory, the information projection is unique at the level of the probability law and takes the form of an exponential tilt of the frozen reference. The measurements determine the constrained directions; the reference determines how the remaining directions are completed.

- **Uses a frozen dynamical reference.**  
  The reference is learned once, for example from independently trusted endpoint information, and is never retrained as the measurement design changes. Every candidate measurement system is therefore compared against the same dynamical geometry.

- **Separates scientific usefulness from dynamical compatibility.**  
  FIDE first restricts attention to designs that are already near-optimal for a user-specified scientific objective. Only within this admissible set does the **Full** criterion ask which design produces the most transportable law-level reconstruction.

- **Measures compatibility at the level of the full distribution.**  
  Rather than asking only whether the measured moments evolve correctly, the **Full action** measures the minimum kinetic correction required for the entire measurement-implied probability law to evolve consistently with the frozen reference dynamics.

- **Characterizes Full transportability through a weighted Poisson problem.**  
  In the balanced probability-law setting, the minimum-energy Full correction is the gradient of a potential solving a density-weighted Poisson equation. Equivalently, the Full action is a weighted negative-Sobolev $H^{-1}$ norm of the reference-relative continuity residual.

- **Separates visible and hidden dynamical corrections.**  
  The minimum Full correction admits an exact orthogonal decomposition into a component visible through the measured moment rates and a complementary *hidden* component that is invisible to those measurements:

  $$\text{Full action} = \text{Tangent action} + \text{Hidden action.}$$

  The decomposition therefore quantifies how much of the law-level dynamical discrepancy can (and cannot) be detected from the chosen observables.

- **Shows that matching moment dynamics can be fundamentally insufficient.**  
  The gap between moment-level and law-level compatibility is not merely numerical. There is no universal constant controlling Full action by Tangent action: smooth examples exist in which the moment-rate correction vanishes while the Full law-level correction remains strictly positive.

- **Supports differentiable experimental design.**  
  When sensor locations or measurement parameters vary continuously, both the calibrated information projection and the Full action can be differentiated with respect to the design using Tesseract, allowing for fast convergece.
## Table of Contents
- [FIDE: Fiber-Informed Differentiable Experimental Design](#fide-fiber-informed-differentiable-experimental-design)
    - [Contact](#contact)
  - [Key Features](#key-features)
  - [Table of Contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Problem Statement](#problem-statement)
- [\\mathcal{F}_\\eta(\\hat c_\\eta(t))](#mathcalfetahat-cetat)
- [\\mathbb{E}_{Q}\[\\Phi_\\eta(X)\]](#mathbbeqphietax)
  - [Methodology](#methodology)
- [Q\_t^\\eta](#q_teta)
- [\\right)](#right)
- [\\right)](#right-1)
- [A(\\eta)](#aeta)
  - [\\eta](#eta)
  - [Where Does Tesseract Enter the Picture?](#where-does-tesseract-enter-the-picture)
    - [The two Tesseracts used by the analytical example](#the-two-tesseracts-used-by-the-analytical-example)
    - [Where they sit in one Full-action evaluation](#where-they-sit-in-one-full-action-evaluation)
- [\\sum\_i q\_i(\\lambda,\\eta),\\Phi\_\\eta(x\_i)](#sum_i-q_ilambdaetaphi_etax_i)
- [C\_{t,\\eta}](#c_teta)
- [D\_\\eta\\lambda\_{t,\\eta}](#d_etalambda_teta)
    - [What this looks like at the scale of the analytical run](#what-this-looks-like-at-the-scale-of-the-analytical-run)
    - [Benchmark: Tesseract versus a full JAX implementation](#benchmark-tesseract-versus-a-full-jax-implementation)
  - [Numerical Experiments](#numerical-experiments)
  - [Structure of this Repository](#structure-of-this-repository)
  - [Getting Started](#getting-started)
  - [Future Work](#future-work)
  - [Tech Stack](#tech-stack)
    - [Hardware configuration](#hardware-configuration)
    - [Software environment](#software-environment)


## Motivation
Many scientific systems are naturally described not by a single state, but by a **distribution over microscopic states**. Such a behaviour shows up in many domains of science, for example:

- a cloud of tracer particles has a spatial density;
- a population of cells has a distribution over molecular phenotypes;
- an ensemble of molecules has a distribution over conformations;
- a plasma has a distribution over positions and velocities.

In practice, however, experiments rarely reveal these distributions directly. Instead, they return a small number of aggregate summaries: the intensity at a detector, the expression of a marker panel, a scattering measurement at a particular angle, a spatially averaged concentration, or a handful of low-order moments.

Once observations are aggregate, there is generally no **unique** microscopic law consistent with them. A measurement tells us something about the population, but it also leaves many aspects of that population unresolved. Two sensor configurations can therefore be almost equally useful for the stated scientific task while constraining very different directions of the underlying distribution.

**Fiber-Informed Differentiable Experimental Design (FIDE)** is useful when we care not only about whether a measurement system is informative enough, but also about **what sort of complete dynamical story that measurement system forces us to tell**.

The distinction becomes especially important when we already possess a reasonable dynamical reference model. The reference might be learned from well-characterized endpoint experiments, obtained from a separately validated model, or constructed from data considered reliable for a specific purpose. **We do not require this reference to be the true intermediate dynamics.** Indeed, endpoint information alone cannot generally tell us what happened in between. Its role is instead to provide a common dynamical geometry against which every candidate experiment can be compared. The reference is trained once and then frozen.

The trust assumptions are deliberately weaker than trusting a complete simulator trajectory. We require the endpoint information used to anchor the reference to be credible **for that purpose**, and we require the intermediate aggregate measurements (or the predicted responses used to design them) to be credible in the observable directions they represent. We do not require the simulator's entire intermediate state distribution to be physically correct.

A model can be adequate for a detector response, a moment, or a quantity of interest without being adequate for every unobserved degree of freedom in its state space. FIDE is designed precisely for this kind of quantity-specific trust.

For a given experiment, the aggregate measurements define a family of compatible distributions: a **moment fiber**. FIDE does not pretend that these measurements identify the true law. Instead, it completes what they leave unresolved by selecting the member of the fiber closest to the frozen reference. In this sense, the reconstruction contains two different kinds of information: **the experiment determines what must change, while the reference determines how to fill in what the experiment does not tell us.**

This distinction is particularly useful in simulation-assisted science. A detailed simulator may be available and may even produce complete intermediate microscopic states. But using those states as ground truth would make a much stronger assumption: that the simulator is physically reliable in *all* directions of the state space.

Often this is not what has actually been validated. We may trust the simulator's prediction of a detector response, a spatial average, a marker concentration, or a downstream quantity of interest without trusting the entire simulated population law. FIDE lets us use the simulator where it is credible without silently promoting the rest of the simulated state to physical truth.

This also explains why endpoint information can be enough to construct the reference for FIDE's purpose. The reference is not meant to certify what truly happened between the endpoints. It is a **common dynamical background**. Intermediate aggregate observations then tell us where that background must be corrected.

FIDE asks:

> *If two experiments answer the scientific question equally well, which one forces us to invent the least additional dynamics in the parts of the population law that neither experiment directly observes?*

This ordering is important. A useless measurement could be extremely easy to reconcile with the reference simply because it says almost nothing. FIDE therefore makes scientific adequacy primary: it first restricts attention to designs lying within a predeclared near-optimal scientific-risk set, and only then uses full-law transportability to choose among them.

> **FIDE is most useful when many experiments are good enough for the scientific task, but those experiments leave different parts of an evolving population distribution unresolved, and we want the unresolved part to be completed in a way that remains compatible with a common, independently anchored dynamical reference.**

## Problem Statement

Consider a dynamical system whose microscopic state at time $t$ is a random variable

$$
X_t \sim P_t,
$$

where $P_t$ is the unknown population law. In many experiments we cannot observe $P_t$ directly. Instead, a **measurement design** $\eta$ (for example, a set of sensor locations, projection angles, or observable parameters) determines an observable map $\Phi_\eta$, and the experiment gives access only to aggregate quantities of the form

$$
c_\eta(t) = \mathbb{E}_{P_t}\left[\Phi_\eta(X_t)\right].
$$

In practice these quantities may themselves be observed sparsely and noisily, so we reconstruct a smooth trajectory $\hat c_\eta(t)$ from the available measurements.

The central difficulty is that finitely many aggregate measurements do **not** uniquely identify the underlying law. At each time they instead define a **moment fiber**

$$
\mathcal{F}_\eta(\hat c_\eta(t))
=
\left\{
Q :
\mathbb{E}_{Q}[\Phi_\eta(X)]
=
\hat c_\eta(t)
\right\},
$$

containing all population laws that reproduce the measured observables.

Changing the measurement design therefore changes more than the values we observe: it changes **which directions of the population law are constrained and which remain unresolved**.

At the same time, suppose we have a common frozen reference dynamics $(\widetilde Q_t, u_t)$, for example learned from independently trusted endpoint populations. We do **not** assume that this reference describes the true intermediate dynamics. Its role is to provide the same dynamical geometry against which every candidate measurement design is evaluated.

This leads to the design question at the heart of FIDE:

> **Among measurement systems that are already scientifically adequate, which one implies a complete population-law evolution that is most compatible with the same frozen reference dynamics?**

FIDE separates these two requirements deliberately. A user-specified **scientific risk** $R(\eta)$ determines whether a measurement system is useful for the scientific task. Only among designs satisfying the desired risk tolerance do we compare their law-level dynamical compatibility.

The resulting **FIDE/Full design** solves, schematically,

$$
\eta_{\mathrm{Full}}
\in
\arg\min_{\eta:\,R(\eta)\leq R_{\max}}
A(\eta),
$$

where $A(\eta)$ is the **Full action**: the minimum dynamical correction required to realize the complete law path implied by the measurements.


## Methodology

![FIDE workflow: from measurements to dynamically compatible laws](visual_abstract/output_png/fide_diag3.png)

*FIDE workflow. The forward pass constructs a measurement-implied law and evaluates its scientific risk and Full action; gradients are then propagated backward through the pipeline to update the measurement design.*

FIDE turns the problem above into a differentiable pipeline from **measurement design** to **law reconstruction** to **dynamical compatibility**.

1. **Choose a measurement design and reconstruct its aggregate observations.**

   A design $\eta$ specifies the sensor geometry or, more generally, the parameters of the observable map $\Phi_\eta$. Sparse population-level measurements are used to reconstruct the moment trajectory

   $$
   \hat c_\eta(t)
   \quad\text{and, when needed,}\quad
   \dot{\hat c}_\eta(t).
   $$

   These are the pieces of intermediate information we require the experiment or predictive simulator to provide reliably.

2. **Construct the moment fiber.**

   At each time, the reconstructed measurements determine the set

   $$
   \mathcal{F}_\eta(\hat c_\eta(t)),
   $$

   containing every law consistent with the observations.

   The experiment therefore does not provide a unique population law. It provides a constraint on the law.

3. **Lift the measurements to a canonical law.**

   FIDE resolves the ambiguity inside the moment fiber using the same frozen reference $\widetilde Q_t$ for every candidate design. At each time it computes the information projection

   $$
   Q_t^\eta
   =
   \arg\min_{Q \in \mathcal{F}_\eta(\hat c_\eta(t))}
   D_{\mathrm{KL}}\!\left(Q\,\|\,\widetilde Q_t\right).
   $$

   The resulting path $t \mapsto Q_t^\eta$ is the **measurement-implied law path**.

   Importantly, $Q_t^\eta$ is not claimed to be the unknown physical truth. It is a canonical completion of the aggregate evidence: the measurements determine what must be matched, while the reference supplies what remains unresolved.

4. **Evaluate scientific adequacy.**

   The projected law is evaluated using an externally specified scientific risk

   $$
   R(\eta),
   $$

   such as error in a quantity of interest, held-out reconstruction error, or another task-specific criterion.

   This step remains primary: a design with poor scientific performance is not made attractive simply because it is easy to reconcile with the reference.

5. **Measure full-law dynamical compatibility.**

   Let $q_t^\eta$ denote the density of the projected law and let $u_t$ be the frozen reference velocity. FIDE asks for the smallest velocity correction $\delta_t$ such that the **entire projected law** satisfies the continuity equation

   $$
   \partial_t q_t^\eta
   +
   \nabla\!\cdot
   \left(
   q_t^\eta (u_t+\delta_t)
   \right)
   =
   0.
   $$

   In the balanced setting (no probability mass can be created), the minimum-energy correction has the form

   $$
   \delta_t^\star = -\nabla\psi_t^\star,
   $$

   where $\psi_t^\star$ solves the density-weighted Poisson problem

   $$
   \nabla\!\cdot
   \left(
   q_t^\eta\nabla\psi_t^\star
   \right)
   =
   \partial_t q_t^\eta
   +
   \nabla\!\cdot(q_t^\eta u_t).
   $$
  We retrieve similar equations for the unbalanced dynamics case as well. The corresponding **Full action**

   $$
   A(\eta)
   =
   \int
   \mathbb{E}_{Q_t^\eta}
   \left[
   \|\delta_t^\star(X)\|^2
   \right]
   \rho(dt)
   $$

   measures how much additional dynamical effort is required to realize the complete measurement-implied law relative to the frozen reference.

6. **Optimize the experiment, not the reference.**

   FIDE minimizes Full action only inside the scientifically admissible set,

   $$
   \min_\eta A(\eta)
   \qquad
   \text{subject to}
   \qquad
   R(\eta)\leq R_{\max}.
   $$

   When $\eta$ is continuous, gradients are propagated through the moment reconstruction, information projection, scientific risk, and Full-action computation. The measurement geometry can therefore be updated with a gradient-based step such as

   $$
   \eta
   \leftarrow
   \eta
   -
   \alpha\nabla_\eta \mathcal{L}.
   $$

   The frozen reference itself is never retrained during this optimization. Every candidate design is judged against the same background dynamics.

In short, the FIDE pipeline is

> **design measurements → reconstruct aggregate information → define the moment fiber → project the frozen reference onto that fiber → measure scientific risk and Full action → differentiate through the pipeline using Tesseract → update the experiment.**

The optimization therefore searches for measurements that remain useful for the scientific task while inducing a complete law-level reconstruction that is as dynamically compatible as possible with the common reference.

## Where Does Tesseract Enter the Picture?

FIDE's mathematical pipeline contains two implicit numerical problems inside every differentiable Full-action search evaluation: an **information-projection problem** and a **density-weighted Poisson problem**. 
In the analytical example, both must be solved repeatedly as the two sensor angles change, and the derivative of the final action must pass through both solutions to reach those angles.

This is exactly where Tesseract becomes pivotal. Our outer optimization is written in JAX, while the search's inner solvers are double-precision C++17/OpenMP kernels with their own numerical algorithms. Without Tesseract we would face an unattractive choice:

- lose the speedup that C++ offers over a Jax-based implementation; or
- call the C++ code as an opaque accelerator and break the gradient at the Python/C++ boundary.

Tesseract lets us keep the native solvers **and** expose their mathematically correct derivatives to JAX. The result is not a collection of separately optimized components; it is one end-to-end differentiable sensor-design
function.

### The two Tesseracts used by the analytical example

| Pipeline stage             | What it computes                                                                                                       | Native implementation                                                                                               | How it differentiates                                                                                      |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **Empirical I-projection** | The exponential tilt of the frozen reference particles that matches the two reconstructed sensor moments at every time | Batched C++/OpenMP damped Newton solver in [`native/iprojection_tesseract`](native/iprojection_tesseract/README.md) | Implicit JVP/VJP through the converged moment equation using the moment-covariance system                  |
| **Weighted Poisson**       | The minimum-energy correction that realizes the complete projected law relative to the frozen reference flow           | Batched C++17/OpenMP matrix-free PCG solver in [`native/poisson_tesseract`](native/poisson_tesseract/README.md)     | Implicit reverse pass through an adjoint Poisson solve, rather than backpropagation through PCG iterations |

Both components are loaded in-process with `Tesseract.from_tesseract_api` and
called from JAX through `tesseract-jax`. Complete trial/time batches cross each
boundary in a single call, which avoids launching one native call for every
small nonlinear or linear system.

The analytical example enables the two backends explicitly in
[`config.json`](experiments/toy_example_percentage/config.json):

```json
{
  "projection": {
    "trajectory_backend": "tesseract_cpp"
  },
  "optimization": {
    "full_gradient_poisson_backend": "tesseract_cpp",
    "full_exact_poisson_backend": "tesseract_cpp"
  }
}
```

The JAX-facing adapters live in
[`src/mfsi/projection_tesseract.py`](src/mfsi/projection_tesseract.py) and
[`src/mfsi/poisson_tesseract.py`](src/mfsi/poisson_tesseract.py). The analytical
experiment composes them in
[`experiments/toy_example_percentage/experiment.py`](experiments/toy_example_percentage/experiment.py).

### Where they sit in one Full-action evaluation

For the analytical example, the design variable is
$\eta=(\theta_1,\theta_2)$, the angular positions of two Gaussian sensors on a
fixed ring. One forward-and-backward evaluation follows this chain:

```text
sensor angles eta
    |
    v
sensor responses Phi_eta and reconstructed moments c_eta(t)       [JAX]
    |
    v
calibrated exponential-tilt multipliers lambda_eta(t)              [I-projection Tesseract]
    |
    v
projected particle law, continuity forcing, and physical raster    [JAX]
    |
    v
weighted-Poisson potential and Full action A(eta)                  [Poisson Tesseract]
    |
    v
gradient dA/deta and constrained sensor update                     [JAX]
```

The first Tesseract solves, at each selected time, the calibration equation

$$
\sum_i q_i(\lambda,\eta)\,\Phi_\eta(x_i)
=
\hat c_\eta(t),
\qquad
q_i(\lambda,\eta)
\propto
b_i\exp\!\left(\lambda^\top\Phi_\eta(x_i)\right).
$$

The sensor angles affect both the observable values and the reconstructed
moment targets. Rather than differentiating through every Newton step, the
Tesseract differentiates the converged calibration condition. Its central
linear system is the covariance of the two sensor observables,

$$
C_{t,\eta}
=
\operatorname{Cov}_{Q_t^\eta}(\Phi_\eta,\Phi_\eta),
\qquad
D_\eta\lambda_{t,\eta}
=
-C_{t,\eta}^{-1}D_\eta F.
$$

The calibrated particle weights determine the projected density and its
reference-relative continuity forcing. After rasterization, the second
Tesseract solves the weighted-Poisson system that defines the Full correction.
Its adjoint rule propagates the action derivative back through the PDE solution
without storing or differentiating through hundreds of PCG iterations.

The backward pass therefore crosses both native boundaries:

```text
dA/deta
  <- Poisson implicit adjoint
  <- raster and continuity-forcing derivatives
  <- I-projection implicit covariance solve
  <- sensor-response and reconstruction derivatives
```

This composition matters because the two inner problems cannot be optimized
independently. Moving a sensor changes the moment fiber; that changes the
information-projected law; that changes the coefficients and forcing of the
Poisson equation; and only then does it change the Full action used to update
the sensor.

### What this looks like at the scale of the analytical run

The frozen reference contains `2,592` weighted particles and each candidate
design has two moment constraints. During differentiable Full candidate
generation, the configured proxy uses four common-random-number trials and
seven time nodes. The I-projection Tesseract solves the corresponding
`[trial, time]` trajectory batch, and the Poisson Tesseract receives all
`4 x 7 = 28` raster systems of size `41 x 41` in one call. OpenMP parallelizes
the independent systems on the CPU while JAX retains ownership of the outer
objective and sensor optimizer.

### Benchmark: Tesseract versus a full JAX implementation

We benchmarked each native component against its equivalent compiled JAX
implementation on the laptop configuration reported below. Here we retain the
original practical timing boundary: the Tesseract measurements include the
Python/native call and host–device transfer costs, while JAX inputs remain on
the GPU. Compilation and one-time setup are excluded from both paths. Times are
steady-state medians after warm-up; the I-projection and Poisson results use
five timed repetitions.

| Component and path                         | Workload                                                      |   JAX GPU | Tesseract, transfer-inclusive |          JAX / Tesseract |
| :----------------------------------------- | :------------------------------------------------------------ | --------: | ----------------------------: | -----------------------: |
| I-projection trajectory, forward           | `4 × 7` projections, 2,592 particles, 2 moments               |  84.48 ms |                       5.64 ms |               **14.99×** |
| I-projection trajectory, value + gradient  | Same workload, including implicit VJP                         |  93.73 ms |                      16.24 ms |                **5.77×** |
| Complete differentiable I-projection stage | Sensor response, reconstruction, trajectory, value + gradient | 104.88 ms |                      31.43 ms |                **3.34×** |
| Weighted Poisson, forward                  | 28 systems of size `41 × 41`                                  |  21.25 ms |                       1.59 ms |               **13.40×** |
| Weighted Poisson, value + gradient         | Same workload, including adjoint solve and VJP                |  50.33 ms |                       8.97 ms |                **5.61×** |
| Galerkin assembly, forward only            | `[256, 280, 16, 2]` fixed-feature chunk                       |         — |                             — | **0.21x** (4.79× slower) |

The I-projection result is the strongest reason not to implement the complete pipeline only in JAX. Its workload consists of many small, sequentially warm-started Newton solves: a poor fit for accelerator control flow, but a good fit for the batched C++/OpenMP implementation. The weighted-Poisson systems also benefit  substantially from the native matrix-free PCG and implicit adjoint. Because both operations occur inside repeated sensor-objective and gradient evaluations, these measured reductions affect the expensive inner loop rather than a one-time setup stage.

The Galerkin result is deliberately included even though it goes the other way. Its dense contraction is well suited to the GPU; for this tested shape, the transfer-inclusive native CPU assembler is `4.79×` slower. We therefore do **not** use the Galerkin Tesseract merely on the assumption that native code must be faster. JAX should remain the performance default for this workload unless a different platform, memory constraint, or larger study demonstrates an advantage. The author also acknowledge the Galerkin result might be confounded by the amount of time spent optimizing that specific Tesseract. Further work on that component might deliver better results.

The comparisons also verify numerical and derivative agreement:

- I-projection: all native systems converged, maximum calibration residual `9.95e-8`, and trajectory-gradient relative difference `9.76e-6`;
- weighted Poisson: all native systems converged to the configured `1e-6` tolerance, with forward relative difference `5.01e-8` and relative gradient difference `1.25e-7`; and
- Galerkin: maximum absolute discrepancy `1.78e-13`.

These numbers justify a **selective heterogeneous implementation**: Tesseract is load-bearing for the I-projection and weighted-Poisson solvers, where it preserves gradients and remains faster even after crossing the CPU/GPU boundary. It is not a blanket replacement for JAX, as the Galerkin counterexample makes clear. Reproducible benchmark details are stored in the 
[I-projection results](experiments/toy_example_percentage/outputs/iprojection_backend_benchmark.json),
[Poisson results](experiments/toy_example_percentage/outputs/poisson_backend_benchmark.json),
and [Galerkin benchmark driver](native/galerkin_tesseract/benchmark.py).

Finally, the differentiable proxy is used for **candidate generation**, not as a substitute for scientific validation. Promising sensor geometries are re-evaluated with the authoritative physical-density Full solver on the frozen selection bank, using a `101 x 101` raster and all `21` scientific time nodes. Only after the geometry is frozen is it evaluated on the disjoint `128`-trial validation bank. Tesseract makes the heterogeneous search differentiable; the higher-resolution, fail-closed audit protects the reported scientific
result.

## Numerical Experiments

## Structure of this Repository

## Getting Started

## Future Work

Future work will focus on:

- **Scaling to many-body systems.**  
  Extend the current benchmark suite beyond low-dimensional transport problems to interacting many-body systems, with magnetic skyrmion dynamics as a first target.

- **Empirically validating the unbalanced formulation.**  
  The current experiments evaluate the balanced probability-law setting. A natural next step is to test the move–reaction extension on systems in which mass is created, destroyed, or exchanged, such as active nematics.

- **Stress-testing the limits of FIDE.**  
  Run systematic ablation suites over measurement sparsity, noise level, number of observables, reference quality, and scientific-risk tolerance to identify regimes in which Full transportability provides meaningful design information and regimes in which it does not.

- **Studying robustness to the frozen reference.**  
  Full action is reference-relative, so an important question is how stable the selected design is across reference-model seeds, architectures, training procedures, and degrees of reference misspecification.

- **Moving from retrospective to prospective design.**  
  In a prospective experimental-design setting, measurements associated with a candidate design have not yet been acquired. Future work will study predictive models for candidate measurement responses and quantify how errors in those predictions propagate into the final FIDE design.

- **Scaling the Full-action computation to higher-dimensional state spaces.**  
  The current benchmarks solve the weighted Poisson problem on a physical spatial discretization. Extending FIDE to high-dimensional molecular, cellular, or many-body state spaces will require more scalable representations of the Full correction.

- **Sequential and adaptive experimental design.**  
  The current formulation optimizes a fixed measurement design. A natural extension is to allow the design to evolve as observations arrive, choosing subsequent measurements based on the law reconstruction induced by earlier ones.

- **Broadening the class of observables.**  
  Beyond localized sensors, future experiments could consider projections, nonlinear detector responses, pair statistics, structure factors, and other scientifically meaningful aggregate observables.

- **Extending the theory beyond the current regularity regime.**  
  Further work could study rank-deficient or changing observable families, nonsmooth measurement maps, more general boundary conditions, and broader classes of unbalanced dynamics.
## Tech Stack

### Hardware configuration

All reported experiments were executed locally on a high-performance laptop, there was no remote cluster or external job scheduler. The machine used an Intel Core Ultra 9 275HX processor with 24 cores and a maximum turbo frequency of 5.4 GHz, together with an NVIDIA GeForce RTX 5090 Laptop GPU with 24 GB of dedicated GDDR7 memory.

The workload was divided between the GPU and CPU:

- **GPU:** JAX particle simulations, endpoint-reference flow training, and automatic differentiation.
- **CPU:** batched native information-projection trajectories and the Tesseract weighted-Poisson search solves, using C++17 and OpenMP. The authoritative sparse Poisson evaluations also ran on the CPU.

This split is central to the implementation: JAX retains the outer
differentiable experiment-design loop, while Tesseract carries gradients through the CPU-native solvers described above.

### Software environment

The experiments ran under Windows Subsystem for Linux 2 (WSL2), using Ubuntu 24.04.3 LTS and Linux kernel `6.6.87.2-microsoft-standard-WSL2`. The principal implementation used Python 3.12.3 and JAX 0.8.3 with 64-bit floating-point precision enabled. GPU acceleration used the JAX CUDA 12 backend and its
packaged CUDA 12.9 runtime.

The information-projection Tesseract evaluates complete batched empirical projection trajectories with a double-precision Newton solver. Multipliers are warm-started between consecutive time nodes, while independent trajectories and linear-algebra operations are parallelized with OpenMP. The native extension is exposed through pybind11 and integrated into JAX with `tesseract-jax`. Its JVPs and VJPs use implicit moment-covariance solves rather than differentiating through individual Newton iterations. 

The native extensions were compiled in release mode with GNU C++ 13.3, C++17, `-O3`, and `-march=native`. We deliberately did not enable `-ffast-math`, in order to preserve the numerical behavior of the double-precision solvers.

| Component                      | Configuration used for the reported experiments       |
| :----------------------------- | :---------------------------------------------------- |
| Execution host                 | Local high-performance laptop                         |
| Operating system               | Ubuntu 24.04.3 LTS under WSL2                         |
| Linux kernel                   | `6.6.87.2-microsoft-standard-WSL2`                    |
| Processor                      | Intel Core Ultra 9 275HX, 24 cores, up to 5.4 GHz     |
| GPU                            | NVIDIA GeForce RTX 5090 Laptop GPU                    |
| GPU memory                     | 24 GB GDDR7                                           |
| NVIDIA driver                  | 581.57                                                |
| Python                         | 3.12.3                                                |
| JAX                            | 0.8.3                                                 |
| jaxlib                         | 0.8.3                                                 |
| Accelerator backend            | CUDA 12 via JAX CUDA plugin 0.8.3                     |
| CUDA runtime                   | 12.9, packaged with JAX                               |
| NumPy                          | 2.5.1                                                 |
| SciPy                          | 1.18.0                                                |
| Tesseract Core                 | `tesseract-core` 1.11.0                               |
| Tesseract JAX interface        | `tesseract-jax` 0.2.3                                 |
| Information-projection backend | Native Tesseract C++/OpenMP batched trajectory solver |
| Differentiable Poisson backend | Native Tesseract C++/OpenMP batched PCG solver        |
| C++ language standard          | C++17                                                 |
| C/C++ compiler                 | GCC/G++ 13.3.0                                        |
| Build system                   | CMake 4.4.2, release configuration                    |
| Python/C++ bindings            | pybind11 3.1.0                                        |
| CPU parallelization            | OpenMP 4.5                                            |
| Native compilation             | `-O3`, `-march=native`, without `-ffast-math`         |
| Native numerical precision     | IEEE 754 double precision (`float64`)                 |
| Neural optimization            | Custom JAX implementation of Adam                     |
| JAX numerical precision        | 64-bit floating point enabled                         |
| Job scheduling                 | Local execution; no external scheduler                |

These are the versions used to produce the reported results, not the minimum installation requirements. The package itself supports Python 3.11 or newer; see `pyproject.toml` and the experiment-specific reproduction instructions for
installation details.

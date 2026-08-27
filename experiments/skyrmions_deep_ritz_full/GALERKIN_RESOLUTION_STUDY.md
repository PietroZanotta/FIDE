# Galerkin resolution study

## Scope and outcome

This was a selection-development-only fixed-geometry qualification study. No eta optimization, Pareto sweep, winner selection, incumbent replacement, validation construction, or validation access occurred.

Primary K=280 result: **B. K280 REMAINS PHYSICALLY INADEQUATE AT LOW RISK**. Conditional basis/rank study run: `yes`. Final decision: **C. GALERKIN FULL DISCRETIZATION NOT YET QUALIFIED FOR A PARETO SWEEP**.

## Why official Pareto v1 failed

The frozen v1 attempt reproduced K=280 and all gradients, forcing, geometry, algebra, rank, weak, gauge, and moment diagnostics. It stopped before optimization because every exact-risk-feasible 0.5% start had selection-audit Ritz-energy residual 0.1073–0.1084 against the unchanged 0.08 threshold. V1 remains a frozen failed protocol and was neither modified nor resumed.

V1 immutability audit passed: `True`. Resolution protocol SHA-256: `1e3aa27523321c5ab51c8ca917eb7da19fe3dad0fbb9b732ec9c30f2ee5ff7d8`.

## Fixed geometries

| id | provenance | exact eta |
|---|---|---|
| law | config.json envelope.law_eta | `[0.890286510596537, 0.227289528868506, 1.310368832144490, 0.859163192162967, 0.797588822714243, 0.535723001316333, 1.610343150447571, 0.583219225445585]` |
| historical_0p5 | original selection Pareto 0.5% frozen geometry | `[0.888224002114442, 0.226590028285788, 1.308928302996688, 0.862825514790280, 0.786665206117643, 0.541803221343441, 1.616175859255502, 0.584353406982718]` |
| historical_1 | original selection Pareto 1% frozen geometry | `[0.891649766087215, 0.215921041817233, 1.325499049896834, 0.861978425574543, 0.774033375218434, 0.527859082517257, 1.626809481063866, 0.577508746042611]` |
| historical_2 | original selection Pareto 2% frozen geometry | `[0.894577442995983, 0.204111618925572, 1.340086477059110, 0.863550818217665, 0.760018739646390, 0.514351562674927, 1.637615065201357, 0.566685160921243]` |
| eta0_3pct | config.json envelope.eta0 | `[0.895415376776124, 0.205926316324706, 1.334378809838382, 0.865428835291722, 0.750835536576608, 0.517910032926475, 1.642373524978473, 0.588359969589811]` |
| eta_grad_3pct | frozen selection-only continuous 3% Galerkin winner | `[0.895371148114089, 0.205982940238786, 1.334525121515147, 0.865464965382237, 0.750749623351011, 0.518133188490931, 1.642405611981796, 0.588309862016330]` |

## Selection-development data policy and construction

The two new reference banks are labeled `selection_development_only`. They use fresh independent initial draws and the unchanged frozen reference dynamics; the reference model was not retrained. The 32,768 train and 16,384 audit banks provide exact prefixes for the complete prescribed ladder. All basis work was streamed, so no full K=280 basis cache was built.

Validation arrays—including old truth/fit/audit/noise and v1 seed/data—were not opened. Exact initial-state overlap checks covered fresh train versus fresh audit and every permitted historical selection bank; every count was zero. Historical/future validation disjointness rests on the predeclared independent versioned seed namespace and fresh continuous draws, without violating the no-access rule.

| role | samples | seed | SHA-256 | artifact SHA-256 |
|---|---:|---:|---|---|
| train | 32768 | 1268305497 | `41dae404c7e3064f658b2d1124976db91c1d3a112698fd39da31f762574ab67c` | `a9249c1cf04b38d1c4c4be53081cff014da90c9ab380fe8a6a7cfb9a4332ec1d` |
| audit | 16384 | 1787111204 | `87ef5e195aa66af0d0abdcb2fa18e1a6f43a384e1c4c9ce0bc25aabb83b5b272` | `64a16328b81337c4f35b503ea7976fa8bef075a294841d331c10436b3b72eee3` |

## K=280 quadrature-support results

| geometry | train/audit | risk | train action | audit action | discrepancy | |g| | weak | energy | moment | rank frac. | min eig. | max eig. | condition | range | stationarity | identity | certified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| law | 8192/4096 | 5.186549474 | 0.380125417 | 0.387570754 | 0.0192 | 3.61279 | 0.07958 | 0.12531 | 0.01207 | 0.99643 | 1.651e-11 | 1.141e+01 | 5.211e+11 | 6.40e-09 | 6.40e-09 | 1.26e-11 | no |
| law | 16384/8192 | 5.186549474 | 0.370721152 | 0.374007167 | 0.0088 | 3.27370 | 0.05397 | 0.07053 | 0.00644 | 0.99643 | 2.202e-11 | 1.143e+01 | 3.911e+11 | 1.70e-09 | 1.70e-09 | 3.08e-11 | yes |
| law | 16384/16384 | 5.186549474 | 0.370721152 | 0.371682726 | 0.0026 | 3.27370 | 0.04460 | 0.06401 | 0.00555 | 0.99643 | 2.202e-11 | 1.143e+01 | 3.911e+11 | 1.70e-09 | 1.70e-09 | 2.08e-11 | yes |
| law | 32768/16384 | 5.186549474 | 0.358107182 | 0.356185022 | 0.0054 | 2.97159 | 0.05785 | 0.03770 | 0.00518 | 0.99643 | 2.283e-11 | 1.144e+01 | 3.782e+11 | 2.12e-09 | 2.12e-09 | 1.44e-11 | no |
| historical_0p5 | 8192/4096 | 5.203174625 | 0.347736242 | 0.354142961 | 0.0181 | 3.18687 | 0.07781 | 0.12447 | 0.01188 | 0.99643 | 1.655e-11 | 1.132e+01 | 5.200e+11 | 7.01e-09 | 7.01e-09 | 5.57e-12 | no |
| historical_0p5 | 16384/8192 | 5.203174625 | 0.340728533 | 0.343289232 | 0.0075 | 2.91840 | 0.05299 | 0.07043 | 0.00604 | 0.99643 | 2.207e-11 | 1.133e+01 | 3.902e+11 | 1.76e-09 | 1.76e-09 | 1.24e-11 | yes |
| historical_0p5 | 16384/16384 | 5.203174625 | 0.340728533 | 0.341003571 | 0.0008 | 2.91840 | 0.04191 | 0.07666 | 0.00556 | 0.99643 | 2.207e-11 | 1.133e+01 | 3.902e+11 | 1.76e-09 | 1.76e-09 | 8.48e-12 | yes |
| historical_0p5 | 32768/16384 | 5.203174625 | 0.330862516 | 0.329048853 | 0.0055 | 2.71633 | 0.05359 | 0.04747 | 0.00451 | 0.99643 | 2.288e-11 | 1.135e+01 | 3.775e+11 | 2.09e-09 | 2.09e-09 | 6.62e-12 | yes |
| historical_1 | 8192/4096 | 5.225761943 | 0.327625120 | 0.334185356 | 0.0196 | 3.34889 | 0.08225 | 0.17277 | 0.01333 | 0.99643 | 1.662e-11 | 1.129e+01 | 5.176e+11 | 6.69e-09 | 6.69e-09 | 1.77e-11 | no |
| historical_1 | 16384/8192 | 5.225761943 | 0.322327734 | 0.325889160 | 0.0109 | 3.16364 | 0.05490 | 0.10837 | 0.00622 | 0.99643 | 2.215e-11 | 1.131e+01 | 3.887e+11 | 2.25e-09 | 2.25e-09 | 1.18e-11 | no |
| historical_1 | 16384/16384 | 5.225761943 | 0.322327734 | 0.324190616 | 0.0057 | 3.16364 | 0.04018 | 0.09651 | 0.00621 | 0.99643 | 2.215e-11 | 1.131e+01 | 3.887e+11 | 2.25e-09 | 2.25e-09 | 1.21e-11 | no |
| historical_1 | 32768/16384 | 5.225761943 | 0.310262172 | 0.309941318 | 0.0010 | 2.78633 | 0.05076 | 0.05386 | 0.00474 | 0.99643 | 2.300e-11 | 1.132e+01 | 3.754e+11 | 2.49e-09 | 2.49e-09 | 5.16e-12 | no |
| historical_2 | 8192/4096 | 5.284504645 | 0.312296518 | 0.320682091 | 0.0261 | 3.29910 | 0.07568 | 0.25658 | 0.01405 | 0.99643 | 1.669e-11 | 1.124e+01 | 5.154e+11 | 6.39e-09 | 6.39e-09 | 1.27e-11 | no |
| historical_2 | 16384/8192 | 5.284504645 | 0.305552531 | 0.310172147 | 0.0149 | 3.15257 | 0.04838 | 0.16026 | 0.00778 | 0.99643 | 2.219e-11 | 1.126e+01 | 3.879e+11 | 2.56e-09 | 2.56e-09 | 6.62e-12 | no |
| historical_2 | 16384/16384 | 5.284504645 | 0.305552531 | 0.308847957 | 0.0107 | 3.15257 | 0.04482 | 0.12552 | 0.00749 | 0.99643 | 2.219e-11 | 1.126e+01 | 3.879e+11 | 2.56e-09 | 2.56e-09 | 5.74e-12 | no |
| historical_2 | 32768/16384 | 5.284504645 | 0.293855577 | 0.295030601 | 0.0040 | 2.68554 | 0.05255 | 0.06970 | 0.00501 | 0.99643 | 2.318e-11 | 1.127e+01 | 3.723e+11 | 2.75e-09 | 2.75e-09 | 3.80e-12 | no |
| eta0_3pct | 8192/4096 | 5.340106051 | 0.282785110 | 0.287452605 | 0.0162 | 2.92032 | 0.06583 | 0.19298 | 0.01146 | 0.99643 | 1.666e-11 | 1.119e+01 | 5.160e+11 | 6.52e-09 | 6.52e-09 | 2.24e-11 | no |
| eta0_3pct | 16384/8192 | 5.340106051 | 0.281069141 | 0.283586805 | 0.0089 | 2.88689 | 0.04202 | 0.14126 | 0.00616 | 0.99643 | 2.220e-11 | 1.121e+01 | 3.877e+11 | 2.56e-09 | 2.56e-09 | 2.73e-13 | no |
| eta0_3pct | 16384/16384 | 5.340106051 | 0.281069141 | 0.282233598 | 0.0041 | 2.88689 | 0.03614 | 0.12249 | 0.00594 | 0.99643 | 2.220e-11 | 1.121e+01 | 3.877e+11 | 2.56e-09 | 2.56e-09 | 1.82e-11 | no |
| eta0_3pct | 32768/16384 | 5.340106051 | 0.272329758 | 0.271998470 | 0.0012 | 2.56951 | 0.03767 | 0.06301 | 0.00338 | 0.99643 | 2.316e-11 | 1.122e+01 | 3.725e+11 | 2.75e-09 | 2.75e-09 | 7.04e-12 | yes |
| eta_grad_3pct | 8192/4096 | 5.342099811 | 0.282131712 | 0.286771879 | 0.0162 | 2.91522 | 0.06547 | 0.19194 | 0.01148 | 0.99643 | 1.667e-11 | 1.119e+01 | 5.160e+11 | 6.53e-09 | 6.53e-09 | 9.06e-12 | no |
| eta_grad_3pct | 16384/8192 | 5.342099811 | 0.280405226 | 0.282906746 | 0.0088 | 2.88225 | 0.04181 | 0.14040 | 0.00616 | 0.99643 | 2.220e-11 | 1.121e+01 | 3.876e+11 | 2.56e-09 | 2.56e-09 | 2.97e-12 | no |
| eta_grad_3pct | 16384/16384 | 5.342099811 | 0.280405226 | 0.281558392 | 0.0041 | 2.88225 | 0.03593 | 0.12178 | 0.00594 | 0.99643 | 2.220e-11 | 1.121e+01 | 3.876e+11 | 2.56e-09 | 2.56e-09 | 1.02e-11 | no |
| eta_grad_3pct | 32768/16384 | 5.342099811 | 0.271747898 | 0.271414749 | 0.0012 | 2.56716 | 0.03755 | 0.06279 | 0.00339 | 0.99643 | 2.316e-11 | 1.122e+01 | 3.726e+11 | 2.75e-09 | 2.75e-09 | 8.26e-12 | yes |

Projection, minimum ESS, pre-centering forcing mean, covariance conditioning, symmetry, gauge, and complete per-time rank/eigenvalue vectors are retained in each machine-readable case JSON. The table shows their compact controlling extrema.

## Action and gradient convergence

| geometry | support transition | train action change | audit action change | gradient cosine | gradient rel. change | energy change |
|---|---|---:|---:|---:|---:|---:|
| law | 8192/4096→16384/8192 | 0.02537 | 0.03627 | 0.9991728 | 0.11205 | -0.05478 |
| law | 16384/8192→16384/16384 | 0.00000 | 0.00625 | 1.0000000 | 0.00000 | -0.00653 |
| law | 16384/16384→32768/16384 | 0.03522 | 0.04351 | 0.9952774 | 0.14402 | -0.02631 |
| historical_0p5 | 8192/4096→16384/8192 | 0.02057 | 0.03162 | 0.9991234 | 0.10187 | -0.05404 |
| historical_0p5 | 16384/8192→16384/16384 | 0.00000 | 0.00670 | 1.0000000 | 0.00000 | +0.00623 |
| historical_0p5 | 16384/16384→32768/16384 | 0.02982 | 0.03633 | 0.9963538 | 0.11562 | -0.02919 |
| historical_1 | 8192/4096→16384/8192 | 0.01643 | 0.02546 | 0.9986573 | 0.07919 | -0.06440 |
| historical_1 | 16384/8192→16384/16384 | 0.00000 | 0.00524 | 1.0000000 | 0.00000 | -0.01186 |
| historical_1 | 16384/16384→32768/16384 | 0.03889 | 0.04597 | 0.9967126 | 0.16063 | -0.04265 |
| historical_2 | 8192/4096→16384/8192 | 0.02207 | 0.03388 | 0.9964577 | 0.09785 | -0.09633 |
| historical_2 | 16384/8192→16384/16384 | 0.00000 | 0.00429 | 1.0000000 | 0.00000 | -0.03473 |
| historical_2 | 16384/16384→32768/16384 | 0.03981 | 0.04683 | 0.9982033 | 0.18564 | -0.05583 |
| eta0_3pct | 8192/4096→16384/8192 | 0.00611 | 0.01363 | 0.9975040 | 0.07200 | -0.05172 |
| eta0_3pct | 16384/8192→16384/16384 | 0.00000 | 0.00479 | 1.0000000 | 0.00000 | -0.01876 |
| eta0_3pct | 16384/16384→32768/16384 | 0.03209 | 0.03763 | 0.9987029 | 0.13480 | -0.05949 |
| eta_grad_3pct | 8192/4096→16384/8192 | 0.00616 | 0.01366 | 0.9975146 | 0.07182 | -0.05154 |
| eta_grad_3pct | 16384/8192→16384/16384 | 0.00000 | 0.00479 | 1.0000000 | 0.00000 | -0.01862 |
| eta_grad_3pct | 16384/16384→32768/16384 | 0.03186 | 0.03737 | 0.9987104 | 0.13402 | -0.05899 |

## Low-risk energy-residual conclusion

- `law` across supports: 0.125314, 0.070534, 0.064008, 0.037702.
- `historical_0p5` across supports: 0.124469, 0.070432, 0.076663, 0.047474.

The unchanged physical threshold was **0.08** at every stage. The primary decision logic therefore yields **B. K280 REMAINS PHYSICALLY INADEQUATE AT LOW RISK**.

## Conditional K/rank study

Because the primary outcome was B, the predeclared conditional study ran on the same largest 32,768/16,384 support. K prefixes are exact globally ordered prefixes, and only rank tolerances 1e-10, 1e-11, and 1e-12 were tested.

### law

| K | rank tol. | train action | audit action | |g| | weak | energy | moment | min rank frac. | min retained eig. | condition | complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 120 | 1e-10 | 0.312994309 | 0.311194311 | 2.50126 | 0.05842 | 0.02951 | 0.00419 | 0.99167 | 7.390e-10 | 8.946e+09 | no |
| 120 | 1e-11 | 0.312994341 | 0.311194385 | 2.50126 | 0.05842 | 0.02951 | 0.00419 | 0.99167 | 6.518e-11 | 9.244e+10 | no |
| 120 | 1e-12 | 0.312994342 | 0.311194391 | 2.50126 | 0.05842 | 0.02951 | 0.00419 | 1.00000 | 2.109e-11 | 3.134e+11 | no |
| 160 | 1e-10 | 0.327239631 | 0.325240663 | 2.66531 | 0.05837 | 0.02694 | 0.00435 | 0.98750 | 1.006e-09 | 8.533e+09 | no |
| 160 | 1e-11 | 0.327243594 | 0.325256803 | 2.66545 | 0.05834 | 0.02707 | 0.00436 | 0.99375 | 8.451e-11 | 8.431e+10 | no |
| 160 | 1e-12 | 0.327243615 | 0.325256836 | 2.66545 | 0.05834 | 0.02707 | 0.00436 | 0.99375 | 2.285e-11 | 3.478e+11 | no |
| 200 | 1e-10 | 0.343905749 | 0.341934459 | 2.82496 | 0.05928 | 0.03180 | 0.00458 | 0.99000 | 1.006e-09 | 8.613e+09 | no |
| 200 | 1e-11 | 0.343911252 | 0.341953050 | 2.82511 | 0.05925 | 0.03195 | 0.00459 | 0.99500 | 8.449e-11 | 8.528e+10 | no |
| 200 | 1e-12 | 0.343911271 | 0.341953079 | 2.82511 | 0.05925 | 0.03195 | 0.00459 | 0.99500 | 2.284e-11 | 3.520e+11 | no |
| 240 | 1e-10 | 0.350832521 | 0.349003966 | 2.91043 | 0.05817 | 0.03522 | 0.00484 | 0.99167 | 1.006e-09 | 8.737e+09 | no |
| 240 | 1e-11 | 0.350838682 | 0.349023324 | 2.91058 | 0.05815 | 0.03535 | 0.00484 | 0.99583 | 8.447e-11 | 8.649e+10 | no |
| 240 | 1e-12 | 0.350838699 | 0.349023350 | 2.91058 | 0.05815 | 0.03535 | 0.00484 | 0.99583 | 2.284e-11 | 3.578e+11 | no |
| 280 | 1e-10 | 0.358101752 | 0.356168567 | 2.97146 | 0.05787 | 0.03760 | 0.00517 | 0.99286 | 1.003e-09 | 9.365e+09 | no |
| 280 | 1e-11 | 0.358107166 | 0.356184996 | 2.97159 | 0.05785 | 0.03770 | 0.00518 | 0.99643 | 8.445e-11 | 9.029e+10 | no |
| 280 | 1e-12 | 0.358107182 | 0.356185022 | 2.97159 | 0.05785 | 0.03770 | 0.00518 | 0.99643 | 2.283e-11 | 3.782e+11 | no |

### historical_0p5

| K | rank tol. | train action | audit action | |g| | weak | energy | moment | min rank frac. | min retained eig. | condition | complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 120 | 1e-10 | 0.290614820 | 0.289020875 | 2.32385 | 0.05379 | 0.03913 | 0.00374 | 0.99167 | 7.388e-10 | 8.947e+09 | no |
| 120 | 1e-11 | 0.290614853 | 0.289020950 | 2.32385 | 0.05379 | 0.03913 | 0.00374 | 0.99167 | 6.533e-11 | 9.229e+10 | yes |
| 120 | 1e-12 | 0.290614853 | 0.289020956 | 2.32385 | 0.05379 | 0.03913 | 0.00374 | 1.00000 | 2.109e-11 | 3.135e+11 | yes |
| 160 | 1e-10 | 0.303350822 | 0.301599278 | 2.45782 | 0.05388 | 0.03731 | 0.00395 | 0.98750 | 1.006e-09 | 8.533e+09 | no |
| 160 | 1e-11 | 0.303354132 | 0.301612800 | 2.45794 | 0.05386 | 0.03746 | 0.00396 | 0.99375 | 8.468e-11 | 8.419e+10 | yes |
| 160 | 1e-12 | 0.303354152 | 0.301612833 | 2.45794 | 0.05386 | 0.03746 | 0.00396 | 0.99375 | 2.289e-11 | 3.472e+11 | yes |
| 200 | 1e-10 | 0.318230990 | 0.316482360 | 2.59113 | 0.05497 | 0.04213 | 0.00409 | 0.99000 | 1.006e-09 | 8.614e+09 | no |
| 200 | 1e-11 | 0.318235652 | 0.316498343 | 2.59125 | 0.05496 | 0.04227 | 0.00410 | 0.99500 | 8.467e-11 | 8.514e+10 | yes |
| 200 | 1e-12 | 0.318235671 | 0.316498372 | 2.59125 | 0.05496 | 0.04227 | 0.00410 | 0.99500 | 2.289e-11 | 3.514e+11 | yes |
| 240 | 1e-10 | 0.324315979 | 0.322663786 | 2.66466 | 0.05383 | 0.04506 | 0.00430 | 0.99167 | 1.006e-09 | 8.738e+09 | no |
| 240 | 1e-11 | 0.324321231 | 0.322680780 | 2.66480 | 0.05381 | 0.04519 | 0.00431 | 0.99583 | 8.465e-11 | 8.636e+10 | yes |
| 240 | 1e-12 | 0.324321249 | 0.322680807 | 2.66480 | 0.05381 | 0.04519 | 0.00431 | 0.99583 | 2.288e-11 | 3.572e+11 | yes |
| 280 | 1e-10 | 0.330857821 | 0.329034179 | 2.71621 | 0.05361 | 0.04738 | 0.00450 | 0.99286 | 1.002e-09 | 9.366e+09 | no |
| 280 | 1e-11 | 0.330862499 | 0.329048826 | 2.71633 | 0.05359 | 0.04747 | 0.00451 | 0.99643 | 8.462e-11 | 9.015e+10 | yes |
| 280 | 1e-12 | 0.330862516 | 0.329048853 | 2.71633 | 0.05359 | 0.04747 | 0.00451 | 0.99643 | 2.288e-11 | 3.775e+11 | yes |

### historical_1

| K | rank tol. | train action | audit action | |g| | weak | energy | moment | min rank frac. | min retained eig. | condition | complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 120 | 1e-10 | 0.271028302 | 0.270261827 | 2.36486 | 0.05246 | 0.04771 | 0.00462 | 0.99167 | 7.391e-10 | 8.944e+09 | no |
| 120 | 1e-11 | 0.271028334 | 0.270261894 | 2.36486 | 0.05246 | 0.04771 | 0.00462 | 0.99167 | 6.573e-11 | 9.221e+10 | no |
| 120 | 1e-12 | 0.271028335 | 0.270261904 | 2.36486 | 0.05246 | 0.04771 | 0.00462 | 1.00000 | 2.110e-11 | 3.133e+11 | no |
| 160 | 1e-10 | 0.284182952 | 0.283473746 | 2.51468 | 0.05149 | 0.04703 | 0.00418 | 0.98750 | 1.006e-09 | 8.532e+09 | no |
| 160 | 1e-11 | 0.284184471 | 0.283478154 | 2.51474 | 0.05148 | 0.04708 | 0.00418 | 0.99375 | 8.464e-11 | 8.397e+10 | no |
| 160 | 1e-12 | 0.284184487 | 0.283478186 | 2.51474 | 0.05148 | 0.04708 | 0.00418 | 0.99375 | 2.302e-11 | 3.454e+11 | no |
| 200 | 1e-10 | 0.298279571 | 0.297935349 | 2.65915 | 0.05202 | 0.04985 | 0.00437 | 0.99000 | 1.006e-09 | 8.612e+09 | no |
| 200 | 1e-11 | 0.298282031 | 0.297941395 | 2.65922 | 0.05201 | 0.04990 | 0.00437 | 0.99500 | 8.463e-11 | 8.497e+10 | no |
| 200 | 1e-12 | 0.298282048 | 0.297941426 | 2.65922 | 0.05201 | 0.04990 | 0.00437 | 0.99500 | 2.301e-11 | 3.496e+11 | no |
| 240 | 1e-10 | 0.303818753 | 0.303672488 | 2.73000 | 0.05123 | 0.05243 | 0.00454 | 0.99167 | 1.006e-09 | 8.736e+09 | no |
| 240 | 1e-11 | 0.303821722 | 0.303679848 | 2.73008 | 0.05122 | 0.05248 | 0.00454 | 0.99583 | 8.461e-11 | 8.610e+10 | no |
| 240 | 1e-12 | 0.303821737 | 0.303679876 | 2.73008 | 0.05122 | 0.05248 | 0.00454 | 0.99583 | 2.301e-11 | 3.552e+11 | no |
| 280 | 1e-10 | 0.310259660 | 0.309935214 | 2.78626 | 0.05078 | 0.05382 | 0.00473 | 0.99286 | 1.003e-09 | 9.364e+09 | no |
| 280 | 1e-11 | 0.310262158 | 0.309941290 | 2.78633 | 0.05076 | 0.05386 | 0.00474 | 0.99643 | 8.458e-11 | 8.993e+10 | no |
| 280 | 1e-12 | 0.310262172 | 0.309941318 | 2.78633 | 0.05076 | 0.05386 | 0.00474 | 0.99643 | 2.300e-11 | 3.754e+11 | no |

### historical_2

| K | rank tol. | train action | audit action | |g| | weak | energy | moment | min rank frac. | min retained eig. | condition | complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 120 | 1e-10 | 0.255448774 | 0.255503683 | 2.26556 | 0.05534 | 0.06040 | 0.00526 | 0.99167 | 7.391e-10 | 8.943e+09 | no |
| 120 | 1e-11 | 0.255448801 | 0.255503738 | 2.26556 | 0.05534 | 0.06040 | 0.00526 | 0.99167 | 6.632e-11 | 9.185e+10 | no |
| 120 | 1e-12 | 0.255448804 | 0.255503752 | 2.26556 | 0.05534 | 0.06040 | 0.00526 | 1.00000 | 2.110e-11 | 3.132e+11 | no |
| 160 | 1e-10 | 0.269073464 | 0.269557581 | 2.41689 | 0.05320 | 0.06379 | 0.00522 | 0.98750 | 1.007e-09 | 8.532e+09 | no |
| 160 | 1e-11 | 0.269074636 | 0.269558686 | 2.41692 | 0.05320 | 0.06379 | 0.00521 | 0.99375 | 8.529e-11 | 8.325e+10 | no |
| 160 | 1e-12 | 0.269074645 | 0.269558710 | 2.41692 | 0.05320 | 0.06379 | 0.00521 | 0.99375 | 2.320e-11 | 3.428e+11 | no |
| 200 | 1e-10 | 0.282452609 | 0.283476821 | 2.56513 | 0.05354 | 0.06555 | 0.00470 | 0.99000 | 1.006e-09 | 8.612e+09 | no |
| 200 | 1e-11 | 0.282454137 | 0.283477804 | 2.56516 | 0.05354 | 0.06556 | 0.00470 | 0.99500 | 8.527e-11 | 8.430e+10 | no |
| 200 | 1e-12 | 0.282454148 | 0.283477828 | 2.56516 | 0.05354 | 0.06556 | 0.00470 | 0.99500 | 2.320e-11 | 3.469e+11 | no |
| 240 | 1e-10 | 0.287631710 | 0.288937721 | 2.62973 | 0.05322 | 0.06881 | 0.00480 | 0.99167 | 1.006e-09 | 8.736e+09 | no |
| 240 | 1e-11 | 0.287633487 | 0.288939705 | 2.62977 | 0.05321 | 0.06882 | 0.00480 | 0.99583 | 8.526e-11 | 8.535e+10 | no |
| 240 | 1e-12 | 0.287633496 | 0.288939726 | 2.62977 | 0.05321 | 0.06882 | 0.00480 | 0.99583 | 2.319e-11 | 3.523e+11 | no |
| 280 | 1e-10 | 0.293854234 | 0.295029388 | 2.68550 | 0.05256 | 0.06969 | 0.00501 | 0.99286 | 1.003e-09 | 9.364e+09 | no |
| 280 | 1e-11 | 0.293855568 | 0.295030580 | 2.68554 | 0.05255 | 0.06970 | 0.00501 | 0.99643 | 8.523e-11 | 8.920e+10 | no |
| 280 | 1e-12 | 0.293855577 | 0.295030601 | 2.68554 | 0.05255 | 0.06970 | 0.00501 | 0.99643 | 2.318e-11 | 3.723e+11 | no |

### eta0_3pct

| K | rank tol. | train action | audit action | |g| | weak | energy | moment | min rank frac. | min retained eig. | condition | complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 120 | 1e-10 | 0.238342713 | 0.237557051 | 2.21033 | 0.03892 | 0.05977 | 0.00378 | 0.99167 | 7.391e-10 | 8.943e+09 | no |
| 120 | 1e-11 | 0.238342740 | 0.237557104 | 2.21033 | 0.03892 | 0.05977 | 0.00378 | 0.99167 | 6.626e-11 | 9.185e+10 | yes |
| 120 | 1e-12 | 0.238342743 | 0.237557119 | 2.21033 | 0.03892 | 0.05977 | 0.00378 | 1.00000 | 2.110e-11 | 3.132e+11 | yes |
| 160 | 1e-10 | 0.250320035 | 0.249734459 | 2.34278 | 0.03739 | 0.06112 | 0.00354 | 0.98750 | 1.006e-09 | 8.531e+09 | no |
| 160 | 1e-11 | 0.250321208 | 0.249736600 | 2.34281 | 0.03739 | 0.06113 | 0.00353 | 0.99375 | 8.498e-11 | 8.356e+10 | yes |
| 160 | 1e-12 | 0.250321219 | 0.249736625 | 2.34281 | 0.03739 | 0.06113 | 0.00353 | 0.99375 | 2.318e-11 | 3.430e+11 | yes |
| 200 | 1e-10 | 0.262193770 | 0.261937427 | 2.46609 | 0.03824 | 0.06125 | 0.00292 | 0.99000 | 1.006e-09 | 8.612e+09 | no |
| 200 | 1e-11 | 0.262195384 | 0.261939794 | 2.46613 | 0.03824 | 0.06127 | 0.00292 | 0.99500 | 8.497e-11 | 8.459e+10 | yes |
| 200 | 1e-12 | 0.262195397 | 0.261939819 | 2.46613 | 0.03824 | 0.06127 | 0.00292 | 0.99500 | 2.318e-11 | 3.471e+11 | yes |
| 240 | 1e-10 | 0.266705200 | 0.266585273 | 2.52048 | 0.03788 | 0.06331 | 0.00307 | 0.99167 | 1.006e-09 | 8.736e+09 | no |
| 240 | 1e-11 | 0.266707092 | 0.266588827 | 2.52053 | 0.03788 | 0.06333 | 0.00306 | 0.99583 | 8.495e-11 | 8.567e+10 | yes |
| 240 | 1e-12 | 0.266707103 | 0.266588850 | 2.52053 | 0.03788 | 0.06333 | 0.00306 | 0.99583 | 2.317e-11 | 3.525e+11 | yes |
| 280 | 1e-10 | 0.272328186 | 0.271995456 | 2.56947 | 0.03767 | 0.06299 | 0.00339 | 0.99286 | 1.003e-09 | 9.364e+09 | no |
| 280 | 1e-11 | 0.272329747 | 0.271998448 | 2.56951 | 0.03767 | 0.06301 | 0.00338 | 0.99643 | 8.492e-11 | 8.952e+10 | yes |
| 280 | 1e-12 | 0.272329758 | 0.271998470 | 2.56951 | 0.03767 | 0.06301 | 0.00338 | 0.99643 | 2.316e-11 | 3.726e+11 | yes |

### eta_grad_3pct

| K | rank tol. | train action | audit action | |g| | weak | energy | moment | min rank frac. | min retained eig. | condition | complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 120 | 1e-10 | 0.237850715 | 0.237066176 | 2.20865 | 0.03878 | 0.05962 | 0.00379 | 0.99167 | 7.391e-10 | 8.943e+09 | no |
| 120 | 1e-11 | 0.237850743 | 0.237066230 | 2.20865 | 0.03878 | 0.05962 | 0.00379 | 0.99167 | 6.625e-11 | 9.186e+10 | yes |
| 120 | 1e-12 | 0.237850746 | 0.237066245 | 2.20865 | 0.03878 | 0.05962 | 0.00379 | 1.00000 | 2.110e-11 | 3.132e+11 | yes |
| 160 | 1e-10 | 0.249794354 | 0.249207350 | 2.34099 | 0.03727 | 0.06090 | 0.00355 | 0.98750 | 1.006e-09 | 8.531e+09 | no |
| 160 | 1e-11 | 0.249795515 | 0.249209464 | 2.34102 | 0.03727 | 0.06091 | 0.00354 | 0.99375 | 8.497e-11 | 8.357e+10 | yes |
| 160 | 1e-12 | 0.249795527 | 0.249209489 | 2.34102 | 0.03727 | 0.06091 | 0.00354 | 0.99375 | 2.318e-11 | 3.430e+11 | yes |
| 200 | 1e-10 | 0.261638308 | 0.261379463 | 2.46398 | 0.03812 | 0.06103 | 0.00292 | 0.99000 | 1.006e-09 | 8.612e+09 | no |
| 200 | 1e-11 | 0.261639913 | 0.261381811 | 2.46401 | 0.03812 | 0.06105 | 0.00292 | 0.99500 | 8.496e-11 | 8.460e+10 | yes |
| 200 | 1e-12 | 0.261639926 | 0.261381836 | 2.46401 | 0.03812 | 0.06105 | 0.00292 | 0.99500 | 2.317e-11 | 3.471e+11 | yes |
| 240 | 1e-10 | 0.266136084 | 0.266013789 | 2.51816 | 0.03776 | 0.06307 | 0.00308 | 0.99167 | 1.006e-09 | 8.736e+09 | no |
| 240 | 1e-11 | 0.266137963 | 0.266017315 | 2.51820 | 0.03775 | 0.06309 | 0.00307 | 0.99583 | 8.494e-11 | 8.568e+10 | yes |
| 240 | 1e-12 | 0.266137974 | 0.266017338 | 2.51820 | 0.03775 | 0.06309 | 0.00307 | 0.99583 | 2.317e-11 | 3.525e+11 | yes |
| 280 | 1e-10 | 0.271746338 | 0.271411761 | 2.56711 | 0.03756 | 0.06278 | 0.00340 | 0.99286 | 1.003e-09 | 9.364e+09 | no |
| 280 | 1e-11 | 0.271747887 | 0.271414726 | 2.56716 | 0.03755 | 0.06279 | 0.00339 | 0.99643 | 8.491e-11 | 8.952e+10 | yes |
| 280 | 1e-12 | 0.271747898 | 0.271414749 | 2.56716 | 0.03755 | 0.06279 | 0.00339 | 0.99643 | 2.316e-11 | 3.726e+11 | yes |

The predeclared whole-range qualification selected K=`None` and rank tolerance `None`. Future discretization qualified: `no`.

## Candidate v2 initialization gate

A start may logically enter a future optimizer after exact risk, geometry, projection/ESS/forcing/covariance, Galerkin algebra, rank/range/stationarity pass, without already passing held-out weak/energy/moment certificates. This is safe only because every official endpoint, incumbent, and winner still requires the complete certificate. Both v1 low-risk anchors would have entered under this candidate rule, but neither could have become an endpoint in its observed state. This capability was implemented and unit-tested only; no eta step ran.

## Future v2 feasible-manifold start generator

| allowance | feasible starts | total unique pool | min pairwise eta distance | median pairwise eta distance |
|---:|---:|---:|---:|---:|
| 0.5% | 208 | 337 | 1.860e-05 | 6.330e-03 |
| 1% | 233 | 337 | 1.860e-05 | 1.095e-02 |
| 2% | 260 | 337 | 1.860e-05 | 1.465e-02 |
| 3% | 286 | 337 | 1.860e-05 | 2.142e-02 |
| 4% | 301 | 337 | 1.860e-05 | 2.538e-02 |
| 5% | 304 | 337 | 1.860e-05 | 2.578e-02 |

The generator uses selection risk only: Law-to-history interpolation, deterministic local clouds, risk-tangent perturbations, and a small fixed global component. It evaluated feasibility/diversity only and never evaluated or optimized Full action.

## Limitations and recommendation

- This is empirical finite-support qualification, not an infinite-dimensional convergence proof.
- Development audit banks are not independent validation and must never be reported as such.
- No certificate threshold was tuned; energy remained fixed at 0.08.
- No validation quantity influenced K, support, rank cutoff, basis, certificate, start generator, or future logic.

Exact next recommendation: do not freeze Pareto v2; develop a more physically adequate fixed-feature Full discretization before any new sweep.

## Verification and repository audit

`git diff --check` return code at report generation: `0`. The complete isolated
regression suite passed: **131 tests in 97.161 seconds**. V1 byte-identity:
`True`. No validation access: `true`. No sensor optimization: `true`.

Final `git status --short` at report generation:

```text
 M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FINAL_3PCT_GALERKIN_CROSSCHECK.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_RESOLUTION_STUDY.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_RESOLUTION_STUDY_PROTOCOL.md
?? experiments/skyrmions_deep_ritz_full/OFFICIAL_GALERKIN_PARETO_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/OFFICIAL_GALERKIN_PARETO_PROTOCOL.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_common.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_report.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_run.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_selection.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_validation.py
?? experiments/skyrmions_deep_ritz_full/resolution_study.py
?? experiments/skyrmions_deep_ritz_full/resolution_study_report.py
?? experiments/skyrmions_deep_ritz_full/resolution_study_run.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/test_official_pareto.py
?? experiments/skyrmions_deep_ritz_full/test_resolution_study.py
```

All task-created paths are inside `experiments/skyrmions_deep_ritz_full/`; numerical outputs are confined to `outputs/galerkin_resolution_study/`. Historical output trees were not overwritten.

C. GALERKIN FULL DISCRETIZATION NOT YET QUALIFIED FOR A PARETO SWEEP

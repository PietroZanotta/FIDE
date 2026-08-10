# Gray–Scott Experiment C design report

This report contains design-split morphology discovery and shared-target endpoint calibration only. No learned MFSI or tangent rollout result was computed or used.

Global pooled threshold: `0.15398028`.

## Regime scan

| id | F | k | class | regime gate | minority components | Euler | interface | anisotropy | diversity |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| gs_01 | 0.0180 | 0.0510 | rejected_unstable | False | 0.5 | 0.344 | 0.0256 | 0.0393 | 0.0497 |
| gs_02 | 0.0220 | 0.0510 | rejected_unstable | False | 2.08 | -0.609 | 0.16 | 0.303 | 0.0969 |
| gs_03 | 0.0260 | 0.0550 | labyrinth_like | True | 1.78 | -0.422 | 0.215 | 0.287 | 0.0906 |
| gs_04 | 0.0300 | 0.0550 | ambiguous | True | 4.3 | -4.3 | 0.154 | 0.3 | 0.0669 |
| gs_05 | 0.0340 | 0.0580 | ambiguous | True | 2.36 | -1.77 | 0.213 | 0.332 | 0.0943 |
| gs_06 | 0.0380 | 0.0610 | labyrinth_like | True | 1.66 | 0.891 | 0.231 | 0.327 | 0.112 |
| gs_07 | 0.0420 | 0.0590 | spot_like | True | 4.62 | -4.59 | 0.151 | 0.24 | 0.0788 |
| gs_08 | 0.0460 | 0.0630 | labyrinth_like | True | 1.97 | 1.69 | 0.216 | 0.316 | 0.126 |
| gs_09 | 0.0540 | 0.0620 | labyrinth_like | True | 1.95 | -0.578 | 0.193 | 0.276 | 0.133 |
| gs_10 | 0.0620 | 0.0610 | rejected_unstable | False | 1.97 | -1.67 | 0.109 | 0.249 | 0.145 |
| gs_11 | 0.0700 | 0.0600 | rejected_unstable | False | 0.422 | -0.328 | 0.0131 | 0.137 | 0.155 |
| gs_12 | 0.0780 | 0.0610 | rejected_unstable | False | 0.672 | 0.672 | 0.0448 | 0.119 | 0.108 |
| gs_13 | 0.0220 | 0.0610 | rejected_unstable | False | 0.547 | 0.547 | 0.0163 | 0.00841 | 0.0406 |
| gs_14 | 0.0260 | 0.0610 | rejected_unstable | False | 5.91 | 5.91 | 0.17 | 0.0444 | 0.104 |
| gs_15 | 0.0300 | 0.0620 | spot_like | True | 6.56 | 6.56 | 0.194 | 0.0673 | 0.111 |
| gs_16 | 0.0340 | 0.0630 | spot_like | True | 6.67 | 6.67 | 0.204 | 0.0811 | 0.116 |
| gs_17 | 0.0380 | 0.0640 | spot_like | True | 6.31 | 6.31 | 0.201 | 0.0976 | 0.12 |
| gs_18 | 0.0420 | 0.0640 | ambiguous | True | 4.53 | 4.52 | 0.21 | 0.196 | 0.124 |
| gs_19 | 0.0460 | 0.0650 | ambiguous | True | 3.84 | 3.84 | 0.17 | 0.164 | 0.128 |
| gs_20 | 0.0500 | 0.0650 | rejected_unstable | False | 2.08 | 2.08 | 0.152 | 0.401 | 0.13 |
| grid_F0220_k0550 | 0.0220 | 0.0550 | rejected_unstable | False | 2.67 | 2.23 | 0.149 | 0.213 | 0.1 |
| grid_F0220_k0570 | 0.0220 | 0.0570 | rejected_unstable | False | 3.08 | 3.03 | 0.112 | 0.0683 | 0.0928 |
| grid_F0220_k0590 | 0.0220 | 0.0590 | rejected_unstable | False | 4.12 | 4.09 | 0.126 | 0.0382 | 0.096 |
| grid_F0220_k0630 | 0.0220 | 0.0630 | rejected_unstable | False | 0 | 0 | 0 | 0 | 0 |
| grid_F0220_k0640 | 0.0220 | 0.0640 | rejected_unstable | False | 0 | 0 | 0 | 0 | 0 |
| grid_F0220_k0650 | 0.0220 | 0.0650 | rejected_unstable | False | 0 | 0 | 0 | 0 | 0 |
| grid_F0260_k0570 | 0.0260 | 0.0570 | labyrinth_like | True | 2.44 | 1.38 | 0.211 | 0.359 | 0.1 |
| grid_F0260_k0590 | 0.0260 | 0.0590 | rejected_unstable | False | 5.7 | 5.7 | 0.188 | 0.0953 | 0.109 |
| grid_F0260_k0630 | 0.0260 | 0.0630 | rejected_unstable | False | 1.52 | 1.52 | 0.0455 | 0.0358 | 0.0682 |
| grid_F0260_k0640 | 0.0260 | 0.0640 | rejected_unstable | False | 0.281 | 0.281 | 0.00845 | 0.00269 | 0.0253 |
| grid_F0260_k0650 | 0.0260 | 0.0650 | rejected_unstable | False | 0.0312 | 0.0312 | 0.000885 | 4.97e-06 | 0.00371 |
| grid_F0300_k0570 | 0.0300 | 0.0570 | labyrinth_like | True | 1.81 | -0.359 | 0.225 | 0.302 | 0.0948 |
| grid_F0300_k0590 | 0.0300 | 0.0590 | labyrinth_like | True | 2.33 | 1.55 | 0.226 | 0.382 | 0.104 |
| grid_F0300_k0610 | 0.0300 | 0.0610 | spot_like | True | 6.8 | 6.78 | 0.213 | 0.0971 | 0.11 |
| grid_F0300_k0630 | 0.0300 | 0.0630 | spot_like | True | 5.8 | 5.78 | 0.17 | 0.0592 | 0.11 |
| grid_F0300_k0640 | 0.0300 | 0.0640 | rejected_unstable | False | 3.53 | 3.53 | 0.108 | 0.0576 | 0.0992 |
| grid_F0300_k0650 | 0.0300 | 0.0650 | rejected_unstable | False | 1.12 | 1.12 | 0.0331 | 0.0127 | 0.0593 |
| grid_F0340_k0550 | 0.0340 | 0.0550 | rejected_unstable | False | 0 | 0 | 0 | 0.000618 | 0.000783 |
| grid_F0340_k0570 | 0.0340 | 0.0570 | ambiguous | True | 4.02 | -4 | 0.179 | 0.339 | 0.0796 |
| grid_F0340_k0590 | 0.0340 | 0.0590 | rejected_unstable | False | 1.75 | -0.25 | 0.226 | 0.334 | 0.103 |
| grid_F0340_k0610 | 0.0340 | 0.0610 | ambiguous | True | 3.3 | 3.14 | 0.228 | 0.327 | 0.111 |
| grid_F0340_k0640 | 0.0340 | 0.0640 | spot_like | True | 6 | 6 | 0.177 | 0.0576 | 0.115 |
| grid_F0340_k0650 | 0.0340 | 0.0650 | spot_like | True | 4.2 | 4.2 | 0.128 | 0.0519 | 0.108 |
| grid_F0380_k0550 | 0.0380 | 0.0550 | rejected_unstable | False | 0 | 0 | 0 | 0.0018 | 3.49e-07 |
| grid_F0380_k0570 | 0.0380 | 0.0570 | rejected_unstable | False | 0 | 0 | 0 | 0.000161 | 0.000284 |
| grid_F0380_k0590 | 0.0380 | 0.0590 | ambiguous | True | 2.94 | -2.62 | 0.205 | 0.318 | 0.0958 |
| grid_F0380_k0630 | 0.0380 | 0.0630 | ambiguous | True | 4.98 | 4.95 | 0.221 | 0.202 | 0.119 |
| grid_F0380_k0650 | 0.0380 | 0.0650 | spot_like | True | 5.36 | 5.36 | 0.167 | 0.0738 | 0.118 |
| grid_F0420_k0550 | 0.0420 | 0.0550 | rejected_unstable | False | 0 | 0 | 0 | 0.000628 | 3.22e-07 |
| grid_F0420_k0570 | 0.0420 | 0.0570 | rejected_unstable | False | 0 | 0 | 0 | 0.00402 | 3.46e-07 |
| grid_F0420_k0610 | 0.0420 | 0.0610 | labyrinth_like | True | 1.78 | -0.531 | 0.223 | 0.325 | 0.112 |
| grid_F0420_k0630 | 0.0420 | 0.0630 | ambiguous | True | 3.08 | 2.98 | 0.225 | 0.324 | 0.121 |
| grid_F0420_k0650 | 0.0420 | 0.0650 | spot_like | True | 5.44 | 5.44 | 0.181 | 0.0874 | 0.124 |
| grid_F0460_k0550 | 0.0460 | 0.0550 | rejected_unstable | False | 0 | 0 | 0 | 0.000541 | 3.36e-07 |
| grid_F0460_k0570 | 0.0460 | 0.0570 | rejected_unstable | False | 0 | 0 | 0 | 0.000633 | 3.92e-07 |
| grid_F0460_k0590 | 0.0460 | 0.0590 | rejected_unstable | False | 0 | 0 | 0 | 0.00055 | 8.41e-07 |
| grid_F0460_k0610 | 0.0460 | 0.0610 | ambiguous | True | 2.56 | -1.75 | 0.21 | 0.366 | 0.112 |
| grid_F0460_k0640 | 0.0460 | 0.0640 | ambiguous | True | 2.88 | 2.8 | 0.201 | 0.298 | 0.129 |
| grid_F0260_k0580 | 0.0260 | 0.0580 | rejected_unstable | False | 3.86 | 3.5 | 0.18 | 0.243 | 0.107 |
| grid_F0260_k0585 | 0.0260 | 0.0585 | ambiguous | True | 5.06 | 4.98 | 0.192 | 0.17 | 0.108 |
| grid_F0260_k0595 | 0.0260 | 0.0595 | rejected_unstable | False | 6 | 6 | 0.191 | 0.0719 | 0.108 |
| grid_F0260_k0600 | 0.0260 | 0.0600 | rejected_unstable | False | 6.28 | 6.28 | 0.191 | 0.0621 | 0.106 |
| grid_F0260_k0605 | 0.0260 | 0.0605 | spot_like | True | 6.36 | 6.36 | 0.184 | 0.0501 | 0.104 |
| grid_F0260_k0615 | 0.0260 | 0.0615 | rejected_unstable | False | 4.91 | 4.91 | 0.14 | 0.0314 | 0.102 |
| grid_F0260_k0620 | 0.0260 | 0.0620 | rejected_unstable | False | 3.75 | 3.73 | 0.111 | 0.0399 | 0.0952 |
| grid_F0260_k0625 | 0.0260 | 0.0625 | rejected_unstable | False | 2.66 | 2.66 | 0.0799 | 0.0365 | 0.0867 |
| grid_F0280_k0580 | 0.0280 | 0.0580 | labyrinth_like | True | 2.17 | 1.03 | 0.223 | 0.404 | 0.101 |
| grid_F0280_k0585 | 0.0280 | 0.0585 | ambiguous | True | 2.67 | 1.91 | 0.211 | 0.337 | 0.105 |
| grid_F0280_k0590 | 0.0280 | 0.0590 | ambiguous | True | 4.41 | 4.08 | 0.22 | 0.307 | 0.104 |
| grid_F0280_k0595 | 0.0280 | 0.0595 | spot_like | True | 5.64 | 5.56 | 0.217 | 0.197 | 0.105 |
| grid_F0280_k0600 | 0.0280 | 0.0600 | spot_like | True | 6.69 | 6.67 | 0.211 | 0.114 | 0.107 |
| grid_F0280_k0605 | 0.0280 | 0.0605 | spot_like | True | 6.67 | 6.67 | 0.2 | 0.0783 | 0.108 |
| grid_F0280_k0610 | 0.0280 | 0.0610 | spot_like | True | 6.61 | 6.61 | 0.194 | 0.066 | 0.108 |
| grid_F0280_k0615 | 0.0280 | 0.0615 | spot_like | True | 6.41 | 6.41 | 0.186 | 0.0604 | 0.108 |
| grid_F0280_k0620 | 0.0280 | 0.0620 | rejected_unstable | False | 5.89 | 5.88 | 0.17 | 0.0509 | 0.107 |
| grid_F0280_k0625 | 0.0280 | 0.0625 | rejected_unstable | False | 4.83 | 4.83 | 0.141 | 0.0383 | 0.104 |
| grid_F0280_k0630 | 0.0280 | 0.0630 | rejected_unstable | False | 3.94 | 3.94 | 0.116 | 0.0357 | 0.099 |
| grid_F0300_k0580 | 0.0300 | 0.0580 | labyrinth_like | True | 1.78 | 0.312 | 0.228 | 0.335 | 0.1 |
| grid_F0300_k0585 | 0.0300 | 0.0585 | labyrinth_like | True | 1.83 | 0.719 | 0.226 | 0.367 | 0.102 |
| grid_F0300_k0595 | 0.0300 | 0.0595 | ambiguous | True | 2.95 | 2.55 | 0.225 | 0.342 | 0.105 |
| grid_F0300_k0600 | 0.0300 | 0.0600 | ambiguous | True | 4.53 | 4.41 | 0.223 | 0.295 | 0.107 |
| grid_F0300_k0605 | 0.0300 | 0.0605 | spot_like | True | 6.06 | 6.05 | 0.221 | 0.159 | 0.109 |
| grid_F0300_k0615 | 0.0300 | 0.0615 | spot_like | True | 6.73 | 6.73 | 0.203 | 0.0767 | 0.111 |
| grid_F0300_k0625 | 0.0300 | 0.0625 | spot_like | True | 6.25 | 6.25 | 0.182 | 0.0583 | 0.111 |
| grid_F0320_k0580 | 0.0320 | 0.0580 | labyrinth_like | True | 1.83 | -0.281 | 0.226 | 0.313 | 0.0987 |
| grid_F0320_k0585 | 0.0320 | 0.0585 | labyrinth_like | True | 1.69 | -0.0469 | 0.228 | 0.351 | 0.102 |
| grid_F0320_k0590 | 0.0320 | 0.0590 | labyrinth_like | True | 1.69 | 0.453 | 0.229 | 0.401 | 0.104 |
| grid_F0320_k0595 | 0.0320 | 0.0595 | labyrinth_like | True | 1.89 | 1.09 | 0.229 | 0.398 | 0.105 |
| grid_F0320_k0600 | 0.0320 | 0.0600 | labyrinth_like | True | 2.62 | 2.12 | 0.229 | 0.38 | 0.107 |
| grid_F0320_k0605 | 0.0320 | 0.0605 | ambiguous | True | 3.67 | 3.5 | 0.226 | 0.316 | 0.109 |
| grid_F0320_k0610 | 0.0320 | 0.0610 | ambiguous | True | 5.45 | 5.42 | 0.226 | 0.214 | 0.11 |
| grid_F0320_k0615 | 0.0320 | 0.0615 | spot_like | True | 6.67 | 6.67 | 0.22 | 0.112 | 0.112 |
| grid_F0320_k0620 | 0.0320 | 0.0620 | rejected_unstable | False | 6.7 | 6.69 | 0.211 | 0.0983 | 0.113 |
| grid_F0320_k0625 | 0.0320 | 0.0625 | spot_like | True | 6.64 | 6.64 | 0.2 | 0.0742 | 0.113 |
| grid_F0320_k0630 | 0.0320 | 0.0630 | spot_like | True | 6.42 | 6.42 | 0.19 | 0.0656 | 0.113 |
| grid_F0340_k0585 | 0.0340 | 0.0585 | rejected_unstable | False | 1.95 | -0.594 | 0.225 | 0.338 | 0.0995 |
| grid_F0340_k0595 | 0.0340 | 0.0595 | labyrinth_like | True | 1.52 | 0.25 | 0.229 | 0.394 | 0.105 |
| grid_F0340_k0600 | 0.0340 | 0.0600 | labyrinth_like | True | 1.78 | 1.02 | 0.231 | 0.384 | 0.108 |
| grid_F0340_k0605 | 0.0340 | 0.0605 | labyrinth_like | True | 2.23 | 1.8 | 0.23 | 0.39 | 0.109 |
| grid_F0340_k0615 | 0.0340 | 0.0615 | ambiguous | True | 4.8 | 4.78 | 0.228 | 0.245 | 0.112 |
| grid_F0340_k0620 | 0.0340 | 0.0620 | spot_like | True | 5.94 | 5.92 | 0.222 | 0.157 | 0.114 |
| grid_F0340_k0625 | 0.0340 | 0.0625 | rejected_unstable | False | 6.52 | 6.5 | 0.214 | 0.115 | 0.115 |
| grid_F0360_k0580 | 0.0360 | 0.0580 | ambiguous | True | 3.58 | -3.41 | 0.194 | 0.323 | 0.0876 |
| grid_F0360_k0585 | 0.0360 | 0.0585 | ambiguous | True | 2.73 | -2.36 | 0.208 | 0.322 | 0.0945 |
| grid_F0360_k0590 | 0.0360 | 0.0590 | labyrinth_like | True | 1.98 | -1.03 | 0.221 | 0.322 | 0.1 |
| grid_F0360_k0595 | 0.0360 | 0.0595 | labyrinth_like | True | 1.8 | -0.438 | 0.225 | 0.324 | 0.104 |
| grid_F0360_k0600 | 0.0360 | 0.0600 | labyrinth_like | True | 1.53 | 0.0938 | 0.229 | 0.38 | 0.107 |
| grid_F0360_k0605 | 0.0360 | 0.0605 | labyrinth_like | True | 1.67 | 0.875 | 0.231 | 0.351 | 0.11 |
| grid_F0360_k0610 | 0.0360 | 0.0610 | labyrinth_like | True | 2.11 | 1.59 | 0.23 | 0.366 | 0.111 |
| grid_F0360_k0615 | 0.0360 | 0.0615 | ambiguous | True | 3 | 2.81 | 0.228 | 0.341 | 0.113 |
| grid_F0360_k0620 | 0.0360 | 0.0620 | ambiguous | True | 4.28 | 4.27 | 0.228 | 0.259 | 0.114 |
| grid_F0360_k0625 | 0.0360 | 0.0625 | ambiguous | True | 5.31 | 5.31 | 0.224 | 0.194 | 0.116 |
| grid_F0360_k0630 | 0.0360 | 0.0630 | spot_like | True | 6.12 | 6.12 | 0.215 | 0.134 | 0.118 |

The class labels are empirical score strata, not assumed labels for `(F,k)` values. Visual confirmation is retained in the paginated `figures/regime_scan_montage*.png` files.

## Endpoint calibration

| pair | residual (std.) | min ESS | morphology effect | pass | reason |
|---|---:|---:|---:|---|---|
| grid_F0340_k0650__gs_09 | 1.8 | 0.0938 | 1.37 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__grid_F0260_k0570 | 1.8 | 0.0313 | 1.37 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__gs_03 | 1.8 | 0.0156 | 1.37 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__gs_08 | 1.8 | 0.0156 | 1.37 | False | calibration_residual;endpoint_ess |
| grid_F0420_k0650__gs_03 | 3.48e-08 | 0.0156 | 1.68e-08 | False | endpoint_ess;morphology_effect |
| grid_F0420_k0650__gs_09 | 7.93e-05 | 0.0156 | 4.67e-05 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0630__gs_08 | 0.0473 | 0.0156 | 0.0365 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0630__gs_09 | 0.0463 | 0.0162 | 0.0357 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0630__gs_03 | 0.0463 | 0.0156 | 0.0357 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0640__gs_08 | 0.0124 | 0.0156 | 0.00913 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0625__gs_08 | 0.00103 | 0.0156 | 0.000796 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0640__gs_03 | 4.67e-05 | 0.0156 | 3.51e-05 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0640__gs_09 | 5.3e-05 | 0.0156 | 3.98e-05 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0625__gs_09 | 0.000356 | 0.0156 | 0.000278 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0625__gs_03 | 0.000356 | 0.0156 | 0.000277 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_17__grid_F0340_k0605 | 0.21 | 0.0625 | 0.712 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0615__grid_F0300_k0580 | 0.291 | 0.145 | 1.13 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0610__grid_F0300_k0580 | 0.247 | 0.117 | 1.17 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0610__grid_F0300_k0570 | 0.339 | 0.111 | 1.25 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0320_k0590 | 0.286 | 0.044 | 0.969 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0620__grid_F0300_k0585 | 0.263 | 0.0438 | 0.618 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0615__grid_F0300_k0580 | 0.291 | 0.0505 | 1.25 | False | calibration_residual;endpoint_ess |
| grid_F0260_k0605__grid_F0360_k0590 | 0.517 | 0.0345 | 2.3 | False | calibration_residual;endpoint_ess |
| grid_F0260_k0605__grid_F0420_k0610 | 0.569 | 0.0207 | 1.84 | False | calibration_residual;endpoint_ess |
| grid_F0420_k0650__grid_F0360_k0610 | 0.305 | 0.0375 | 0.904 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0260_k0605__grid_F0300_k0570 | 0.443 | 0.0426 | 2.33 | False | calibration_residual;endpoint_ess |
| grid_F0380_k0650__grid_F0320_k0600 | 0.311 | 0.0223 | 0.634 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0260_k0605__grid_F0320_k0580 | 0.451 | 0.0412 | 2.27 | False | calibration_residual;endpoint_ess |
| grid_F0260_k0605__grid_F0360_k0595 | 0.475 | 0.03 | 2.22 | False | calibration_residual;endpoint_ess |
| gs_15__grid_F0300_k0590 | 0.233 | 0.0314 | 0.951 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0605__grid_F0300_k0580 | 0.194 | 0.0492 | 0.727 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_16__grid_F0300_k0580 | 0.309 | 0.0756 | 1.19 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0630__grid_F0280_k0580 | 0.201 | 0.0157 | 1.64 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0615__grid_F0300_k0590 | 0.185 | 0.0371 | 0.712 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0605__grid_F0300_k0590 | 0.103 | 0.0284 | 0.308 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0260_k0605__grid_F0320_k0585 | 0.413 | 0.0515 | 1.63 | False | calibration_residual;endpoint_ess |
| gs_15__grid_F0300_k0580 | 0.344 | 0.128 | 1.41 | False | calibration_residual;endpoint_ess |
| gs_16__grid_F0360_k0590 | 0.496 | 0.773 | 1.68 | False | calibration_residual |
| grid_F0320_k0625__grid_F0300_k0590 | 0.202 | 0.0416 | 0.723 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0625__grid_F0300_k0570 | 0.416 | 0.844 | 1.64 | False | calibration_residual |
| grid_F0300_k0610__grid_F0320_k0580 | 0.348 | 0.11 | 1.28 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0610__grid_F0320_k0585 | 0.302 | 0.103 | 1.23 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0360_k0590 | 0.49 | 0.772 | 1.65 | False | calibration_residual |
| grid_F0280_k0610__grid_F0360_k0590 | 0.525 | 0.098 | 1.67 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0625__grid_F0360_k0590 | 0.518 | 0.748 | 1.65 | False | calibration_residual |
| grid_F0380_k0650__gs_03 | 0.302 | 0.0223 | 0.758 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0610__grid_F0360_k0595 | 0.474 | 0.133 | 1.7 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0625__grid_F0320_k0580 | 0.425 | 0.755 | 1.62 | False | calibration_residual |
| grid_F0260_k0605__grid_F0300_k0580 | 0.361 | 0.0523 | 1.55 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0630__grid_F0360_k0590 | 0.567 | 0.737 | 1.62 | False | calibration_residual |
| gs_15__grid_F0300_k0570 | 0.447 | 0.818 | 1.58 | False | calibration_residual |
| grid_F0300_k0615__grid_F0360_k0590 | 0.504 | 0.755 | 1.61 | False | calibration_residual |
| grid_F0320_k0625__grid_F0320_k0590 | 0.337 | 0.752 | 1.6 | False | calibration_residual |
| grid_F0360_k0630__grid_F0320_k0590 | 0.275 | 0.0314 | 0.907 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0625__grid_F0340_k0595 | 0.378 | 0.699 | 1.62 | False | calibration_residual |
| grid_F0300_k0625__grid_F0300_k0570 | 0.505 | 0.842 | 1.55 | False | calibration_residual |
| grid_F0260_k0605__gs_06 | 0.415 | 0.0288 | 1.3 | False | calibration_residual;endpoint_ess |
| gs_16__grid_F0280_k0580 | 0.265 | 0.0168 | 1.3 | False | calibration_residual;endpoint_ess |
| grid_F0260_k0605__grid_F0360_k0600 | 0.433 | 0.0391 | 1.61 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0640__grid_F0360_k0590 | 0.62 | 0.757 | 1.57 | False | calibration_residual |
| gs_15__grid_F0360_k0590 | 0.548 | 0.705 | 1.6 | False | calibration_residual |
| grid_F0320_k0625__grid_F0320_k0585 | 0.38 | 0.688 | 1.6 | False | calibration_residual |
| gs_16__grid_F0340_k0595 | 0.354 | 0.696 | 1.6 | False | calibration_residual |
| grid_F0260_k0605__grid_F0340_k0595 | 0.402 | 0.046 | 1.6 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0615__grid_F0340_k0595 | 0.365 | 0.737 | 1.57 | False | calibration_residual |
| gs_16__grid_F0320_k0580 | 0.402 | 0.646 | 1.6 | False | calibration_residual |
| grid_F0300_k0625__grid_F0360_k0590 | 0.606 | 0.717 | 1.55 | False | calibration_residual |
| gs_15__grid_F0320_k0580 | 0.456 | 0.684 | 1.57 | False | calibration_residual |
| grid_F0300_k0615__grid_F0300_k0570 | 0.402 | 0.671 | 1.57 | False | calibration_residual |
| gs_16__grid_F0300_k0570 | 0.416 | 0.602 | 1.61 | False | calibration_residual |
| grid_F0340_k0640__grid_F0300_k0570 | 0.517 | 0.711 | 1.55 | False | calibration_residual |
| gs_15__grid_F0320_k0590 | 0.367 | 0.687 | 1.55 | False | calibration_residual |
| grid_F0300_k0625__grid_F0320_k0580 | 0.514 | 0.718 | 1.53 | False | calibration_residual |
| gs_16__grid_F0360_k0595 | 0.439 | 0.584 | 1.6 | False | calibration_residual |
| grid_F0300_k0615__grid_F0320_k0590 | 0.323 | 0.688 | 1.54 | False | calibration_residual |
| grid_F0340_k0640__grid_F0320_k0580 | 0.526 | 0.717 | 1.53 | False | calibration_residual |
| grid_F0360_k0630__grid_F0360_k0590 | 0.422 | 0.818 | 1.47 | False | calibration_residual |
| grid_F0320_k0625__grid_F0360_k0595 | 0.461 | 0.566 | 1.6 | False | calibration_residual |
| grid_F0280_k0610__grid_F0300_k0570 | 0.437 | 0.19 | 1.58 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0615__grid_F0320_k0580 | 0.412 | 0.651 | 1.55 | False | calibration_residual |
| gs_15__grid_F0340_k0595 | 0.408 | 0.623 | 1.56 | False | calibration_residual |
| grid_F0320_k0630__grid_F0340_k0595 | 0.425 | 0.587 | 1.58 | False | calibration_residual |
| grid_F0320_k0630__grid_F0320_k0580 | 0.473 | 0.548 | 1.59 | False | calibration_residual |
| grid_F0280_k0615__grid_F0300_k0570 | 0.485 | 0.647 | 1.54 | False | calibration_residual |
| grid_F0320_k0615__grid_F0360_k0590 | 0.416 | 0.737 | 1.49 | False | calibration_residual |
| grid_F0320_k0630__grid_F0300_k0570 | 0.464 | 0.514 | 1.6 | False | calibration_residual |
| grid_F0320_k0625__grid_F0320_k0595 | 0.293 | 0.739 | 1.49 | False | calibration_residual |
| grid_F0260_k0605__grid_F0300_k0585 | 0.31 | 0.0468 | 1.41 | False | calibration_residual;endpoint_ess |
| gs_15__grid_F0320_k0585 | 0.41 | 0.588 | 1.55 | False | calibration_residual |
| grid_F0300_k0615__grid_F0320_k0585 | 0.366 | 0.616 | 1.54 | False | calibration_residual |
| grid_F0320_k0615__grid_F0300_k0570 | 0.399 | 0.821 | 1.44 | False | calibration_residual |
| grid_F0320_k0630__grid_F0320_k0590 | 0.383 | 0.571 | 1.56 | False | calibration_residual |
| grid_F0300_k0615__grid_F0320_k0595 | 0.28 | 0.784 | 1.45 | False | calibration_residual |
| gs_16__grid_F0360_k0605 | 0.34 | 0.702 | 1.49 | False | calibration_residual |
| grid_F0300_k0625__grid_F0320_k0585 | 0.468 | 0.619 | 1.53 | False | calibration_residual |
| grid_F0420_k0650__grid_F0340_k0600 | 0.328 | 0.0628 | 1.08 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0615__grid_F0360_k0595 | 0.448 | 0.583 | 1.54 | False | calibration_residual |
| grid_F0320_k0630__grid_F0360_k0595 | 0.509 | 0.521 | 1.57 | False | calibration_residual |
| grid_F0320_k0625__grid_F0360_k0605 | 0.363 | 0.697 | 1.48 | False | calibration_residual |
| gs_17__grid_F0300_k0585 | 0.38 | 0.561 | 1.54 | False | calibration_residual |
| grid_F0340_k0640__grid_F0320_k0585 | 0.479 | 0.628 | 1.51 | False | calibration_residual |
| gs_16__gs_06 | 0.362 | 0.751 | 1.45 | False | calibration_residual |
| grid_F0300_k0630__grid_F0300_k0570 | 0.556 | 0.799 | 1.42 | False | calibration_residual |
| grid_F0280_k0615__grid_F0360_k0590 | 0.585 | 0.592 | 1.52 | False | calibration_residual |
| gs_16__grid_F0320_k0585 | 0.355 | 0.52 | 1.56 | False | calibration_residual |
| grid_F0320_k0615__grid_F0340_k0595 | 0.277 | 0.771 | 1.43 | False | calibration_residual |
| grid_F0340_k0640__grid_F0340_k0595 | 0.477 | 0.608 | 1.51 | False | calibration_residual |
| grid_F0280_k0605__grid_F0360_k0590 | 0.492 | 0.151 | 1.5 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0615__grid_F0360_k0605 | 0.35 | 0.723 | 1.45 | False | calibration_residual |
| grid_F0280_k0610__grid_F0320_k0580 | 0.447 | 0.26 | 1.56 | False | calibration_residual |
| grid_F0300_k0625__grid_F0320_k0590 | 0.424 | 0.567 | 1.52 | False | calibration_residual |
| grid_F0300_k0625__grid_F0340_k0595 | 0.465 | 0.545 | 1.53 | False | calibration_residual |
| grid_F0280_k0605__gs_03 | 0.343 | 0.101 | 1.75 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0615__grid_F0320_k0590 | 0.406 | 0.517 | 1.53 | False | calibration_residual |
| grid_F0280_k0610__grid_F0340_k0595 | 0.401 | 0.382 | 1.58 | False | calibration_residual |
| grid_F0280_k0610__grid_F0320_k0590 | 0.363 | 0.469 | 1.55 | False | calibration_residual |
| gs_17__grid_F0360_k0595 | 0.43 | 0.579 | 1.51 | False | calibration_residual |
| grid_F0320_k0625__grid_F0340_k0600 | 0.33 | 0.634 | 1.48 | False | calibration_residual |
| gs_15__grid_F0320_k0595 | 0.324 | 0.697 | 1.44 | False | calibration_residual |
| grid_F0280_k0605__grid_F0340_k0595 | 0.366 | 0.485 | 1.55 | False | calibration_residual |
| grid_F0320_k0625__gs_06 | 0.384 | 0.733 | 1.42 | False | calibration_residual |
| grid_F0280_k0615__grid_F0320_k0580 | 0.494 | 0.505 | 1.54 | False | calibration_residual |
| gs_16__grid_F0320_k0590 | 0.311 | 0.514 | 1.53 | False | calibration_residual |
| grid_F0340_k0640__grid_F0320_k0590 | 0.434 | 0.591 | 1.49 | False | calibration_residual |
| grid_F0300_k0615__grid_F0340_k0600 | 0.317 | 0.684 | 1.44 | False | calibration_residual |
| grid_F0380_k0650__grid_F0360_k0590 | 0.639 | 0.628 | 1.47 | False | calibration_residual |
| grid_F0280_k0615__gs_03 | 0.373 | 0.148 | 1.71 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0630__grid_F0360_k0590 | 0.657 | 0.678 | 1.44 | False | calibration_residual |
| grid_F0280_k0605__grid_F0300_k0570 | 0.404 | 0.494 | 1.53 | False | calibration_residual |
| grid_F0340_k0640__grid_F0360_k0595 | 0.562 | 0.55 | 1.5 | False | calibration_residual |
| grid_F0300_k0625__grid_F0360_k0595 | 0.549 | 0.526 | 1.51 | False | calibration_residual |
| grid_F0280_k0610__grid_F0320_k0585 | 0.404 | 0.337 | 1.54 | False | calibration_residual |
| gs_15__grid_F0360_k0595 | 0.491 | 0.465 | 1.54 | False | calibration_residual |
| grid_F0320_k0615__grid_F0320_k0585 | 0.278 | 0.745 | 1.4 | False | calibration_residual |
| grid_F0300_k0615__gs_06 | 0.382 | 0.754 | 1.39 | False | calibration_residual |
| grid_F0280_k0605__grid_F0320_k0590 | 0.327 | 0.523 | 1.51 | False | calibration_residual |
| grid_F0320_k0615__grid_F0320_k0580 | 0.323 | 0.734 | 1.4 | False | calibration_residual |
| grid_F0260_k0605__gs_03 | 0.341 | 0.075 | 1.73 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0630__grid_F0320_k0585 | 0.426 | 0.378 | 1.57 | False | calibration_residual |
| gs_17__gs_06 | 0.354 | 0.747 | 1.39 | False | calibration_residual |
| grid_F0280_k0610__grid_F0300_k0585 | 0.314 | 0.568 | 1.47 | False | calibration_residual |
| grid_F0300_k0630__grid_F0320_k0580 | 0.564 | 0.714 | 1.4 | False | calibration_residual |
| grid_F0280_k0615__grid_F0340_k0595 | 0.445 | 0.443 | 1.53 | False | calibration_residual |
| grid_F0280_k0615__grid_F0320_k0585 | 0.448 | 0.399 | 1.54 | False | calibration_residual |
| grid_F0320_k0630__grid_F0360_k0605 | 0.411 | 0.628 | 1.44 | False | calibration_residual |
| gs_15__grid_F0360_k0605 | 0.393 | 0.644 | 1.43 | False | calibration_residual |
| gs_16__grid_F0340_k0600 | 0.306 | 0.594 | 1.45 | False | calibration_residual |
| grid_F0260_k0605__grid_F0320_k0590 | 0.365 | 0.0432 | 1.5 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0605__grid_F0360_k0595 | 0.443 | 0.256 | 1.55 | False | calibration_residual |
| grid_F0320_k0625__grid_F0340_k0605 | 0.283 | 0.751 | 1.37 | False | calibration_residual |
| grid_F0280_k0605__grid_F0320_k0585 | 0.37 | 0.486 | 1.5 | False | calibration_residual |
| grid_F0280_k0605__grid_F0320_k0580 | 0.413 | 0.461 | 1.51 | False | calibration_residual |
| grid_F0280_k0610__grid_F0420_k0610 | 0.577 | 0.0827 | 1.62 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0320_k0600 | 0.254 | 0.761 | 1.35 | False | calibration_residual |
| grid_F0300_k0615__grid_F0340_k0605 | 0.271 | 0.798 | 1.33 | False | calibration_residual |
| grid_F0360_k0630__grid_F0360_k0595 | 0.366 | 0.71 | 1.38 | False | calibration_residual |
| grid_F0320_k0630__gs_06 | 0.432 | 0.696 | 1.38 | False | calibration_residual |
| grid_F0280_k0610__grid_F0300_k0580 | 0.351 | 0.286 | 1.51 | False | calibration_residual |
| grid_F0280_k0605__grid_F0300_k0580 | 0.318 | 0.486 | 1.48 | False | calibration_residual |
| grid_F0280_k0615__grid_F0300_k0585 | 0.353 | 0.529 | 1.46 | False | calibration_residual |
| gs_15__grid_F0340_k0600 | 0.36 | 0.588 | 1.43 | False | calibration_residual |
| grid_F0340_k0620__grid_F0360_k0590 | 0.384 | 0.801 | 1.32 | False | calibration_residual |
| grid_F0360_k0630__grid_F0340_k0595 | 0.279 | 0.749 | 1.35 | False | calibration_residual |
| grid_F0300_k0630__grid_F0320_k0585 | 0.518 | 0.656 | 1.39 | False | calibration_residual |
| gs_15__gs_06 | 0.417 | 0.696 | 1.37 | False | calibration_residual |
| grid_F0280_k0615__grid_F0360_k0595 | 0.528 | 0.378 | 1.53 | False | calibration_residual |
| grid_F0320_k0615__grid_F0320_k0590 | 0.234 | 0.715 | 1.36 | False | calibration_residual |
| grid_F0320_k0615__grid_F0360_k0605 | 0.262 | 0.748 | 1.33 | False | calibration_residual |
| grid_F0320_k0615__grid_F0360_k0595 | 0.36 | 0.602 | 1.41 | False | calibration_residual |
| grid_F0300_k0625__grid_F0300_k0580 | 0.411 | 0.406 | 1.5 | False | calibration_residual |
| grid_F0320_k0615__grid_F0320_k0595 | 0.203 | 0.776 | 1.32 | False | calibration_residual |
| gs_17__grid_F0360_k0605 | 0.329 | 0.634 | 1.39 | False | calibration_residual |
| gs_16__grid_F0360_k0610 | 0.29 | 0.702 | 1.35 | False | calibration_residual |
| gs_15__grid_F0300_k0585 | 0.311 | 0.548 | 1.43 | False | calibration_residual |
| grid_F0280_k0605__grid_F0420_k0610 | 0.548 | 0.0815 | 1.33 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0625__grid_F0360_k0610 | 0.313 | 0.696 | 1.35 | False | calibration_residual |
| grid_F0300_k0630__grid_F0320_k0590 | 0.474 | 0.63 | 1.38 | False | calibration_residual |
| gs_16__grid_F0340_k0605 | 0.258 | 0.708 | 1.34 | False | calibration_residual |
| grid_F0300_k0630__grid_F0340_k0595 | 0.515 | 0.591 | 1.4 | False | calibration_residual |
| grid_F0320_k0625__grid_F0320_k0600 | 0.247 | 0.767 | 1.31 | False | calibration_residual |
| grid_F0280_k0615__grid_F0300_k0580 | 0.393 | 0.291 | 1.53 | False | calibration_residual |
| grid_F0320_k0615__grid_F0340_k0600 | 0.23 | 0.727 | 1.33 | False | calibration_residual |
| grid_F0380_k0650__grid_F0300_k0580 | 0.424 | 0.0317 | 1.26 | False | calibration_residual;endpoint_ess |
| grid_F0420_k0650__grid_F0360_k0590 | 0.548 | 0.4 | 1.48 | False | calibration_residual |
| grid_F0320_k0615__gs_06 | 0.312 | 0.78 | 1.29 | False | calibration_residual |
| grid_F0280_k0610__gs_03 | 0.338 | 0.0707 | 1.64 | False | calibration_residual;endpoint_ess |
| gs_15__grid_F0340_k0605 | 0.314 | 0.715 | 1.32 | False | calibration_residual |
| grid_F0300_k0615__grid_F0360_k0610 | 0.3 | 0.724 | 1.31 | False | calibration_residual |
| grid_F0280_k0605__grid_F0300_k0585 | 0.276 | 0.518 | 1.42 | False | calibration_residual |
| grid_F0300_k0615__grid_F0320_k0600 | 0.235 | 0.799 | 1.28 | False | calibration_residual |
| grid_F0320_k0630__grid_F0340_k0600 | 0.376 | 0.505 | 1.42 | False | calibration_residual |
| grid_F0340_k0620__grid_F0300_k0570 | 0.438 | 0.827 | 1.26 | False | calibration_residual |
| grid_F0360_k0630__grid_F0360_k0605 | 0.266 | 0.771 | 1.28 | False | calibration_residual |
| grid_F0340_k0640__grid_F0360_k0605 | 0.462 | 0.58 | 1.38 | False | calibration_residual |
| grid_F0280_k0610__grid_F0320_k0595 | 0.319 | 0.495 | 1.41 | False | calibration_residual |
| gs_15__grid_F0320_k0600 | 0.279 | 0.757 | 1.28 | False | calibration_residual |
| grid_F0320_k0630__grid_F0320_k0595 | 0.337 | 0.511 | 1.4 | False | calibration_residual |
| grid_F0340_k0620__grid_F0340_k0595 | 0.244 | 0.823 | 1.25 | False | calibration_residual |
| grid_F0280_k0615__grid_F0320_k0595 | 0.361 | 0.523 | 1.39 | False | calibration_residual |
| grid_F0360_k0630__gs_06 | 0.289 | 0.806 | 1.25 | False | calibration_residual |
| grid_F0280_k0605__grid_F0320_k0595 | 0.284 | 0.531 | 1.38 | False | calibration_residual |
| grid_F0380_k0650__grid_F0360_k0595 | 0.58 | 0.495 | 1.4 | False | calibration_residual |
| grid_F0320_k0625__grid_F0360_k0600 | 0.406 | 0.247 | 1.52 | False | calibration_residual |
| grid_F0340_k0640__gs_06 | 0.484 | 0.653 | 1.32 | False | calibration_residual |
| grid_F0320_k0615__grid_F0340_k0605 | 0.183 | 0.834 | 1.22 | False | calibration_residual |
| grid_F0280_k0605__grid_F0360_k0600 | 0.399 | 0.313 | 1.48 | False | calibration_residual |
| grid_F0300_k0625__grid_F0360_k0605 | 0.449 | 0.537 | 1.36 | False | calibration_residual |
| grid_F0340_k0620__grid_F0320_k0580 | 0.352 | 0.801 | 1.22 | False | calibration_residual |
| grid_F0320_k0625__grid_F0300_k0585 | 0.278 | 0.379 | 1.43 | False | calibration_residual |
| grid_F0360_k0630__grid_F0320_k0580 | 0.401 | 0.611 | 1.31 | False | calibration_residual |
| grid_F0320_k0630__grid_F0340_k0605 | 0.329 | 0.62 | 1.31 | False | calibration_residual |
| grid_F0280_k0605__grid_F0360_k0605 | 0.358 | 0.489 | 1.37 | False | calibration_residual |
| gs_15__grid_F0360_k0610 | 0.343 | 0.637 | 1.3 | False | calibration_residual |
| grid_F0320_k0630__grid_F0360_k0610 | 0.36 | 0.621 | 1.3 | False | calibration_residual |
| grid_F0280_k0615__grid_F0360_k0605 | 0.43 | 0.51 | 1.36 | False | calibration_residual |
| grid_F0280_k0610__grid_F0360_k0600 | 0.428 | 0.134 | 1.5 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0630__grid_F0360_k0595 | 0.599 | 0.48 | 1.37 | False | calibration_residual |
| grid_F0300_k0610__grid_F0360_k0590 | 0.447 | 0.295 | 1.46 | False | calibration_residual |
| grid_F0300_k0615__grid_F0360_k0600 | 0.394 | 0.297 | 1.46 | False | calibration_residual |
| grid_F0280_k0615__gs_06 | 0.461 | 0.604 | 1.3 | False | calibration_residual |
| grid_F0300_k0625__gs_06 | 0.471 | 0.62 | 1.3 | False | calibration_residual |
| grid_F0300_k0625__grid_F0300_k0585 | 0.367 | 0.408 | 1.4 | False | calibration_residual |
| grid_F0340_k0620__grid_F0360_k0595 | 0.328 | 0.722 | 1.24 | False | calibration_residual |
| grid_F0340_k0620__grid_F0320_k0585 | 0.308 | 0.787 | 1.2 | False | calibration_residual |
| grid_F0300_k0610__grid_F0340_k0595 | 0.309 | 0.372 | 1.41 | False | calibration_residual |
| gs_17__grid_F0340_k0595 | 0.338 | 0.353 | 1.41 | False | calibration_residual |
| grid_F0380_k0650__grid_F0300_k0590 | 0.331 | 0.0501 | 1.13 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0605__grid_F0340_k0600 | 0.318 | 0.455 | 1.36 | False | calibration_residual |
| grid_F0360_k0630__grid_F0300_k0585 | 0.374 | 0.661 | 1.25 | False | calibration_residual |
| grid_F0340_k0650__grid_F0360_k0590 | 0.819 | 0.733 | 1.21 | False | calibration_residual |
| gs_16__gs_08 | 0.439 | 0.646 | 1.25 | False | calibration_residual |
| grid_F0300_k0610__grid_F0340_k0600 | 0.263 | 0.542 | 1.31 | False | calibration_residual |
| grid_F0320_k0630__grid_F0360_k0600 | 0.451 | 0.167 | 1.49 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0610__gs_06 | 0.355 | 0.554 | 1.3 | False | calibration_residual |
| grid_F0280_k0615__grid_F0340_k0600 | 0.397 | 0.42 | 1.36 | False | calibration_residual |
| grid_F0300_k0625__grid_F0360_k0600 | 0.494 | 0.223 | 1.46 | False | calibration_residual |
| grid_F0420_k0650__grid_F0360_k0595 | 0.488 | 0.314 | 1.41 | False | calibration_residual |
| grid_F0380_k0650__grid_F0300_k0570 | 0.531 | 0.34 | 1.4 | False | calibration_residual |
| grid_F0380_k0650__grid_F0320_k0580 | 0.541 | 0.369 | 1.38 | False | calibration_residual |
| grid_F0300_k0610__grid_F0360_k0605 | 0.302 | 0.501 | 1.32 | False | calibration_residual |
| grid_F0280_k0615__grid_F0280_k0580 | 0.263 | 0.151 | 1.49 | False | calibration_residual;endpoint_ess |
| grid_F0420_k0650__grid_F0300_k0580 | 0.344 | 0.0944 | 1.49 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0615__grid_F0360_k0610 | 0.225 | 0.741 | 1.2 | False | calibration_residual |
| grid_F0340_k0640__grid_F0320_k0595 | 0.386 | 0.48 | 1.33 | False | calibration_residual |
| gs_16__grid_F0360_k0600 | 0.381 | 0.23 | 1.45 | False | calibration_residual |
| grid_F0380_k0650__grid_F0340_k0595 | 0.49 | 0.364 | 1.38 | False | calibration_residual |
| gs_15__gs_03 | 0.393 | 0.0705 | 1.53 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0630__grid_F0360_k0605 | 0.5 | 0.569 | 1.27 | False | calibration_residual |
| gs_17__grid_F0300_k0580 | 0.398 | 0.255 | 1.43 | False | calibration_residual |
| grid_F0340_k0650__grid_F0300_k0570 | 0.717 | 0.678 | 1.22 | False | calibration_residual |
| grid_F0280_k0610__grid_F0340_k0600 | 0.353 | 0.346 | 1.38 | False | calibration_residual |
| gs_07__grid_F0340_k0605 | 0.58 | 0.809 | 1.15 | False | calibration_residual |
| gs_17__grid_F0320_k0580 | 0.396 | 0.267 | 1.42 | False | calibration_residual |
| grid_F0320_k0625__gs_08 | 0.479 | 0.639 | 1.23 | False | calibration_residual |
| grid_F0340_k0620__grid_F0360_k0605 | 0.23 | 0.794 | 1.15 | False | calibration_residual |
| grid_F0300_k0615__grid_F0300_k0585 | 0.265 | 0.327 | 1.38 | False | calibration_residual |
| grid_F0340_k0650__grid_F0320_k0580 | 0.726 | 0.677 | 1.21 | False | calibration_residual |
| grid_F0340_k0640__grid_F0360_k0600 | 0.505 | 0.256 | 1.42 | False | calibration_residual |
| grid_F0420_k0650__grid_F0300_k0570 | 0.429 | 0.0921 | 1.48 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0630__grid_F0320_k0600 | 0.292 | 0.597 | 1.25 | False | calibration_residual |
| grid_F0340_k0640__grid_F0340_k0600 | 0.425 | 0.409 | 1.34 | False | calibration_residual |
| grid_F0300_k0630__grid_F0320_k0595 | 0.429 | 0.528 | 1.28 | False | calibration_residual |
| grid_F0300_k0625__grid_F0320_k0595 | 0.376 | 0.404 | 1.34 | False | calibration_residual |
| grid_F0300_k0610__grid_F0340_k0605 | 0.217 | 0.656 | 1.21 | False | calibration_residual |
| grid_F0280_k0605__gs_06 | 0.407 | 0.43 | 1.32 | False | calibration_residual |
| grid_F0280_k0610__grid_F0340_k0605 | 0.309 | 0.504 | 1.28 | False | calibration_residual |
| gs_07__grid_F0360_k0610 | 0.535 | 0.851 | 1.11 | False | calibration_residual |
| grid_F0300_k0615__gs_08 | 0.503 | 0.662 | 1.2 | False | calibration_residual |
| grid_F0300_k0610__grid_F0360_k0595 | 0.391 | 0.292 | 1.38 | False | calibration_residual |
| grid_F0340_k0640__grid_F0360_k0610 | 0.411 | 0.584 | 1.24 | False | calibration_residual |
| grid_F0280_k0615__grid_F0340_k0605 | 0.35 | 0.535 | 1.26 | False | calibration_residual |
| grid_F0300_k0630__gs_06 | 0.522 | 0.628 | 1.21 | False | calibration_residual |
| gs_17__gs_08 | 0.368 | 0.665 | 1.2 | False | calibration_residual |
| grid_F0320_k0630__gs_03 | 0.411 | 0.02 | 1.51 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0605__grid_F0360_k0610 | 0.326 | 0.534 | 1.26 | False | calibration_residual |
| grid_F0280_k0610__grid_F0280_k0580 | 0.222 | 0.175 | 1.43 | False | calibration_residual;endpoint_ess |
| grid_F0380_k0650__gs_06 | 0.502 | 0.615 | 1.21 | False | calibration_residual |
| grid_F0320_k0630__grid_F0300_k0580 | 0.359 | 0.063 | 1.48 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0610__grid_F0360_k0605 | 0.384 | 0.311 | 1.35 | False | calibration_residual |
| grid_F0340_k0620__grid_F0340_k0600 | 0.197 | 0.768 | 1.13 | False | calibration_residual |
| grid_F0340_k0620__gs_06 | 0.262 | 0.814 | 1.11 | False | calibration_residual |
| grid_F0360_k0630__grid_F0340_k0600 | 0.229 | 0.653 | 1.19 | False | calibration_residual |
| grid_F0300_k0625__grid_F0340_k0600 | 0.414 | 0.366 | 1.33 | False | calibration_residual |
| grid_F0320_k0630__gs_08 | 0.505 | 0.643 | 1.19 | False | calibration_residual |
| grid_F0420_k0650__grid_F0320_k0590 | 0.347 | 0.0736 | 1.43 | False | calibration_residual;endpoint_ess |
| gs_15__grid_F0360_k0600 | 0.434 | 0.131 | 1.44 | False | calibration_residual;endpoint_ess |
| gs_16__gs_09 | 0.624 | 0.366 | 1.33 | False | calibration_residual |
| grid_F0340_k0640__grid_F0300_k0580 | 0.415 | 0.227 | 1.39 | False | calibration_residual |
| grid_F0320_k0615__grid_F0360_k0600 | 0.307 | 0.372 | 1.32 | False | calibration_residual |
| grid_F0320_k0630__grid_F0300_k0585 | 0.321 | 0.217 | 1.39 | False | calibration_residual |
| gs_15__gs_08 | 0.524 | 0.647 | 1.18 | False | calibration_residual |
| grid_F0360_k0630__grid_F0300_k0570 | 0.48 | 0.447 | 1.28 | False | calibration_residual |
| grid_F0380_k0650__grid_F0360_k0605 | 0.477 | 0.505 | 1.25 | False | calibration_residual |
| grid_F0320_k0615__grid_F0320_k0600 | 0.162 | 0.732 | 1.13 | False | calibration_residual |
| grid_F0300_k0630__grid_F0340_k0600 | 0.466 | 0.464 | 1.26 | False | calibration_residual |
| grid_F0420_k0650__grid_F0320_k0585 | 0.39 | 0.0549 | 1.45 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0420_k0610 | 0.488 | 0.222 | 1.38 | False | calibration_residual |
| gs_07__grid_F0320_k0595 | 0.593 | 0.805 | 1.09 | False | calibration_residual |
| grid_F0280_k0610__gs_06 | 0.42 | 0.314 | 1.33 | False | calibration_residual |
| grid_F0300_k0625__grid_F0360_k0610 | 0.399 | 0.531 | 1.22 | False | calibration_residual |
| grid_F0280_k0615__grid_F0360_k0610 | 0.381 | 0.49 | 1.24 | False | calibration_residual |
| gs_17__grid_F0320_k0595 | 0.284 | 0.283 | 1.34 | False | calibration_residual |
| grid_F0280_k0610__grid_F0360_k0610 | 0.34 | 0.432 | 1.27 | False | calibration_residual |
| gs_17__grid_F0360_k0610 | 0.276 | 0.541 | 1.21 | False | calibration_residual |
| grid_F0360_k0630__grid_F0360_k0610 | 0.215 | 0.722 | 1.12 | False | calibration_residual |
| grid_F0340_k0650__grid_F0320_k0585 | 0.678 | 0.543 | 1.21 | False | calibration_residual |
| grid_F0300_k0630__grid_F0300_k0580 | 0.456 | 0.323 | 1.31 | False | calibration_residual |
| gs_07__grid_F0340_k0600 | 0.538 | 0.801 | 1.08 | False | calibration_residual |
| grid_F0280_k0615__grid_F0360_k0600 | 0.47 | 0.0708 | 1.43 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0605__grid_F0340_k0605 | 0.271 | 0.502 | 1.23 | False | calibration_residual |
| gs_16__grid_F0320_k0595 | 0.261 | 0.339 | 1.3 | False | calibration_residual |
| grid_F0320_k0625__grid_F0300_k0580 | 0.311 | 0.0979 | 1.37 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__grid_F0320_k0590 | 0.632 | 0.463 | 1.24 | False | calibration_residual |
| grid_F0280_k0615__grid_F0320_k0600 | 0.314 | 0.562 | 1.19 | False | calibration_residual |
| grid_F0340_k0650__grid_F0340_k0595 | 0.674 | 0.484 | 1.23 | False | calibration_residual |
| grid_F0320_k0625__gs_03 | 0.448 | 0.0199 | 1.46 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0610__grid_F0320_k0590 | 0.266 | 0.272 | 1.33 | False | calibration_residual |
| grid_F0320_k0615__gs_08 | 0.438 | 0.648 | 1.14 | False | calibration_residual |
| grid_F0420_k0650__grid_F0340_k0595 | 0.382 | 0.056 | 1.35 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0615__gs_08 | 0.574 | 0.656 | 1.14 | False | calibration_residual |
| grid_F0340_k0650__grid_F0360_k0595 | 0.761 | 0.53 | 1.2 | False | calibration_residual |
| grid_F0280_k0610__gs_09 | 0.737 | 0.385 | 1.27 | False | calibration_residual |
| grid_F0320_k0625__grid_F0420_k0610 | 0.54 | 0.138 | 1.37 | False | calibration_residual;endpoint_ess |
| gs_16__grid_F0420_k0610 | 0.507 | 0.14 | 1.37 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0640__grid_F0300_k0585 | 0.372 | 0.282 | 1.32 | False | calibration_residual |
| grid_F0280_k0600__grid_F0300_k0585 | 0.226 | 0.41 | 1.25 | False | calibration_residual |
| grid_F0300_k0610__gs_08 | 0.482 | 0.573 | 1.17 | False | calibration_residual |
| grid_F0340_k0620__grid_F0320_k0590 | 0.269 | 0.67 | 1.12 | False | calibration_residual |
| grid_F0320_k0625__gs_09 | 0.659 | 0.374 | 1.27 | False | calibration_residual |
| grid_F0320_k0630__grid_F0420_k0610 | 0.579 | 0.155 | 1.36 | False | calibration_residual;endpoint_ess |
| grid_F0360_k0630__grid_F0360_k0600 | 0.309 | 0.426 | 1.23 | False | calibration_residual |
| grid_F0260_k0605__grid_F0320_k0595 | 0.318 | 0.0378 | 1.27 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__grid_F0320_k0590 | 0.273 | 0.286 | 1.29 | False | calibration_residual |
| grid_F0300_k0615__gs_09 | 0.688 | 0.373 | 1.26 | False | calibration_residual |
| grid_F0360_k0630__grid_F0320_k0585 | 0.351 | 0.438 | 1.23 | False | calibration_residual |
| grid_F0300_k0610__grid_F0320_k0595 | 0.223 | 0.379 | 1.25 | False | calibration_residual |
| grid_F0280_k0615__gs_09 | 0.764 | 0.505 | 1.19 | False | calibration_residual |
| gs_07__grid_F0320_k0600 | 0.631 | 0.637 | 1.12 | False | calibration_residual |
| grid_F0340_k0620__grid_F0340_k0605 | 0.15 | 0.843 | 1.02 | False | calibration_residual |
| grid_F0300_k0610__grid_F0360_k0610 | 0.269 | 0.51 | 1.19 | False | calibration_residual |
| gs_16__grid_F0300_k0585 | 0.275 | 0.0682 | 0.995 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0600__grid_F0340_k0595 | 0.31 | 0.21 | 1.32 | False | calibration_residual |
| gs_07__grid_F0320_k0590 | 0.554 | 0.764 | 1.06 | False | calibration_residual |
| gs_07__grid_F0360_k0605 | 0.489 | 0.773 | 1.05 | False | calibration_residual |
| grid_F0280_k0600__grid_F0300_k0570 | 0.348 | 0.226 | 1.31 | False | calibration_residual |
| grid_F0300_k0630__grid_F0300_k0585 | 0.414 | 0.397 | 1.24 | False | calibration_residual |
| gs_15__gs_09 | 0.707 | 0.415 | 1.22 | False | calibration_residual |
| gs_07__gs_06 | 0.531 | 0.744 | 1.06 | False | calibration_residual |
| gs_17__gs_09 | 0.585 | 0.339 | 1.26 | False | calibration_residual |
| grid_F0320_k0630__gs_09 | 0.693 | 0.404 | 1.23 | False | calibration_residual |
| grid_F0420_k0650__grid_F0320_k0580 | 0.442 | 0.122 | 1.36 | False | calibration_residual;endpoint_ess |
| grid_F0360_k0630__grid_F0300_k0580 | 0.395 | 0.461 | 1.19 | False | calibration_residual |
| grid_F0420_k0650__grid_F0300_k0585 | 0.313 | 0.126 | 1.35 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__grid_F0300_k0580 | 0.264 | 0.291 | 1.27 | False | calibration_residual |
| grid_F0340_k0640__grid_F0340_k0605 | 0.375 | 0.436 | 1.2 | False | calibration_residual |
| grid_F0300_k0605__grid_F0360_k0590 | 0.4 | 0.411 | 1.22 | False | calibration_residual |
| grid_F0300_k0610__grid_F0360_k0600 | 0.339 | 0.24 | 1.3 | False | calibration_residual |
| grid_F0340_k0640__grid_F0420_k0610 | 0.616 | 0.207 | 1.31 | False | calibration_residual |
| gs_07__grid_F0340_k0595 | 0.496 | 0.751 | 1.04 | False | calibration_residual |
| grid_F0300_k0630__grid_F0340_k0605 | 0.418 | 0.5 | 1.16 | False | calibration_residual |
| grid_F0280_k0600__grid_F0360_k0595 | 0.397 | 0.0789 | 1.34 | False | calibration_residual;endpoint_ess |
| grid_F0380_k0650__grid_F0320_k0585 | 0.488 | 0.174 | 1.31 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0620__grid_F0360_k0600 | 0.276 | 0.51 | 1.16 | False | calibration_residual |
| grid_F0300_k0630__grid_F0360_k0610 | 0.449 | 0.544 | 1.14 | False | calibration_residual |
| grid_F0300_k0625__grid_F0340_k0605 | 0.365 | 0.431 | 1.19 | False | calibration_residual |
| grid_F0380_k0650__grid_F0320_k0590 | 0.44 | 0.154 | 1.32 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0625__grid_F0420_k0610 | 0.627 | 0.174 | 1.32 | False | calibration_residual;endpoint_ess |
| gs_15__grid_F0420_k0610 | 0.579 | 0.109 | 1.3 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__grid_F0360_k0590 | 0.441 | 0.041 | 1.32 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0360_k0600 | 0.367 | 0.242 | 1.28 | False | calibration_residual |
| grid_F0340_k0620__grid_F0360_k0610 | 0.18 | 0.775 | 1.01 | False | calibration_residual |
| grid_F0300_k0605__gs_09 | 0.657 | 0.421 | 1.19 | False | calibration_residual |
| grid_F0420_k0650__grid_F0300_k0590 | 0.286 | 0.234 | 1.28 | False | calibration_residual |
| grid_F0340_k0620__gs_09 | 0.581 | 0.42 | 1.19 | False | calibration_residual |
| grid_F0340_k0620__grid_F0320_k0595 | 0.234 | 0.709 | 1.04 | False | calibration_residual |
| grid_F0300_k0630__grid_F0360_k0600 | 0.542 | 0.205 | 1.29 | False | calibration_residual |
| grid_F0420_k0650__grid_F0420_k0610 | 0.55 | 0.272 | 1.25 | False | calibration_residual |
| grid_F0280_k0600__grid_F0320_k0585 | 0.314 | 0.231 | 1.26 | False | calibration_residual |
| grid_F0320_k0615__gs_09 | 0.623 | 0.174 | 1.3 | False | calibration_residual;endpoint_ess |
| grid_F0360_k0630__gs_09 | 0.555 | 0.292 | 1.24 | False | calibration_residual |
| grid_F0280_k0600__grid_F0320_k0580 | 0.356 | 0.191 | 1.27 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0620__grid_F0320_k0600 | 0.121 | 0.0485 | 0.206 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0615__grid_F0420_k0610 | 0.623 | 0.107 | 1.29 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0610__gs_09 | 0.667 | 0.204 | 1.28 | False | calibration_residual |
| grid_F0300_k0605__grid_F0360_k0595 | 0.351 | 0.445 | 1.16 | False | calibration_residual |
| gs_07__gs_08 | 0.768 | 0.537 | 1.11 | False | calibration_residual |
| grid_F0280_k0600__grid_F0320_k0595 | 0.23 | 0.324 | 1.21 | False | calibration_residual |
| grid_F0340_k0650__grid_F0300_k0580 | 0.618 | 0.332 | 1.21 | False | calibration_residual |
| grid_F0360_k0630__gs_08 | 0.37 | 0.658 | 1.05 | False | calibration_residual |
| grid_F0300_k0615__grid_F0420_k0610 | 0.544 | 0.124 | 1.29 | False | calibration_residual;endpoint_ess |
| grid_F0360_k0630__grid_F0320_k0600 | 0.244 | 0.706 | 1.02 | False | calibration_residual |
| grid_F0300_k0605__grid_F0340_k0595 | 0.258 | 0.413 | 1.16 | False | calibration_residual |
| grid_F0300_k0610__grid_F0320_k0600 | 0.178 | 0.536 | 1.1 | False | calibration_residual |
| grid_F0420_k0650__grid_F0360_k0600 | 0.425 | 0.162 | 1.28 | False | calibration_residual;endpoint_ess |
| grid_F0260_k0605__grid_F0280_k0580 | 0.201 | 0.0714 | 1.33 | False | calibration_residual;endpoint_ess |
| gs_07__grid_F0300_k0580 | 0.584 | 0.784 | 0.971 | False | calibration_residual;morphology_effect |
| gs_07__grid_F0300_k0585 | 0.619 | 0.701 | 1 | False | calibration_residual |
| grid_F0380_k0650__grid_F0360_k0600 | 0.516 | 0.129 | 1.26 | False | calibration_residual;endpoint_ess |
| grid_F0380_k0650__grid_F0300_k0585 | 0.381 | 0.0922 | 1.25 | False | calibration_residual;endpoint_ess |
| gs_07__grid_F0320_k0585 | 0.514 | 0.768 | 0.963 | False | calibration_residual;morphology_effect |
| gs_15__grid_F0280_k0580 | 0.215 | 0.0824 | 1.3 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0605__grid_F0360_k0600 | 0.307 | 0.453 | 1.11 | False | calibration_residual |
| gs_07__grid_F0360_k0600 | 0.44 | 0.673 | 1 | False | calibration_residual |
| grid_F0280_k0600__grid_F0340_k0600 | 0.277 | 0.298 | 1.18 | False | calibration_residual |
| grid_F0420_k0650__grid_F0320_k0595 | 0.308 | 0.116 | 1.27 | False | calibration_residual;endpoint_ess |
| grid_F0360_k0630__grid_F0340_k0605 | 0.19 | 0.609 | 1.03 | False | calibration_residual |
| grid_F0280_k0600__grid_F0300_k0590 | 0.178 | 0.431 | 1.11 | False | calibration_residual |
| grid_F0260_k0605__grid_F0340_k0600 | 0.341 | 0.0306 | 1.23 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__grid_F0360_k0600 | 0.356 | 0.17 | 1.23 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0620__grid_F0280_k0580 | 0.267 | 0.0665 | 1.05 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0320_k0585 | 0.341 | 0.102 | 1.23 | False | calibration_residual;endpoint_ess |
| gs_07__gs_09 | 0.715 | 0.551 | 1.04 | False | calibration_residual |
| grid_F0340_k0650__grid_F0300_k0585 | 0.574 | 0.319 | 1.15 | False | calibration_residual |
| gs_07__grid_F0420_k0610 | 0.565 | 0.822 | 0.898 | False | calibration_residual;morphology_effect |
| grid_F0340_k0650__grid_F0360_k0605 | 0.658 | 0.477 | 1.06 | False | calibration_residual |
| grid_F0320_k0615__grid_F0420_k0610 | 0.462 | 0.161 | 1.22 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0605__grid_F0360_k0605 | 0.28 | 0.452 | 1.07 | False | calibration_residual |
| grid_F0300_k0610__grid_F0300_k0585 | 0.208 | 0.181 | 1.16 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0630__grid_F0320_k0600 | 0.381 | 0.362 | 1.11 | False | calibration_residual |
| grid_F0300_k0625__grid_F0320_k0600 | 0.326 | 0.354 | 1.12 | False | calibration_residual |
| grid_F0340_k0650__gs_06 | 0.68 | 0.576 | 1 | False | calibration_residual |
| grid_F0300_k0605__gs_06 | 0.333 | 0.493 | 1.04 | False | calibration_residual |
| grid_F0280_k0610__gs_08 | 0.539 | 0.342 | 1.12 | False | calibration_residual |
| grid_F0380_k0650__grid_F0360_k0610 | 0.423 | 0.386 | 1.1 | False | calibration_residual |
| grid_F0280_k0605__grid_F0300_k0590 | 0.219 | 0.228 | 1.17 | False | calibration_residual |
| grid_F0360_k0630__grid_F0420_k0610 | 0.435 | 0.205 | 1.19 | False | calibration_residual |
| grid_F0360_k0630__grid_F0300_k0590 | 0.326 | 0.517 | 1.03 | False | calibration_residual |
| gs_17__grid_F0300_k0570 | 0.464 | 0.0834 | 1.23 | False | calibration_residual;endpoint_ess |
| grid_F0420_k0650__gs_06 | 0.404 | 0.283 | 1.14 | False | calibration_residual |
| grid_F0380_k0650__grid_F0420_k0610 | 0.633 | 0.193 | 1.17 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__grid_F0360_k0605 | 0.329 | 0.214 | 1.16 | False | calibration_residual |
| grid_F0260_k0605__grid_F0360_k0605 | 0.38 | 0.0323 | 1.11 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0300_k0590 | 0.328 | 0.249 | 1.15 | False | calibration_residual |
| grid_F0300_k0610__grid_F0420_k0610 | 0.502 | 0.141 | 1.2 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0280_k0580 | 0.319 | 0.0218 | 1 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0620__gs_08 | 0.39 | 0.651 | 0.946 | False | calibration_residual;morphology_effect |
| grid_F0300_k0605__grid_F0320_k0580 | 0.303 | 0.34 | 1.1 | False | calibration_residual |
| grid_F0260_k0605__grid_F0300_k0590 | 0.25 | 0.0386 | 1.1 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0605__grid_F0340_k0600 | 0.222 | 0.43 | 1.05 | False | calibration_residual |
| grid_F0260_k0605__gs_09 | 0.671 | 0.0282 | 1.22 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0605__grid_F0420_k0610 | 0.487 | 0.363 | 1.08 | False | calibration_residual |
| grid_F0380_k0650__grid_F0340_k0600 | 0.432 | 0.192 | 1.16 | False | calibration_residual;endpoint_ess |
| grid_F0260_k0605__gs_08 | 0.485 | 0.0368 | 1.23 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0610__grid_F0300_k0590 | 0.245 | 0.0947 | 1.1 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0630__grid_F0420_k0610 | 0.667 | 0.174 | 1.17 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__gs_03 | 0.348 | 0.0558 | 1.22 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0615__grid_F0300_k0590 | 0.292 | 0.12 | 1.18 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__grid_F0320_k0595 | 0.582 | 0.345 | 1.08 | False | calibration_residual |
| grid_F0280_k0600__grid_F0340_k0605 | 0.246 | 0.355 | 1.07 | False | calibration_residual |
| grid_F0340_k0650__grid_F0360_k0600 | 0.702 | 0.188 | 1.16 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0620__grid_F0420_k0610 | 0.425 | 0.286 | 1.11 | False | calibration_residual |
| gs_07__grid_F0320_k0580 | 0.474 | 0.701 | 0.894 | False | calibration_residual;morphology_effect |
| grid_F0300_k0605__grid_F0320_k0585 | 0.258 | 0.32 | 1.08 | False | calibration_residual |
| grid_F0320_k0615__grid_F0300_k0585 | 0.27 | 0.298 | 1.08 | False | calibration_residual |
| gs_07__grid_F0360_k0595 | 0.396 | 0.732 | 0.87 | False | calibration_residual;morphology_effect |
| gs_07__grid_F0300_k0570 | 0.508 | 0.686 | 0.886 | False | calibration_residual;morphology_effect |
| grid_F0420_k0650__grid_F0320_k0600 | 0.268 | 0.213 | 1.12 | False | calibration_residual |
| grid_F0280_k0600__grid_F0420_k0610 | 0.522 | 0.046 | 1.17 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__grid_F0360_k0610 | 0.299 | 0.272 | 1.08 | False | calibration_residual |
| grid_F0280_k0600__gs_09 | 0.667 | 0.0925 | 1.17 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0600__grid_F0280_k0580 | 0.159 | 0.134 | 1.15 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0605__gs_08 | 0.461 | 0.519 | 0.958 | False | calibration_residual;morphology_effect |
| grid_F0280_k0605__gs_08 | 0.516 | 0.265 | 1.08 | False | calibration_residual |
| grid_F0300_k0605__grid_F0300_k0570 | 0.301 | 0.264 | 1.08 | False | calibration_residual |
| grid_F0340_k0640__grid_F0320_k0600 | 0.331 | 0.266 | 1.08 | False | calibration_residual |
| gs_16__grid_F0320_k0600 | 0.209 | 0.288 | 1.06 | False | calibration_residual |
| grid_F0280_k0600__gs_06 | 0.376 | 0.167 | 1.12 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0605__grid_F0280_k0580 | 0.139 | 0.0318 | 1.14 | False | calibration_residual;endpoint_ess |
| grid_F0260_k0605__grid_F0360_k0610 | 0.345 | 0.0288 | 1.12 | False | calibration_residual;endpoint_ess |
| gs_17__grid_F0340_k0600 | 0.278 | 0.202 | 1.1 | False | calibration_residual |
| grid_F0300_k0615__gs_03 | 0.385 | 0.0172 | 1.19 | False | calibration_residual;endpoint_ess |
| grid_F0320_k0625__grid_F0280_k0580 | 0.239 | 0.0205 | 1.18 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0605__grid_F0340_k0605 | 0.191 | 0.49 | 0.954 | False | calibration_residual;morphology_effect |
| grid_F0340_k0650__grid_F0340_k0600 | 0.618 | 0.265 | 1.06 | False | calibration_residual |
| grid_F0380_k0650__grid_F0320_k0595 | 0.392 | 0.103 | 1.13 | False | calibration_residual;endpoint_ess |
| grid_F0420_k0650__grid_F0340_k0605 | 0.291 | 0.114 | 1.11 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0605__grid_F0320_k0590 | 0.213 | 0.27 | 1.05 | False | calibration_residual |
| grid_F0260_k0605__grid_F0340_k0605 | 0.29 | 0.0288 | 1.1 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0605__grid_F0320_k0600 | 0.229 | 0.263 | 1.05 | False | calibration_residual |
| grid_F0300_k0605__grid_F0360_k0610 | 0.246 | 0.46 | 0.946 | False | calibration_residual;morphology_effect |
| gs_16__gs_03 | 0.488 | 0.0176 | 1.17 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__grid_F0360_k0610 | 0.606 | 0.394 | 0.968 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__gs_03 | 0.258 | 0.034 | 1.08 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__grid_F0420_k0610 | 0.814 | 0.239 | 1 | False | calibration_residual |
| gs_07__grid_F0360_k0590 | 0.346 | 0.631 | 0.795 | False | calibration_residual;morphology_effect |
| grid_F0300_k0605__grid_F0320_k0595 | 0.169 | 0.265 | 0.981 | False | calibration_residual;morphology_effect |
| grid_F0340_k0650__grid_F0340_k0605 | 0.566 | 0.231 | 0.975 | False | calibration_residual;morphology_effect |
| grid_F0280_k0605__gs_09 | 0.596 | 0.0205 | 1.01 | False | calibration_residual;endpoint_ess |
| grid_F0380_k0650__grid_F0340_k0605 | 0.375 | 0.112 | 1.02 | False | calibration_residual;endpoint_ess |
| grid_F0280_k0610__grid_F0320_k0600 | 0.258 | 0.128 | 1.01 | False | calibration_residual;endpoint_ess |
| grid_F0300_k0610__gs_03 | 0.415 | 0.0353 | 1.04 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0620__grid_F0300_k0580 | 0.298 | 0.0263 | 0.731 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_17__gs_03 | 0.545 | 0.0334 | 1.04 | False | calibration_residual;endpoint_ess |
| grid_F0360_k0630__grid_F0320_k0595 | 0.264 | 0.221 | 0.943 | False | calibration_residual;morphology_effect |
| grid_F0280_k0600__gs_08 | 0.491 | 0.257 | 0.922 | False | calibration_residual;morphology_effect |
| grid_F0320_k0630__grid_F0300_k0590 | 0.251 | 0.0365 | 0.956 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_16__grid_F0300_k0590 | 0.238 | 0.134 | 0.89 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0625__grid_F0300_k0590 | 0.294 | 0.0449 | 1.01 | False | calibration_residual;endpoint_ess |
| grid_F0340_k0650__grid_F0300_k0590 | 0.502 | 0.0655 | 0.964 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0260_k0605__grid_F0320_k0600 | 0.234 | 0.0319 | 0.756 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0615__grid_F0280_k0580 | 0.188 | 0.0233 | 1.02 | False | calibration_residual;endpoint_ess |
| gs_07__grid_F0300_k0590 | 0.642 | 0.132 | 0.955 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0420_k0650__grid_F0360_k0605 | 0.362 | 0.0612 | 0.963 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0640__grid_F0300_k0590 | 0.3 | 0.0476 | 0.943 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0650__grid_F0320_k0600 | 0.521 | 0.151 | 0.901 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0615__gs_03 | 0.468 | 0.0646 | 0.938 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0300_k0570 | 0.282 | 0.275 | 0.837 | False | calibration_residual;morphology_effect |
| grid_F0280_k0600__grid_F0320_k0600 | 0.174 | 0.161 | 0.876 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0610__grid_F0300_k0590 | 0.149 | 0.0636 | 0.662 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0615__grid_F0280_k0580 | 0.214 | 0.0326 | 0.839 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0340_k0595 | 0.276 | 0.26 | 0.829 | False | calibration_residual;morphology_effect |
| grid_F0300_k0605__grid_F0320_k0600 | 0.123 | 0.28 | 0.809 | False | calibration_residual;morphology_effect |
| grid_F0300_k0610__grid_F0280_k0580 | 0.212 | 0.0486 | 0.918 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0320_k0585 | 0.251 | 0.295 | 0.794 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__grid_F0320_k0580 | 0.293 | 0.275 | 0.801 | False | calibration_residual;morphology_effect |
| grid_F0340_k0620__gs_03 | 0.506 | 0.0607 | 0.899 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0630__grid_F0300_k0590 | 0.324 | 0.0448 | 0.81 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0360_k0630__grid_F0280_k0580 | 0.306 | 0.0302 | 0.738 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0620__grid_F0300_k0590 | 0.256 | 0.304 | 0.76 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__grid_F0360_k0600 | 0.329 | 0.237 | 0.789 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__grid_F0360_k0590 | 0.395 | 0.159 | 0.822 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0320_k0590 | 0.216 | 0.263 | 0.769 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__grid_F0360_k0595 | 0.355 | 0.199 | 0.794 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0360_k0630__gs_03 | 0.549 | 0.0688 | 0.855 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0300_k0580 | 0.197 | 0.247 | 0.748 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__grid_F0360_k0605 | 0.298 | 0.235 | 0.716 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__grid_F0280_k0580 | 0.0671 | 0.0446 | 0.793 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0605__gs_03 | 0.396 | 0.0936 | 0.774 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__gs_09 | 0.661 | 0.0746 | 0.786 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0340_k0600 | 0.24 | 0.211 | 0.701 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__grid_F0300_k0585 | 0.152 | 0.203 | 0.702 | False | calibration_residual;morphology_effect |
| grid_F0280_k0595__gs_06 | 0.346 | 0.199 | 0.689 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0605__grid_F0300_k0585 | 0.17 | 0.057 | 0.707 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0420_k0610 | 0.492 | 0.12 | 0.711 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0420_k0650__grid_F0280_k0580 | 0.272 | 0.0301 | 0.75 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_07__grid_F0260_k0570 | 0.576 | 0.0355 | 0.693 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_07__grid_F0280_k0580 | 0.564 | 0.0309 | 0.682 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0380_k0650__grid_F0280_k0580 | 0.244 | 0.0308 | 0.645 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_07__gs_03 | 0.545 | 0.0361 | 0.669 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0320_k0595 | 0.168 | 0.124 | 0.626 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0380_k0650__gs_08 | 0.295 | 0.0212 | 0.504 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0625__grid_F0280_k0580 | 0.18 | 0.0313 | 0.595 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0600__grid_F0260_k0570 | 0.0824 | 0.112 | 0.584 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0615__grid_F0260_k0570 | 0.107 | 0.0928 | 0.575 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0360_k0610 | 0.25 | 0.147 | 0.576 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0340_k0605 | 0.196 | 0.149 | 0.564 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0605__grid_F0280_k0580 | 0.179 | 0.0957 | 0.584 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0615__grid_F0300_k0590 | 0.2 | 0.0979 | 0.566 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_15__grid_F0260_k0570 | 0.125 | 0.0822 | 0.54 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0610__grid_F0260_k0570 | 0.112 | 0.099 | 0.517 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0650__grid_F0280_k0580 | 0.342 | 0.0312 | 0.525 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0640__grid_F0280_k0580 | 0.195 | 0.0311 | 0.537 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0615__grid_F0260_k0570 | 0.112 | 0.082 | 0.504 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0300_k0590 | 0.0872 | 0.0551 | 0.518 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0380_k0650__gs_09 | 0.416 | 0.0169 | 0.467 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__gs_08 | 0.429 | 0.0452 | 0.511 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0630__grid_F0260_k0570 | 0.143 | 0.0741 | 0.453 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0320_k0600 | 0.107 | 0.0573 | 0.44 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0605__grid_F0260_k0570 | 0.0586 | 0.0817 | 0.41 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0625__grid_F0260_k0570 | 0.118 | 0.0701 | 0.403 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0610__grid_F0260_k0570 | 0.0811 | 0.0663 | 0.401 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0320_k0615__grid_F0260_k0570 | 0.0921 | 0.1 | 0.365 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0605__grid_F0260_k0570 | 0.0636 | 0.137 | 0.348 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0280_k0595__grid_F0260_k0570 | 0.0314 | 0.199 | 0.311 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0630__grid_F0280_k0580 | 0.19 | 0.0312 | 0.368 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_16__grid_F0260_k0570 | 0.121 | 0.0638 | 0.354 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0360_k0630__grid_F0260_k0570 | 0.114 | 0.0741 | 0.325 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0620__grid_F0260_k0570 | 0.0787 | 0.102 | 0.289 | False | calibration_residual;endpoint_ess;morphology_effect |
| gs_17__grid_F0260_k0570 | 0.154 | 0.0583 | 0.27 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0380_k0650__grid_F0260_k0570 | 0.108 | 0.0299 | 0.284 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0260_k0605__grid_F0260_k0570 | 0.0187 | 0.0204 | 0.212 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0625__grid_F0260_k0570 | 0.00289 | 0.0187 | 0.0433 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0340_k0640__grid_F0260_k0570 | 0.000524 | 0.0159 | 0.00457 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0300_k0630__grid_F0260_k0570 | 0.00156 | 0.0173 | 0.029 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0420_k0650__grid_F0260_k0570 | 0.0036 | 0.017 | 0.0241 | False | calibration_residual;endpoint_ess;morphology_effect |
| grid_F0420_k0650__gs_08 | 0.00011 | 0.0156 | 0.000292 | False | calibration_residual;endpoint_ess;morphology_effect |

## Decision

Status: `phase_2_failed`.

no candidate passed the predeclared endpoint gates.

The pair is not a frozen Experiment C benchmark. Per protocol, reference-flow quality, interior I-projection ESS, projected morphology shift, and tangent blind-spot diagnostics must pass before `benchmark_selection.yaml` is created or final MFSI training begins.

Timestep convergence diagnostics are recorded in `metrics/timestep_convergence.csv`; all raw sample metrics and failed pairs are retained.

## Phase-2 feasibility continuation v6

The original `phase_2_failed` files above were not overwritten. Their SHA-256
hashes are recorded in
`results/grayscott/phase2_feasibility_v6/preserved_phase2_failed_sha256.json`,
and the post-run verification reports that every source artifact is unchanged.
Version 6 uses new design IC seeds 32000–32255, disjoint from the old design,
training, and final-evaluation roles.

### Exact diagnosis of the 64-sample near misses

The common-hull problem was solved directly with SciPy/HiGHS in the standardized
coordinates used by calibration. Constraints were nonnegative endpoint weights,
two unit-sum equalities, and exact equality of the endpoint weighted Phi means.
The two required pairs
`grid_F0340_k0620__grid_F0340_k0605` and
`grid_F0320_k0615__grid_F0320_k0600` are infeasible under Phi-4. Four additional
pairs selected by the declared method-blind rule are also infeasible. Thus no
target was declared feasible and no exponential tilt was applied to a compromise
target. Complete solver statuses are in `existing_bank_phi4_diagnostics.json`.

For feasible problems, the centrality procedure first maximizes the minimum
endpoint probability with a second HiGHS LP. It then maximizes total endpoint
entropy, `H(a)+H(b)`, through the convex log-sum-exp dual. A target is accepted
only from weights satisfying the equality constraints; an average of unmatched
endpoint means is never used.

### Calibration solver diagnosis

The v6 wrapper keeps the same exponential-tilt objective,
`log(mean(exp(Phi lambda))) - lambda^T c`, but replaces the fixed 20-iteration
policy with residual-based stopping, covariance-whitened damped Newton steps,
Armijo line search, and a 500-iteration safety limit. Final weights and moments
are independently recomputed by
`mfsi_components.empirical_tilt_from_lambda`.

On the ranked passing endpoint target, a separate zero-start check converged in
4 Newton steps for the spot bank and 5 for the labyrinth bank. Initial/final
dual objectives were `0 -> -0.0059935204` and `0 -> -0.4332116033`. Final maximum
standardized residuals were `1.64e-11` and `6.95e-11`; ESS fractions were
`0.98935` and `0.20922`. The direct expression
`weights @ Phi - c` agrees with the reported repository residual to at most
`1.11e-15`. Full iteration traces, covariance eigenvalues, ranks, condition
numbers, multipliers, and maximum weights are serialized in
`selected_candidate_zero_start_calibration.json`.

A synthetic full-rank regression test constructs an interior target from known
strictly positive empirical weights. It passes at residual below `1e-8`, ruling
out the old 20-iteration limit as the cause of the original six infeasible
near misses. Target selection and calibration use the identical stored affine
center/scale; observed round-trip error is at most `3.47e-18`.

### Larger bank and nested observation families

The required follow-up used 256 IC seeds per regime (4x the original bank) and
43 regimes on a local transition scan with kill-rate spacing `0.00025`. The
original simulator, IC generator, physical horizon, timestep gates, threshold,
and Fourier shells were unchanged. Thirty-eight regimes passed simulator gates;
the empirical classifier produced 13 spot-like and 13 labyrinth-like regimes,
giving 169 stable candidate pairs and 507 pair/dimension cells.

| observation family | exact LP-feasible pairs | endpoint-gate passers | best conclusion |
|---|---:|---:|---|
| Phi-2 `(mean, m2)` | 47 | 8 | Phase-2 feasible |
| Phi-3 `(mean, m2, S1)` | 35 | 6 | Phase-2 feasible |
| Phi-4 `(mean, m2, S1, S2)` | 28 | 0 | inadequate nondegenerate overlap |

The unchanged endpoint gates were residual `<=1e-5`, ESS `>=0.20`, hidden
morphology effect `>=1.0`, and stable simulator morphology. Every rejection and
all applicable gate reasons are retained in
`large_bank_nested_phi_results.csv`; simulator rejection reasons are in
`large_bank_regimes.csv`.

The predeclared design score ranks the Phi-2 pair
`v6_F0280_k06025__v6_F0280_k05800` first. Its maximum-entropy common target is
`(0.1182386970, 0.02567595335)` in physical `(mean,m2)` units, endpoint residual
is `2.31e-14`, minimum ESS is `0.20922`, hidden-morphology effect is `1.83305`,
and the maximum-minimum-weight LP retains 82.6% of the uniform per-particle
weight. This is a provisional Phase-2 candidate only. Phi-3 also has six valid
alternatives, including candidates above the preferred ESS `0.35` threshold.
The move from the originally recommended Phi-4 to a provisional Phi-2 design is
recorded as version 6 and was based only on feasibility, ESS, and retained hidden
morphology—not learned MFSI or tangent performance.

### Phase-3 path overlap check and current go/no-go

Phase 2 now passes, but Phase 3 does not. For the ranked Phi-2 candidate, the
maximum-same-IC coupling preserves exact calibrated endpoint marginals and puts
79.1% mass on identical IC indices. Under the unchanged linear Experiment-B
bridge, explicit time-local target-hull LPs show the fixed target is feasible at
only 4 of 9 design times in a 2048-particle screen. A separate 4096-particle run
has minimum interior ESS `0.000244` and infeasible middle-time targets.

All 14 Phase-2 passers were screened on `t=0.1,...,0.9`. Only one candidate had
LP-feasible targets at all nine times, and its minimum ESS was `0.00409`; zero
candidates passed the `0.15` interior ESS gate. Detailed time/pair decisions are
in `phase3_all_candidate_time_rows.csv` and
`phase3_all_candidate_summary.csv`.

**Go/no-go decision:** Phase 2 is passed provisionally under Phi-2/Phi-3, but
the current linear reference path fails Phase-3 overlap. No benchmark-selection
file is created. No reference velocity, tangent model, Deep-Ritz potential, or
final method comparison was trained or run. Phase 4 remains unevaluated until a
method-blind schedule/coupling design passes the Phase-3 overlap and reference-FM
quality gates.

## Phase-3 reference design and quality v7

Version 7 writes only below
`results/grayscott/phase3_reference_design_v7/`. A SHA-256 manifest was taken
of the complete v6 directory before this continuation and verifies unchanged
after every v7 design stage. All fourteen Phase-2 passers and their original
targets were retained; the provisional rank-1 Phi-2 pair was not frozen in
advance. Experiment B also remains unchanged.

### Why a linear field bridge loses the second moment

For every candidate/coupling, the empirical identity was evaluated directly:

`E[m2((1-s)X- + sX+)] = c2 - s(1-s) E[||X+ - X-||^2/N]`.

The maximum absolute numerical LHS/RHS discrepancy over all candidates,
couplings, times, and both the identity and smoothstep time maps was
`1.56e-10`. Smoothstep changes `s(t)` but leaves the strictly positive
`s(1-s)` deficit at every interior time. The complete table is
`linear_second_moment_identity.csv`.

Three exact-marginal couplings were compared on 4096-particle design banks:

| coupling | mean endpoint displacement RMS | range over 14 candidates |
|---|---:|---:|
| maximal same-IC mass | 0.1311 | 0.1195–0.1417 |
| geometric field-L2 OT | 0.1087 | 0.1001–0.1168 |
| independent diagnostic | 0.1528 | 0.1480–0.1571 |

Geometric OT is a sparse SciPy/HiGHS transportation LP with exact calibrated
marginals; its worst marginal residual was `6.38e-17`. Nevertheless, none of
the 42 linear candidate/coupling paths passed. The best linear all-time-feasible
path had minimum ESS `0.01267`, versus the gate `0.15`. This confirms that
geometrically shorter coupling alone does not provide adequate common interior.

### Minimal noisy schedule screen

Because no linear path passed, the prespecified repository-style schedule

`X_t = (1-t)X- + tX+ + A sin(pi t) Z`

was screened without clamping. `Z` is Gaussian, spatially centered, and
unit-RMS per field. The 462 paths comprise all fourteen candidates, all three
couplings, and eleven amplitudes from `0.02` through `0.14`, each at nine fixed
times. Targets and standardization were unchanged. The gate implementation
uses the declared residual threshold `1e-5`; the solver's stricter `1e-10`
target is retained as a separate convergence diagnostic and is not an
additional selection gate.

Nine schedule paths pass across two Phase-2 candidates. Candidate ordering
remains the predeclared Phase-2 order. The selected reference design is rank 7,
Phi-2 `v6_F0280_k05950__v6_F0280_k05800`, maximal-same-IC coupling, and
amplitude `0.07`. On the 4096-particle screen it has:

- all nine fixed targets LP-feasible and calibrated;
- minimum ESS `0.30440`;
- maximum standardized residual `6.27e-11`;
- mean projection KL `0.20002` and maximum lambda norm `4.30664`;
- nontrivial standardized hidden projection shifts at seven of nine times;
- field range `[-0.3335, 0.6371]`, with at most `8.82e-6` of pixels outside
  the declared hard range `[-0.25, 0.75]`.

An independent 8192-particle confirmation retains feasibility at all times,
minimum ESS `0.30735`, maximum residual `1.78e-8`, seven nontrivial hidden-shift
times, and maximum hard-range violation fraction `8.58e-6`. Three calibrations
stopped above the optional `1e-10` solver target (`1.66e-9`, `1.78e-8`, and
`4.66e-10`) but remain more than two orders of magnitude inside the frozen
`1e-5` gate. Phase 3A is therefore confirmed.

The other passing candidate is Phase-2 rank 11, Phi-2
`v6_F0280_k05950__v6_F0280_k05825`, geometric OT, amplitude `0.05`, with
minimum ESS `0.40208` and maximum residual `2.11e-9`. It is retained but does
not supersede rank 7 because cross-candidate ordering was frozen in Phase 2.

For the twelve rejected candidates, the strongest all-time screen outcome and
exact rejection are:

| Phase-2 rank | family | best coupling/amplitude | feasible times | min ESS | rejection |
|---:|---|---|---:|---:|---|
| 1 | Phi-2 | geometric / 0.06 | 9 | 0.12748 | ESS below 0.15 |
| 2 | Phi-2 | geometric / 0.07 | 9 | 0.02010 | ESS below 0.15 |
| 3 | Phi-2 | maximal / 0.08 | 9 | 0.02329 | ESS below 0.15 |
| 4 | Phi-3 | geometric / 0.03 | 8 | 0.00171 | one infeasible time; ESS; residual gate |
| 5 | Phi-2 | maximal / 0.08 | 9 | 0.06636 | ESS below 0.15 |
| 6 | Phi-3 | geometric / 0.02 | 9 | 0.02161 | ESS below 0.15 |
| 8 | Phi-3 | geometric / 0.02 | 9 | 0.01928 | ESS below 0.15 |
| 9 | Phi-3 | geometric / 0.03 | 9 | 0.00813 | ESS below 0.15 |
| 10 | Phi-3 | maximal / 0.03 | 9 | 0.00268 | ESS below 0.15 |
| 12 | Phi-2 | geometric / 0.06 | 9 | 0.12903 | ESS below 0.15 |
| 13 | Phi-2 | maximal / 0.08 | 9 | 0.08714 | ESS below 0.15 |
| 14 | Phi-3 | geometric / 0.03 | 9 | 0.01137 | ESS below 0.15 |

This table gives each rejected candidate's strongest near miss; every tested
path and every applicable reason (hull, residual, ESS, hidden shift, or field
range) remains in `schedule_path_summary.csv` and
`schedule_time_diagnostics.csv`.

### Phase-3B reference CNN result

The periodic translation-equivariant CNN was trained once for 4000 steps on
IC seeds 41001–41512. Model-selection IC seeds 47001–47128 are disjoint from
training and Phase-2 design seeds. The endpoint tilts on the training banks
have ESS `0.9675/0.2428`; on the smaller model-selection banks they have ESS
`0.9680/0.02984`, the latter correctly retained as a generalization warning.

Reference quality fails both predeclared gates:

| diagnostic | observed | gate | pass |
|---|---:|---:|---|
| held-out FM MSE / zero-predictor MSE | 0.42698 | <= 0.35 | no |
| 64-step rollout max standardized target residual | 1.78665 | <= 0.10 | no |

The held-out per-pixel MSE is `0.018894` versus baseline `0.044251`. Rollout
residual is `[-1.06679, -1.78665]` in standardized Phi-2 coordinates; hidden
endpoint error is `0.86227` standardized RMS and the rollout field range is
`[-0.1531, 0.3771]`. The failed quality result is not suppressed despite the
successful Phase-3A overlap design.

**Current go/no-go decision:** Phase 3A passes, but Phase 3B and therefore
Phase 3 fail. Phase 4 is not authorized and `B_tan` was not evaluated. No
`benchmark_selection.yaml` was created, no Deep-Ritz MFSI model was trained,
and no learned-method comparison was run. The next scientifically valid action
is to improve the reference model/training protocol under a new versioned,
predeclared Phase-3B design and re-evaluate it on disjoint seeds; the Phase-3A
path and all failures remain frozen as v7 evidence.

## Phase-3B reference-quality continuation v8

This continuation keeps the v7 Phase-3A construction exactly fixed: Phi-2,
`v6_F0280_k05950__v6_F0280_k05800`, the exact-marginal maximal-same-IC
coupling, the frozen target and affine standardization, and the
`0.07 sin(pi t) Z` bridge without clamping. v6 and v7 are source artifacts;
v8 writes only to `results/grayscott/phase3_reference_quality_v8`.

### Corrected reference-fidelity semantics

The v7 rollout-to-fixed-`c` test is retained as historical output but is not a
valid raw-reference gate. The stochastic interpolant is an unprojected moving
prior, so its raw intermediate law need not lie on the fixed moment fiber.
The Experiment B audit agrees: reference selection uses held-out
stochastic-interpolant velocity regression, while rollout validation compares
the generated law against independently sampled oracle/interpolant laws. v8
therefore separates three quantities at every reported time:

1. learned raw rollout versus direct raw stochastic interpolant (reference
   fidelity);
2. direct raw stochastic interpolant Phi versus frozen `c` (descriptive only);
3. empirical I-projection Phi versus `c` (projection correctness).

The old v7 value `max |Phi(rollout)-c| = 1.78665` was not deleted or relabeled
as a pass. Under the corrected comparison, the same checkpoint has maximum
learned-versus-direct standardized Phi error `2.35977`, integrated raw-field
MMD2 `0.0140219`, integrated downsampled-field MMD2 `0.0146034`, and endpoint
Phi error `1.87243`; it still fails, now for the scientifically relevant
reason.

### Target, coupling, and validation-bank audit

The analytic FM target agrees with a shared-noise central finite difference to
relative error `2.02e-5` (maximum absolute error `2.89e-5`) and with the closed
form to `1.19e-7`. State and target are float32, the same sampled `Z` is used
in both, the CNN input gradient norm is `0.06369`, and no accidental
stop-gradient was found. Endpoint coupling marginal errors are
`4.34e-19/1.39e-17`; standardized noise has maximum spatial-mean magnitude
`9.31e-9` and maximum RMS error `5.96e-8`. The optimized objective remains
uniform continuous time, with stratified uniform draws used only for the fixed
audit.

The old 128-particle model-selection bank had minimum endpoint ESS `0.02984`.
The predeclared replacement rule generated the first contiguous 1024-seed
chunk, seeds 61001--62024, and accepted it without changing `c`:

| endpoint | max standardized residual | ESS fraction | entropy fraction | max weight | covariance condition |
|---|---:|---:|---:|---:|---:|
| minus | `6.66e-16` | `0.96855` | `0.99763` | `0.002265` | `19.85` |
| plus | `1.02e-14` | `0.51489` | `0.97452` | `0.010017` | `486.05` |

Both endpoint residuals remain far inside `1e-5`; the minimum ESS exceeds the
preferred `0.25` level. No extra chunk was added.

### Saved-v7 checkpoint diagnosis

| evaluation bank | normalized FM MSE | cosine alignment | prediction/target velocity RMS | low/mid/high frequency error fraction |
|---|---:|---:|---:|---|
| old degenerate model-selection bank | `0.42649` | `0.7573` | `0.1596/0.2101` | `0.729/0.580/0.085` |
| fresh draws from training endpoint bank | `0.33389` | `0.8162` | `0.1629/0.1983` | `0.626/0.458/0.077` |
| healthy independent validation bank | `0.33718` | `0.8141` | `0.1624/0.1985` | `0.634/0.455/0.077` |

The new healthy bank and fresh training-bank draws both pass the local `0.35`
gate and agree closely. The old bank's failure is therefore a finite-bank
importance pathology, not evidence of ordinary endpoint-bank generalization
failure. The per-time normalized error nevertheless ranges from `0.213` to
`0.965` on the healthy bank and is worst at `t=0`; low-frequency error remains
dominant. The training-trace tail is still slightly decreasing. The combined
diagnosis is (i) a bad old validation bank and (ii) rollout accumulation from
imperfect conditional-velocity approximation. There is no positive evidence
that float precision, target construction, marginal coupling, or a detached
input caused the failure.

### Predeclared controlled sweep

Direct-SI split variability fixed time-specific Phi and raw/downsampled MMD
thresholds before training. Model choice used only healthy-validation FM and
direct-SI reference fidelity; hidden morphology, projected-target, tangent,
MFSI, and future comparison performance were excluded.

| variant | parameters | normalized FM | integrated raw MMD2 | integrated downsampled MMD2 | max / endpoint Phi error | failed time points | result |
|---|---:|---:|---:|---:|---:|---:|---|
| A: exact saved v7 | 17,593 | `0.33685` | `0.0140215` | `0.0146031` | `2.35976 / 1.87242` | 18 | reject: rollout |
| B: v7 + 8,000 steps | 17,593 | `0.25176` | `0.0090991` | `0.0047269` | `1.51165 / 0.37683` | 12 | reject: rollout Phi |
| C: residual receptive CNN, 12,000 steps | 37,717 | `0.19496` | `0.0078729` | `0.0036646` | `0.93136 / 0.31297` | 8 | reject: rollout Phi |

All variants pass the local FM gate and the declared serious field-range gate.
Variant C is the diagnostic best model, but it fails the predeclared Phi gate
at `t = 0.203125, 0.296875, 0.3515625, 0.3984375, 0.453125, 0.5,
0.546875, 0.6015625`. It is not promoted based on being the best near miss.
The per-time tables retain raw/direct/projected Phi, field MMDs, smooth hidden
observables, field extrema, empirical projection residual, and ESS. Near-flat
states can make the descriptive structure-tensor anisotropy ratio numerically
sensitive; it is not used for selection.

For variant C, the common-grid ODE audit gives:

| Heun steps | integrated raw MMD2 | integrated downsampled MMD2 | max Phi error | endpoint Phi error |
|---:|---:|---:|---:|---:|
| 64 | `0.0074533` | `0.0050975` | `1.06596` | `0.29202` |
| 128 | `0.0071739` | `0.0050598` | `0.99788` | `0.28264` |
| 256 | `0.0070485` | `0.0050530` | `0.96125` | `0.28121` |

The 128-to-256 downsampled-MMD improvement is only `0.134%`, so integration
under-resolution is not the explanation.

**Current go/no-go decision:** Phase 3B remains failed. No sweep candidate
satisfies both local conditional-velocity and direct-SI rollout fidelity.
Phase 4 is not authorized; `B_tan` was not computed. No Deep-Ritz MFSI model,
final learned-method comparison, benchmark selection, or simplification was
performed. The remaining obstacle is accumulated raw-flow error from the
conditional-velocity approximation, concentrated in morphology-scale Phi
modes. The next run must be another versioned reference-only investigation;
v8 does not silently continue training or alter the frozen Phase-3A path.

## Final global spectral reference attempt v9

### Frozen diagnosis and hypothesis

v9 consumes the exact v7/v8 Phase-3A geometry read-only: Phi-2, endpoint pair
`v6_F0280_k05950__v6_F0280_k05800`, frozen target and affine
standardization, exact-marginal maximal-same-IC coupling, and the unclamped
`0.07 sin(pi t) Z` bridge. The healthy 1024-seed validation bank remains at
minimum endpoint ESS `0.51489`. Exact target, center, and scale equality was
checked before training. Pre/post SHA-256 manifests show no change in v6, v7,
or v8.

The final rescue hypothesis was that v8's approximately 43-pixel receptive
field allowed small systematic domain-scale velocity errors that accumulated
under rollout. v9 therefore tested one explicitly global periodic spectral
family. It did not redesign the benchmark or recompute the v8 thresholds.

### Predeclared spectral model and protocol

Before held-out rollout evaluation, v9 froze a compact periodic Fourier neural
operator with width 32, four spectral residual blocks, one positive and one
negative 12x12 low-mode multiplier per block, a parallel physical-space
pointwise path, SiLU activations, residual connections, and a direct
input/output skip. Raw time plus three Fourier time frequencies are spatially
constant; no positional coordinates, clipping, or nonperiodic operations are
used. The model has 2,364,899 float32 parameters and full-domain dependence.
Twelve modes reach 11/64 cycles per pixel, covering global and much of the
morphology-scale band without learning the full 64x64 spectrum.

The sole standard run used AdamW, cosine learning rate `8e-4 -> 1e-5`, weight
decay `1e-6`, global gradient clip 5, batch 32, continuous uniform time, and a
hard cap of 18,000 steps. It used only the frozen raw-SI flow-matching loss.
Checkpoint selection used only healthy-validation normalized FM MSE. The best
checkpoint occurred at the final step, with validation ratio `0.15616`; total
training time was 208 seconds. The v8 control checkpoint was not retrained.

The architecture, seeds, adaptation trigger, optimizer, step cap, and exact v8
threshold source are serialized in `v9_predeclared_protocol.json`.

### Paired FM and frequency diagnosis

Both models were evaluated on the same fresh direct-SI draws:

| diagnostic | v8 residual CNN control | v9 global spectral |
|---|---:|---:|
| parameters | 37,717 | 2,364,899 |
| training steps | 12,000 total | 18,000 |
| FM MSE per pixel | `0.0075926` | `0.0062199` |
| normalized FM MSE | `0.19269` | `0.15785` |
| cosine alignment | `0.89850` | `0.91772` |
| predicted / target velocity RMS | `0.17822 / 0.19850` | `0.18385 / 0.19850` |
| low-frequency error / target energy | `0.35755` | `0.27985` |
| middle-frequency error / target energy | `0.32222` | `0.26655` |
| high-frequency error / target energy | `0.03471` | `0.03828` |

The spectral model passes the unchanged local `0.35` gate comfortably and
does address the proposed local frequency defect: at the 21 fixed times it has
lower low-band error at 19 times and lower middle-band error at 17. Mean
per-time low-band error fraction drops from `0.37772` to `0.31475`; mean
middle-band error drops from `0.46570` to `0.43753`. The high-band aggregate is
slightly worse. Full fixed-time band energies and ratios are retained in
`paired_fm_by_time.csv` and `paired_fm_radial_bands_by_time.png`.

Short-horizon diagnostics do not convert that local gain into better flow.
Across the frozen 1/16, 1/8, and 1/4 horizons, maximum standardized Phi error
is `0.46733` for v8 and `0.78441` for the spectral model. Mean low-frequency
power MMD2 is `0.00885` versus `0.02398`. These diagnostics were never used as
a training loss.

### Direct-SI rollout comparison

Reference fidelity remains learned raw rollout versus independently sampled
direct raw SI. Raw-SI Phi versus `c` and empirical I-projection versus `c`
remain separate descriptive and correctness diagnostics.

| paired 128-step diagnostic | v8 residual CNN control | v9 global spectral |
|---|---:|---:|
| maximum standardized Phi discrepancy | `0.89646` | `1.43550` |
| endpoint standardized Phi discrepancy | `0.37150` | `1.18274` |
| integrated raw MMD2 | `0.0093037` | `0.0104678` |
| integrated downsampled MMD2 | `0.0042814` | `0.0054428` |
| failed time points | 6 | 13 |
| serious field-range fraction | `0` | `0` |
| local FM gate | pass | pass |
| complete rollout gate | fail | fail |

All raw and downsampled MMD thresholds pass for both models. Failure is
entirely the frozen time-specific Phi gate. The spectral failures occur at
`t = 0.203125, 0.25, 0.296875, 0.3515625, 0.3984375, 0.453125, 0.5,
0.546875, 0.6015625, 0.6484375, 0.703125, 0.75, 0.953125`. Learned/direct
mean and second-moment trajectories, radial powers, smooth hidden statistics,
MMD trajectories, empirical projection residual/ESS, and extrema are retained
per time in `paired_rollout_by_time.csv`.

Thus genuine global context improves the supervised conditional-velocity
regression, including its low-frequency component, but makes accumulated Phi
transport substantially worse. The v9 hypothesis is only locally supported
and fails at the flow-realization level.

### ODE convergence

The common-grid paired resolution audit gives:

| model | Heun steps | integrated raw MMD2 | integrated downsampled MMD2 | max Phi error | endpoint Phi error |
|---|---:|---:|---:|---:|---:|
| v8 control | 64 | `0.0086503` | `0.0036223` | `0.94366` | `0.26466` |
| v8 control | 128 | `0.0083049` | `0.0035729` | `0.87647` | `0.25449` |
| v8 control | 256 | `0.0081500` | `0.0035624` | `0.84040` | `0.25261` |
| v9 spectral | 64 | `0.0117300` | `0.0063289` | `1.54927` | `1.09695` |
| v9 spectral | 128 | `0.0118201` | `0.0065662` | `1.55150` | `1.13274` |
| v9 spectral | 256 | `0.0118329` | `0.0066614` | `1.54952` | `1.13449` |

For v9, 128-to-256 refinement changes integrated downsampled MMD by `-1.45%`
(a slight worsening), and maximum Phi error is essentially unchanged. The
failure is model-flow error, not ODE under-resolution.

### Optional rescue and hard stopping decision

The single optional harmonic velocity adaptation was predeclared but not
permitted. Although local FM and field-range conditions pass and all MMD gates
pass, the spectral model fails 13 rather than at most 4 times, its largest Phi
error is `3.51x` its applicable threshold rather than at most `1.35x`, maximum
Phi error worsens by `60.1%` relative to the paired v8 control, and integrated
downsampled MMD worsens by `27.1%`. It is neither an improving model nor a
genuine near miss. No adaptation seeds were consumed and no repeated tuning
occurred.

Phase 3B therefore fails after the final permitted reference family.
`gray_scott_reference_failed_after_v9 = true`. Endpoint fiber feasibility and
intermediate empirical I-projection overlap remain successful; the obstacle is
faithful realization of the raw stochastic-interpolant reference flow.

Phase 4 is not authorized and was not run. No `B_tan`, tangent comparison,
Deep-Ritz/MFSI training, final learned-method comparison,
`benchmark_selection.yaml`, endpoint/Phi/target/coupling change, threshold
relaxation, or further architecture search was performed. Gray–Scott is
retained as a documented negative/high-complexity benchmark-design result and
is parked as the headline Experiment C path.

`GRAY_SCOTT_PARKED_AFTER_V9`

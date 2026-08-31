# Frozen inputs

This directory contains the compact numerical inputs needed by the standalone vortices experiment and its publication visualizations. These files are immutable inputs rather than regenerated outputs.

| File | Role | SHA-256 |
| :--- | :--- | :--- |
| `reference_endpoints.npz` | Common 50,000-particle endpoint dataset used to train the three reference models | `ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba` |
| `truth_bank.npz` | Frozen 50,000-particle double-gyre truth trajectory used for observations and visualization | `d897ff7fc44c0b85d7bb5391c0cc25895b4301e9c2ce00184697a1899d853b5b` |
| `visualization_reference_bank.npz` | Qualified seed-`310000101` reference rollout used by the static figures and GIF | `8cde1314f978aeb612338b059fe3500916c26cb07803fbee9d062e381fe0e140` |
| `visualization_holdout_bank.npz` | Frozen namespace-`23`, 64-trial independent holdout bank used by the figures and GIF | `aea3955915df37dd3282c13273d1093506fe27105ab421111dff98c4a5f3f1ac` |

The two visualization banks are sufficient to reproduce the published snapshots and animation without retraining a reference or rerunning selection and confirmation. Their qualification and identity receipts are exposed under [`../outputs/published`](../outputs/published/README.md).

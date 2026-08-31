# Published result artifacts

This directory is the compact, tracked output surface for the confirmed V2.1 vortices result. It contains the numerical records, inferential receipt, frozen geometries, and provenance needed to verify the reported result and regenerate every plot without the 16 GB development/search tree.

| File | Role |
| :--- | :--- |
| `pareto_data.json` | Complete Law, Tangent, and Full selection/holdout coordinates used by the Pareto renderers |
| `simultaneous_inference.json` | Reference-wise effects and pointwise/simultaneous intervals for the primary Full-versus-Law family |
| `selection_geometries.json` | Frozen four-sensor Full geometries at 0.5%, 1%, and 2% |
| `selection_pause_receipt.json` | Proof that the 3%–5% branches remained paused |
| `execution_receipt.json` | Identity of the accepted 64-trial confirmatory execution |
| `exact_action_evaluation_receipt.json` | Exact-action completion and numerical-gate receipt |
| `finite_risk_evaluation_receipt.json` | Independent finite-risk cross-evaluation receipt |
| `relative_action_metrics_receipt.json` | Supplementary relative Full/Tangent action metrics used by the comparison plots |
| `tangent_evaluation_receipt.json` | Supplementary Tangent evaluation used by the comparison plots |
| `holdout_bank_receipt.json` | Identity and hash of `inputs/visualization_holdout_bank.npz` |
| `reference_qualification_receipt.json` | Qualification and rollout hash for `inputs/visualization_reference_bank.npz` |
| `manifest.json` | Relocation-independent SHA-256 inventory of this publication bundle |

Run `.venv/bin/python experiments/vortices_percentage/verify_saved_result.py --json` to verify the bundle. Run `.venv/bin/python experiments/vortices_percentage/render_v2_1_c3_64_pareto.py`, `.venv/bin/python experiments/vortices_percentage/visualize_v2_1_partial_paper.py`, and `.venv/bin/python experiments/vortices_percentage/visualize_v2_1_partial_paper_gif.py` to regenerate the plots, snapshots, and GIF from these exposed artifacts.

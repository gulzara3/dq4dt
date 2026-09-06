# Figures

PDF (vector) and PNG (300 dpi) versions of every figure in the paper, plus `architecture.png`.

| File | Paper figure | Content |
|---|---|---|
| architecture.png | Fig. 1 | pipeline and formalism |
| fig6_convergence | Fig. 2 | training audit |
| fig7_cross_dataset | Fig. 3 | cross-dataset MNAR overlay (invariant collapse point) |
| fig1_dose_response_FD001 / FD004 | Fig. 4 | dose-response curves per dimension |
| fig2_elasticity_FD001 / FD004 | Fig. 5 | elasticity heatmaps |
| fig3_interactions | Fig. 6 | interaction panels |
| fig8_injection_points | Fig. 7 | lifecycle stage of the fault |
| fig9_qcg_ablation | Fig. 8 | gating ablation |
| fig5_iso_fidelity | Fig. 9 | iso-fidelity budget surface |
| fig4_calibration | Fig. 10 | reliability diagram before and after sigma scaling |

File names follow the notebook's numbering; the paper figure numbers differ because the manuscript reorders them. `scripts/make_figures.py` regenerates the CSV-driven figures and adds `fig10_coverage_decay` (coverage against severity), which is not in the notebook.

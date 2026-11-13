# Results and figure guide

## Core result files

| Analysis | File | Meaning |
|---|---|---|
| Temporal calibration | results_real/fit_metrics.csv | Calibration and holdout metrics |
| Temporal parameter estimates | results_real/predictions_and_params.npz | Archived predictions and PINN parameter set |
| Full-system continuation | results_formal_verification/verification_report.txt | Fold/Hopf screen and validation summary |
| Bifurcation robustness | results_robustness/robustness_summary.csv | Parameter-perturbation audit |
| Reduced cusp scenario | figure/fig9_bifurcation.png | Exploratory normal-form hysteresis |
| Turing compatibility | results_v4/turing_summary.csv | Reduced reaction--diffusion linear check |
| Spatial holdout | results_spatial_pinn/residual_multiwindow_metrics.csv | Per-window model and baseline metrics |
| Spatial pattern decision | results_spatial_pinn/residual_holdout_assessment.csv | Pre-specified pattern and level pass/fail |
| Standard vs residual | results_spatial_pinn/standard_vs_residual_holdout.csv | Direct negative-control comparison |
| Cross-basin comparison | results_cross_basin/cross_basin_metrics.csv | Descriptive Tarim--Heihe statistics |

## Manuscript figure mapping

| Figure | File |
|---|---|
| Fig. 1 observed annual series | figure/fig1_observed_series.png |
| Fig. 2 training loss | figure/fig_training_loss.png |
| Fig. 3 fit and holdout | figure/fig3_fit.png |
| Fig. 4 parameter estimates | figure/fig4_parameter_boxplots.png |
| Fig. 5 scenarios | figure/fig5_scenarios.png |
| Fig. 9 reduced cusp scenario | figure/fig9_bifurcation.png |
| Fig. 15 qualitative spatial reference | figure/fig15_tarim_patch_maps.png |
| Fig. 16 patch mosaic | figure/fig16_spatial_patch_mosaic.png |
| Fig. 17 Turing verification | figure/fig17_turing_verification.png |
| Full-system continuation | figure/fig_formal_continuation.png |
| Cross-basin trends | figure/fig_cross_basin_trends.png |

The exact figure captions and current wording are in paper/Manuscript.pdf and paper/Supplementary_Material.pdf.

## Interpretation boundaries

- The temporal PINN is not a validated forecasting model.
- The full calibrated system yielded no validated fold or Hopf point.
- The 0.35/0.65 values in the cusp figure are reduced-normal-form scenario coordinates.
- The reduced Turing model is a compatibility test, not a spatial fit to the Tarim landscape.
- The residual spatial PINN supports pattern-level evidence only. It does not establish basin-wide spatial prediction.
- The Heihe comparison is descriptive, not an independent calibration or validation.

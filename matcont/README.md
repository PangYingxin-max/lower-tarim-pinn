# Optional MATCONT/COCO verification

The Python continuation used for the archived results is in formal_ode_verification.py and full_ode_bifurcation.py. The MATLAB files here are optional templates for an independent continuation check:

- VegetationWater.m: right-hand side of the calibrated three-state system.
- run_matcont_hopf.m: MATCONT starter for equilibrium continuation and Hopf detection.
- run_coco_hopf.m: COCO-compatible continuation template.

MATCONT or COCO is not required to read the repository or reproduce the archived Python screen. Any new continuation run should be reported with its parameter range, step size, boundary conditions, and eigenvalue tolerances.

# Physics Informed Machine Learning Examples
Tutorial codes demonstrating a variety of PIML methods.
Comparison methods:
1. Pure physics
1. Pure NN

PIML methods:
1. Indirect measurement
1. PINN
1. Delta learning
1. Informed structure

# To generate data
1. In `./simulink`, run `run_experiments_v2.m` to create .csv files for `no_friction` dataset.
1. In `run_experiments_v2.m`, change `with_friction` to true. Run the file again to generate `with_friction` dataset.
1. Run `run_pinn_v1.m` to create PINN dataset.
1. Run `to_numpy.py` to convert .csv files to .npy.

# Files too large for GitHub
This repo has some files too large for GitHub. We may try to get these to work with lfs in the future, for now you can find the files [here](https://www.dropbox.com/scl/fo/p7u6dwy1t8o83hk3dgnlz/AO3S344MKnTcE55aHe9J2Gs?rlkey=6aor3ivq2wrm9j06h62nnedgn&dl=0).

The files are:
1. PIML examples/data/v5/no_friction.npy - replaced with npz file that should push
1. PIML examples/data/v5/with_friction.npy - replaced with npz file that should push
1. PIML examples/model_predictions/pure_physics/k_pred.npy - under 200 MB, I thought this would push.

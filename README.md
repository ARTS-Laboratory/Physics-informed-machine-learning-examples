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

# Files too large for GitHub
This repo has some files too large for GitHub. The work around it to generate these files on your own. 
1. In `physics based modeling`, run `to_numpy.py` to convert .csv files to .npy. This will save `.npz` versions of data in the `PIML examples/data/`.

We may try to get these to work with lfs in the future, for now you can find the files [here](https://www.dropbox.com/scl/fo/p7u6dwy1t8o83hk3dgnlz/AO3S344MKnTcE55aHe9J2Gs?rlkey=6aor3ivq2wrm9j06h62nnedgn&dl=0).

The files are:
1. PIML examples/data/v5/no_friction.npy
1. PIML examples/data/v5/with_friction.npy
1. PIML examples/model_predictions/pure_physics/k_pred.npy (May or may not bee needed. )


# Version control
Tensorflow 2.10.0 (Nile) and 2.13.0 (Austin) was used to develop this code.






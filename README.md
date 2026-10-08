# Physics-Informed Machine Learning Examples
Working Repo for a journal tutorial paper on PIML 

Tutorial code demonstrating a variety of PIML methods. 



Comparison methods:
1. Pure physics
1. Pure NN

PIML methods:
1. Indirect measurement
1. PINN
1. Delta learning
1. Informed structure

## Files too large for GitHub
This repo has some files too large for GitHub. The work around it to generate these files on your own. 
1. In `physics based modeling`, run `to_numpy.py` to convert .csv files to .npy. This will save `.npz` versions of data in the `PIML examples/data/`.

We may try to get these to work with lfs in the future, for now you can find the files [here](https://www.dropbox.com/scl/fo/p7u6dwy1t8o83hk3dgnlz/AO3S344MKnTcE55aHe9J2Gs?rlkey=6aor3ivq2wrm9j06h62nnedgn&dl=0).

The files are:
1. PIML examples/data/v5/no_friction.npy
1. PIML examples/data/v5/with_friction.npy
1. PIML examples/model_predictions/pure_physics/k_pred.npy (May or may not bee needed. )


## Version control
Tensorflow 2.10.0 (Nile) and 2.13.0 (Austin) was used to develop this code.

##  Related Publications
Building on the following conference papers:
1.  Eleonora Maria Tronci, Austin R.J. Downey, Azin Mehrjoo, Puja Chowdhury, and Daniel Coble. Physics-informed machine learning part I: Different strategies to incorporate physics into engineering problems. In Conference Proceedings of the Society for Experimental Mechanics Series. Springer Nature Switzerland, 2024. doi:10.1007/978-3-031-68142-4_1 
1.  Austin R.J. Downey, Eleonora Maria Tronci, Puja Chowdhury, and Daniel Coble. Physics-informed machine learning part II: Applications in structural response forecasting. In Conference Proceedings of the Society for Experimental Mechanics Series. Springer Nature Switzerland, 2024. doi:10.1007/978-3-031-68142-4_8
1.  Mohsen Gol Zardian, Austin R. J. Downey, Eleonora Maria Tronci, Conor Madden, Daniel Coble, Sina Navidi, and Chao Hu. Physics-informed machine learning part III: Hard-constraint ODE method for structural dynamics. In Proceedings of the Society for Experimental Mechanics (SEM) IMAC Conference, 2026 
1.  Eleonora Maria Tronci, Austin R. J. Downey, Connor Madden, Mohsen Gol Zardian, and Daniel Coble. Physics-informed machine learning part IV: Weight-tuned soft-constraint method for structural dynamics. In Proceedings of the Society for Experimental Mechanics (SEM) IMAC Conference, 2026



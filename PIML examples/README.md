# PIML Example

## Codes
1. [train_pinn_control](train_pinn_control.py)
    * Code runs and will train epochs
1. [train_pinn](train_pinn.py)
    * Code will train epochs
1. [plot_results](plot_results.py)
    * Code runs and generates plots
1. [train_pure_nn](train_pure_nn.py)
    * Will train epochs
1. [solve_pure_physics.py](solve_pure_physics.py)
    * Runs and starts running tests.
1. [test_delta.py](test_delta.py)
    * Code starts and runs
1. [test_indirect.py](test_indirect.py)
    * Code runs, no errors
1. [analyze_pinn.py](analyze_pinn.py)
    * Code crashes kernal
    * V2 created to work on the crashing issue. 
    * V2 line 173 "model = keras.models.load_model('./model_saves/pinn')" kills the kernal.
    * It only points to a directiry, shoud it point to a file? There are two files there 1) model_saves/pinn/saved_model.pb and model_saves/pinn/keras_metadata.pb
    * I get this error, but only the first time I open spyder 
        * 2024-12-14 09:54:33.528287: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
        * To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
        * OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.
        * OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/
    * A workaround for the error was found  
    [here](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial)
    and the I added a break statment before the last iteration of the for loop.
1. [train_delta_learning_control.py](train_delta_learning_control.py)
    * Will start training epochs
1. [delta_learning_v1.py](delta_learning_v1.py)
    * Will start training epochs
1. [indirect_no_friction_v1.py](indirect_no_friction_v1.py)
    * Code runs and starts training epochs"
1. [test_indirect.py](test_indirect.py)
    * Code starts running and training epochs
1. [train_informed_structure.py](train_informed_structure.py)
    * Code starts to train epochs.
    * Code did not run" OSError: Cannot parse keras metadata at path ./model_saves/pure_nn\keras_metadata.pb: Received error: Field number 0 is illegal."
    * Per Nile in email, "This seems like a tensorflow version error. I was using tensorflow 2.13.0, it's probably easiest to retrain the pure NN in your environment. Either way, the lines that cause this error (119-124) are completely non-essential and can be commented out if you want."
    * Lines 119-124 were commented out. 
    * Code now starts to train epochs.
1. [metrics_v1.py](metrics_v1.py)
    * runs and finishes

## Open Questions
1. ...




























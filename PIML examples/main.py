"""
Run each training file.
"""
from pure_physics_v2 import main as pure_physics
from pure_nn_v1 import main as pure_nn
from indirect_v3 import main as indirect
from pinn_v1 import main as pinn
from pinn_control_v1 import main as pinn_control
from delta_learning_v1 import main as delta_learning
from delta_learning_control_v1 import main as delta_learning_control
from informed_structure_v1 import main as informed_structure

if __name__ == '__main__':
    training_funcs = [
        pure_physics,
        pure_nn,
        indirect,
        pinn,
        pinn_control,
        delta_learning,
        delta_learning_control,
        informed_structure
    ]
    training_names = [
        'pure physics',
        'pure nn',
        'indirect',
        'pinn',
        'pinn control',
        'delta learning',
        'delta learning control',
        'informed structure'
    ]
    
    for func, name in zip(training_funcs, training_names):
        print('running script', name)
        func()
        
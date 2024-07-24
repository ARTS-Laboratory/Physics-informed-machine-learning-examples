import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN
from numpy.lib.stride_tricks import sliding_window_view
"""
ML model with physics-integrated components does indirect measurement of k
to solve inverse problem.
"""

class SpringMass(Layer):
    
    '''
    Implemented solvers are:
        soln (exact solution assuming constant F)
        rk4 (Runge-Kutta method)
        euler (forward Euler)
    '''
    def __init__(self, dt, m, c, k_call, solver='rk4', **kwargs):
        self.dt = dt
        self.m = m
        self.c = c
        self.k_call = k_call
        self.solver = solver
        self.state_size = [tf.TensorShape([1]), tf.TensorShape([1])]
        self.output_size = (1,)
        if(solver == 'rk4'):
            # precompute rate of decay and its square
            self.r = self.c/(2*self.m)
            self.r2 = self.r**2
        super(SpringMass, self).__init__(**kwargs)
        self.build(input_shape=[k_call.input_shape, [None, 1]])
        self.built = True
    
    def get_config(self):
        return {"dt": self.dt, "m": self.m, "c": self.c}
    
    '''
    redirect call to the implementation of the chosen solver
    
    inputs: [y, F] where y is the input to k_call and F is forcing
    
    returns output, [states] as reconstructed xddot [updated xdot, x]
    '''
    def call(self, inputs, states):
        if(self.solver == 'soln'):
            return self.soln(inputs, states)
        elif(self.solver == 'rk4'):
            return self.rk4(inputs, states)
        elif(self.solver == 'euler'):
            return self.euler(inputs, states)
        elif(self.solver == 'lugre'):
            return self.lugre(inputs, states)
    
    '''
    Exact solution to damped harmonic motion assuming F is constant over dt.
    '''
    def soln(self, inputs, states):
        y, F = inputs
        xdot0, x0 = states
        k = self.k_call(y)
        dt = self.dt
        omega_d = tf.sqrt(k/self.m - self.r2)
        B = x0
        A = (xdot0 + x0*self.r)/omega_d
        
        # We use the autodifferentiation engine to save calculating xdot
        with tf.GradientTape() as tape:
            tape.watch(dt)
            x = A*tf.sin(omega_d*self.dt) + B*tf.cos(omega_d*self.dt)
            x = tf.exp(-self.r*dt)*x
        xdot = tape.gradient(x, dt)
        xddot = (-self.c*xdot - k*x + F)/self.m
        return xddot, [xdot, x]
    
    '''
    Runge-Kutta method.
    '''
    def rk4(self, inputs, states):
        y, F = inputs
        xdot0, x0 = states
        k = self.k_call(y)
        dt = self.dt
        halfdt = 0.5*dt
        def f(xdot, x):
            xddot = (-self.c*xdot - self.k*x + F)/self.m
            
            return xddot, xdot
        
        xdot1, x1 = f(xdot0, x0)
        xdot2, x2 = f(xdot0 + halfdt*xdot1, x0 + halfdt*x1)
        xdot3, x3 = f(xdot0 + halfdt*xdot2, x0 + halfdt*x2)
        xdot4, x4 = f(xdot0 + dt*xdot3, x0 + dt*x3)
        
        xdot = xdot0 + dt/6*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
        x = x0 + dt/6*(x1 + 2*x2 + 2*x3 + x4)
        xddot = (-self.c*xdot - k*x + F)/self.m
        return xddot, [xdot, x]
    
    '''
    Forward Euler method.
    '''
    def euler(self, inputs, states):
        y, F = inputs
        xdot0, x0 = states
        k = self.k_call(y)
        xddot = 1/self.m*(F - self.c*xdot0 - k*x0)
        
        xdot = self.dt*xddot
        x = self.dt*xdot0
        xddot = -self.c*xdot - k*x + F
        return x, [xdot, x]
    
    """
    call as a test
    """
    def lugre(self, inputs, states):
        sz_0 = states[0]
        v, F_c, F_s = tf.split(inputs, 3, -1)
        
        model_in = tf.concat([v, sz_0], -1)
        
        sigma_0 = self.sigma_call(model_in)
        
        sigma_1 = tf.abs(self.sigma_1); sigma_2 = tf.abs(self.sigma_2);
        v_s = self.v_s; alpha = self.alpha;
        sign = tf.sign(v)
        
        dif = F_s - F_c
        g = F_c + dif*tf.exp(negative(tf.pow(tf.abs(v/v_s), alpha)))
        k = g*sign
        
        sz = k + (sz_0 - k)*tf.exp(tf.scalar_mul(-self.dt, sigma_0*tf.abs(v)/g))
        zdot = v - sz*tf.abs(v)/g
        sz = tf.clip_by_value(sz, tf.negative(F_s), F_s) # protection
        F = sz + sigma_1*zdot + sigma_2*v
        return F, [sz]

'''
RNN Layer for using SpringMass layer 
'''
class SpringMassRNN(RNN):
    
    def __init__(self, dt, m, c, k_call, solver='rk4',
                 return_sequences=False, stateful=False, **kwargs):
        self.dt = dt
        self.m = m
        self.c = c
        self.k_call = k_call
        self.solver = solver
        self.cell = SpringMass(dt, m, c, k_call, solver=solver)
        # self.output_shape = 
        super(SpringMassRNN, self).__init__(
            self.cell,
            return_sequences=return_sequences,
            stateful=stateful,
            **kwargs
        )
    
    def get_config(self):
        return {"dt": self.dt, "m": self.m, "c": self.c}

"""
The training generator creates a sort-of virtual array so that passing
over the data per epoch his done optimally.
"""
class TrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, x, v, a, F, train_len=50, y_len = 50):
        self.x = x
        self.v = v
        self.a = a
        self.F = F
        self.train_len= train_len
        self.y_len = y_len
        
        self.N = x[0].shape[0]
        self.T = self.x.shape[1] - (self.train_len+y_len) + 1
    
    def __len__(self):
        return self.N*self.T
    
    def __getitem__(self, index):
        i = index // self.T
        j = (index % self.T) + self.y_len + self.train_len
        
        v_init = self.v[i:i+1,j-train_len-1]
        x_init = self.x[i:i+1,j-train_len-1]
        F_input = self.F[i:i+1,j-train_len:j]
        a_output = self.a[i:i+1,j-train_len:j]
        y_input = sliding_window_view(self.a[i:i+1,j-self.y_len-self.train_len+1:j], [self.y_len], axis=1)
        
        return [y_input, F_input, v_init, x_init], a_output
#%% load data
all_data = np.load('./data/all_data.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]

# divide train and test
x_train = x[:80]; x_test = x[80:]
v_train = v[:80]; v_test = v[80:]
a_train = a[:80]; a_test = a[80:]
k_train = k[:80]; k_test = k[80:]
F_train = F[:80]; F_test = F[80:]

# normalize x and k
a_m = np.mean(a); x_std = np.std(a)
k_m = np.mean(k); k_std = np.std(k)

train_len = 50

training_generator = TrainingGenerator(x_train, v_train, a_train, F_train, train_len=train_len, y_len=train_len)
testing_generator = TrainingGenerator(x_test, v_test, a_test, F_test, train_len=train_len, y_len=train_len)
#%% system constants
dt = tf.constant(t[1] - t[0])
m = tf.constant(1.0)
c = tf.constant(0.2)
#%% construct keras model
k_model = keras.models.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=[50,]),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(1, activation=None)
])
y_input = keras.Input([None, 50])
F_input = keras.Input([None, 1])
xdot_init = keras.Input([1,])
x_init = keras.Input([1,])

k = k_model(y_input)

spring_mass = SpringMassRNN(dt, 
                            m=m,
                            c=c,
                            k_call=k_model,
                            solver='rk4',
                            return_sequences=True,
                            stateful=False,
                            # input_shape=([None, 50],[None, 1])
)([y_input, F_input], initial_state=[xdot_init, x_init])

training_model = keras.model(
    inputs = [y_input, F_input, xdot_init, x_init],
    outputs = [spring_mass]
)
#%%
y_tester = tf.constant(np.zeros(50))
F_tester = tf.constant(np.zeros(1))

# xdot0, x0 = 

spring_mass_cell = SpringMass(dt, m, c, k_model, solver='rk4')

spring_mass_cell.call()

#%%
print("building LuGre model...")
alpha = 2
v_s=0.01
sigma_0_star = 24 # doesn't matter
units=50

F_cs_input = keras.Input(shape=[None, 2], name="F_cs_input")
# X_input = keras.Input(shape=[None, 2], name='X')
v_input = keras.Input(shape=[None, 1], name='v')
y_init_input = keras.Input(shape=[1,], name='y_init')


dense1 = keras.layers.Dense(units, use_bias=True, activation='relu')
dense2 = keras.layers.Dense(units, use_bias=True, activation='relu')
dense3 = keras.layers.Dense(1, use_bias=True, activation='relu')

dense1.build(input_shape=[1,2])
dense2.build(input_shape=[1,units])
dense3.build(input_shape=[1,units])

def sigma_call(inputs):
    y = dense1(inputs)
    y = dense2(y)
    y = dense3(y)
    return y

sigma_call_state_size = 0
sigma_model_weights = dense1.weights + dense2.weights + dense3.weights

concat = keras.layers.Concatenate()([v_input, F_cs_input])

spring_mass = SpringMassRNN(dt, 
                            m=m,
                            c=c,
                            k_call=k_model,
                            solver='rk4',
                            return_sequences=True,
                            stateful=False,
                            # input_shape=([None, 50],[None, 1])
)(concat, initial_state=[y_init_input])


model = keras.Model(
    inputs = [v_input, F_cs_input, y_init_input],
    outputs = [lugre]
)
#%% train model


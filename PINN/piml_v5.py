import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import keras.backend as K
import tqdm

from piml_classes import ModelInterior, ModelInitial, ModelBoundary, WaveLoss,\
    InitialLoss, BoundaryLoss, WeightSum

from callbacks import RestoreBestWeights

"""
convergence!!!
"""
#%% describe system
# physical constants
T = 1
L = 1
lam = 1
x_bounds = [0,1]
t_bounds = [0,1]
# g1 - initial position
TF_PI = tf.constant(np.pi)
def g1(x):
    return tf.math.sin(TF_PI*x)
# g2 - initial velocity
def g2(x):
    return tf.zeros(x.shape, dtype=x.dtype)

# training constants
rhof = 1
rho0 = 1
rhob = 1

def interior_loss(model, x_interior, t_interior):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            inputs = tf.concat([x_interior, t_interior], axis=1)
            w = model(inputs)
        jacobian = tape2.batch_jacobian(w, inputs)
        jacobian = tf.squeeze(jacobian)
    jacobian2 = tape1.batch_jacobian(jacobian, inputs)
    d2wdx2 = jacobian2[:,0,0]
    d2wdt2 = jacobian2[:,1,1]
    e_interior = tf.math.reduce_mean(tf.square(lam*d2wdx2 - d2wdt2), axis=0)
    return w, e_interior

def initial_loss(model, x_initial):
    with tf.GradientTape() as tape:
        t_initial = tf.zeros(x_initial.shape, dtype=x_initial.dtype)
        inputs = tf.concat([x_initial, t_initial], axis=1)
        w = model(inputs)
    jacobian = tape.batch_jacobian(w, inputs)
    dwdt = jacobian[:,0,:1]
    
    e_initial = tf.square(w - g1(x_initial)) + tf.square(dwdt - g2(x_initial))
    # e_initial = tf.square(w - g1(x_initial))
    
    e_initial = tf.math.reduce_mean(e_initial)
    return w, e_initial

def boundary_loss(model, t_boundary):
    x_boundary0 = tf.zeros(t_boundary.shape)
    x_boundary1 = tf.ones(t_boundary.shape)
    inputs0 = tf.concat([x_boundary0, t_boundary], axis=1)
    inputs1 = tf.concat([x_boundary1, t_boundary], axis=1)
    
    w0 = model(inputs0)
    w1 = model(inputs1)
    
    e_boundary = tf.square(w0) + tf.square(w1)
    e_boundary = tf.math.reduce_mean(e_boundary)
    
    return w0, w1, e_boundary

#%%
# define model
# x = keras.layers.Input(batch_input_shape=[64, 1], name='x_interior')
# t = keras.layers.Input(batch_input_shape=[64, 1], name='t_interior')
# concat = keras.layers.Concatenate()([x, t])
# nn model

initializer = keras.initializers.RandomUniform(
    minval=-0.05, maxval=0.05, seed=None
)

inputs = keras.layers.Input(2)
dense1 = keras.layers.Dense(100,
                            activation='tanh',
                            use_bias=True,
                            kernel_initializer=initializer,
                            trainable=True)(inputs)
dense2 = keras.layers.Dense(100,
                            activation='tanh',
                            use_bias=True,
                            kernel_initializer=initializer,
                            trainable=True)(dense1)
dense3 = keras.layers.Dense(100,
                            activation='tanh',
                            use_bias=True,
                            kernel_initializer=initializer,
                            trainable=True)(dense2)
dense4 = keras.layers.Dense(1,
                            activation=None,
                            use_bias=True,
                            kernel_initializer=initializer,
                            trainable=True)(dense3)
model = keras.Model(
    inputs=[inputs],
    outputs=[dense4]
)
# opt = keras.optimizers.SGD(learning_rate=.01)
opt = keras.optimizers.Adam(
    learning_rate=0.00001,
    beta_1=.95,
    beta_2=0.99999,
)
model.compile(optimizer=opt, loss='mse')

restore_best_weights = RestoreBestWeights(monitor='loss')
callbacks = [restore_best_weights]
callbacks = keras.callbacks.CallbackList(callbacks, add_history=True, model=model)

#%% custom training
num_batches = 10000
Nf = 2500 # number of interior points per batch
N0 = 50 # number of initial points per batch
Nb = 50 # number of boundary points per batch
loss_history = np.empty((num_batches, 4)); loss_history[:] = np.nan
printevery = 20

x0 = x_bounds[0]; x1 = x_bounds[1]; dx = x1 - x0
t0 = t_bounds[0]; t1 = t_bounds[1]; dt = t1 - t0

x0 = tf.Variable(x0, dtype=tf.float32)
x1 = tf.Variable(x1, dtype=tf.float32)
t0 = tf.Variable(t0, dtype=tf.float32)
t1 = tf.Variable(t1, dtype=tf.float32)
rhof = tf.Variable(rhof, dtype=tf.float32)
rho0 = tf.Variable(rho0, dtype=tf.float32)
rhob = tf.Variable(rhob, dtype=tf.float32)

logs = {}
callbacks.on_train_begin(logs=logs)

#tqdm fancy printout
batches = tqdm.trange(
    num_batches,
    desc='Batches',
    unit='batch',
    postfix = 'loss = {loss:.5f}, interior = {e_int:.5f}, initial = {e_init:.5f}, boundary = {e_bound;.5f}'
)
batches.set_postfix(loss=0, e_int=0, e_init=0, e_bound=0)

for batch in batches:
    callbacks.on_epoch_begin(batch, logs=logs)
    
    x_int = np.random.rand(Nf, 1)*dx + x0
    t_int = np.random.rand(Nf, 1)*dt + t0
    x_init = np.random.rand(N0, 1)*dx + x0
    # x_init = np.arange(0, 1, step=0.01).reshape(-1, 1)*dx + x0
    t_bound = np.random.rand(Nb, 1)*dt + t0
    
    x_int = tf.Variable(x_int, dtype=tf.float32)
    t_int = tf.Variable(t_int, dtype=tf.float32)
    x_init = tf.Variable(x_init, dtype=tf.float32)
    t_bound = tf.Variable(t_bound, dtype=tf.float32)
    with tf.GradientTape() as tape:
        # w_int, e_int = interior_loss(model, x_int, t_int)
        e_int = 0
        w_init, e_init = initial_loss(model, x_init)
        # w_bound0, w_bound1, e_bound = boundary_loss(model, t_bound)
        e_bound = 0
        loss = rhof*e_int + rho0*e_init + rhob*e_bound
    grads = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    # save losses
    loss = float(loss)
    e_int = float(e_int)
    e_init = float(e_init)
    e_bound = float(e_bound)
    
    loss_history[batch] = [e_int, e_init, e_bound, loss]
    # update logs
    logs['loss'] = loss
    logs['e_int'] = e_int
    logs['e_init'] = e_init
    logs['e_bound'] = e_bound
    # apply callbacks
    callbacks.on_epoch_end(batch, logs=logs)
    
    # update tqdm presentation
    batches.set_postfix(
        loss=loss,
        e_int=e_int,
        e_init=e_init,
        e_bound=e_bound
    )

callbacks.on_train_end(logs=logs)
#%%
# x_init = np.linspace(0, 1, num=100, endpoint=True).reshape(-1, 1)
# t_init = np.zeros(x_init.shape)
# coords = np.concatenate([x_init, t_init], axis=1)

# model.fit(coords, g1(x_init), epochs=400)
#%%
plt.figure()
plt.plot(loss_history[:,3], label='loss')
plt.plot(loss_history[:,0], label='interior loss')
plt.plot(loss_history[:,1], label='initial loss')
plt.plot(loss_history[:,2], label='boundary loss')
plt.legend(loc=1)
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.ylim((0, 1))
plt.tight_layout()
#%% plot the approximating model

# make a new model copying the weights from the original, may not be necessary
# model1 = keras.models.Sequential([
#     keras.layers.Dense(30, activation='sigmoid', use_bias=True, trainable=True, input_shape=[2]),
#     keras.layers.Dense(30, activation='sigmoid', use_bias=True, trainable=True),
#     keras.layers.Dense(1, activation='sigmoid', use_bias=True, trainable=True),
# ])
# for layer1, layer2 in zip(model1.layers, model.layers):
#     layer1.set_weights(layer2.get_weights())

# model = model1

x_axis = np.linspace(x0, x1, num=500)
t_axis = np.linspace(t0, t1, num=500)

x_mesh, t_mesh = np.meshgrid(x_axis, t_axis)

coords = np.vstack((x_mesh.reshape(-1), t_mesh.reshape(-1))).T

x = coords[:,:1]
t = coords[:,1:]

w_pred = model(coords).numpy()
w_pred = w_pred.reshape(500, 500)

fig = plt.figure(figsize=(7, 2))
pc = plt.pcolormesh(t_mesh, x_mesh, w_pred, vmin=-1, vmax=1)
cbar = fig.colorbar(pc)
plt.xlabel(r'$t$ (ul)')
plt.ylabel(r'$x$ (ul)')
plt.savefig("./figures/PINN spring.png", dpi=300)
#%% test initial and boundary losses

# initial condition
x_initial = tf.Variable(np.linspace(0, 1, num=100, endpoint=True).reshape(-1, 1)*dx + x0, dtype=tf.float32)
with tf.GradientTape() as tape:
    t_initial = tf.zeros(x_initial.shape, dtype=x_initial.dtype)
    inputs = tf.concat([x_initial, t_initial], axis=1)
    w = model(inputs)
jacobian = tape.batch_jacobian(w, inputs)
dwdt = jacobian[:,0,:1]

w_true = g1(x_initial)
dwdt_true = g2(x_initial)

plt.figure(figsize=(6,4))
plt.plot(x_initial, w, label='model initial value')
plt.plot(x_initial, w_true, label='initial value condition')
plt.legend(loc=1)
plt.xlim(x_bounds)
plt.xlabel('x (ul)')
plt.ylabel('w (ul)')
plt.tight_layout()
plt.savefig('./figures/initial val.png', dpi=300)

plt.figure(figsize=(6, 4))
plt.plot(x_initial, dwdt, label='model initial deriv.')
plt.plot(x_initial, dwdt_true, label='initial deriv. condition')
plt.legend(loc=1)
plt.xlim(x_bounds)
plt.xlabel('x (ul)')
plt.ylabel('y (ul)')
plt.tight_layout()
plt.savefig('./figures/initial deriv.png', dpi=300)

# boundary condition
t_boundary = tf.Variable(np.linspace(0, 1, num=100, endpoint=True).reshape(-1, 1)*dt + t0, dtype=tf.float32)
x_boundary0 = tf.zeros(t_boundary.shape)
x_boundary1 = tf.ones(t_boundary.shape)
inputs0 = tf.concat([x_boundary0, t_boundary], axis=1)
inputs1 = tf.concat([x_boundary1, t_boundary], axis=1)

w0 = model(inputs0)
w1 = model(inputs1)

plt.figure(figsize=(6,4))
plt.plot(t_boundary, w0, label='x=0')
plt.plot(t_boundary, w1, label='x=1')
plt.legend()
plt.xlabel('t (ul)')
plt.ylabel('w (ul)')
plt.xlim(t_bounds)
plt.tight_layout()
plt.savefig('./figures/boundary.png', dpi=300)
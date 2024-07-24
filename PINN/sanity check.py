import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import keras.backend as K
"""
sanity check
"""

initializer = keras.initializers.RandomUniform(
    minval=-2, maxval=2, seed=None
)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='sigmoid', kernel_initializer=initializer),
    keras.layers.Dense(30, activation='sigmoid', kernel_initializer=initializer),
    keras.layers.Dense(1, activation=None, kernel_initializer=initializer)
])


# inputs = keras.layers.Input(1)
# dense1 = keras.layers.Dense(10, activation='sigmoid', use_bias=True, trainable=True)(inputs)
# dense2 = keras.layers.Dense(10, activation='sigmoid', use_bias=True, trainable=True)(dense1)
# dense3 = keras.layers.Dense(1, activation=None, use_bias=True, trainable=True)(dense2)
# model = keras.Model(
#     inputs=[inputs],
#     outputs=[dense3]
# )


# opt = keras.optimizers.SGD(learning_rate=.001)

opt = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=.9,
    beta_2=0.99999,
)

model.compile(
    optimizer=opt,
    loss='mse'
)

def g1(x):
    return tf.math.sin(np.pi*x)
#%%
x = np.arange(0, 1, step=0.01).reshape(-1, 1)
y = g1(x)
# x = x*100

history = model.fit(
    x, y,
    epochs=2000,
    shuffle=True,
    verbose=0,
)
loss_history = history.history['loss']
#%%
y_pred = model.predict(x)

plt.figure()
plt.plot(x, y, label='true')
plt.plot(x, y_pred, label='predicted')
plt.legend()
plt.xlim((0, 1))
plt.tight_layout()
#%%
plt.figure()
plt.plot(loss_history)
plt.ylim((0, 1))
plt.tight_layout()
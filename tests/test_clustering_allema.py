import allema as alm
import numpy as np
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import optax
from flax.training import train_state
from flax import serialization
import pickle
import os
from tensorflow.keras.datasets import fashion_mnist
from jax import jit, grad
import jax

# clustering problem as described in:
# Sahin, Mehmet Fatih, et al. "An inexact augmented Lagrangian framework for nonconvex optimization with nonlinear constraints." Advances in Neural Information Processing Systems 32 (2019).

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)

MODEL_FILE = ["./tests/saved_models/fashion_mnist_model.pkl", "./saved_models/fashion_mnist_model.pkl"]

for model_file in MODEL_FILE:
    if os.path.exists(model_file):
        MODEL_FILE = model_file
        break

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, y_train, x_test, y_test

class FeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=50)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=10)(x)
        x = nn.sigmoid(x)
        return x

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 28*28)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def compute_loss_and_accuracy(params, model, x, y):
    logits = model.apply(params, x)
    y_one_hot = jax.nn.one_hot(y, num_classes=10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_one_hot))
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, accuracy

@jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        y_one_hot = jax.nn.one_hot(y, num_classes=10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_one_hot))
        return loss

    grads = grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

def train_or_load_model(x_train, y_train, x_test, y_test, num_epochs=1000, batch_size=64, learning_rate=1e-3):
    key = random.PRNGKey(0)
    model = FeatureExtractor()

    if os.path.exists(MODEL_FILE):
        print("Loading saved model...")
        with open(MODEL_FILE, "rb") as f:
            params = serialization.from_bytes(model.init(key, jnp.ones((1, 28*28))), pickle.load(f))
        return params

    print("Training new model...")
    state = create_train_state(key, model, learning_rate)
    x_train, y_train = jnp.array(x_train), jnp.array(y_train)
    x_test, y_test = jnp.array(x_test), jnp.array(y_test)

    num_batches = x_train.shape[0] // batch_size
    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]
            state = train_step(state, batch_x, batch_y)

        if (epoch) % 50 == 0:
            train_loss, train_acc = compute_loss_and_accuracy(state.params, model, x_train, y_train)
            test_loss, test_acc = compute_loss_and_accuracy(state.params, model, x_test, y_test)

            print(f"Epoch {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
            print(f"  Test  Loss: {test_loss:.4f}, Test  Accuracy: {test_acc*100:.2f}%\n")

    print("Saving trained model...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(serialization.to_bytes(state.params), f)

    return state.params

def compute_distance_matrix(Z):
    return jnp.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)

n, r, s = 100, 5, 3 #1000, 20, 10  # n: points, r: embedding dim, s: cluster count

x_train, y_train, x_test, y_test = load_fashion_mnist()
params = train_or_load_model(x_train, y_train, x_test, y_test)

model = FeatureExtractor()
Z = model.apply(params, x_test[:n])
D = compute_distance_matrix(Z)

def f(x):
    V = x.reshape(n, r)
    return jnp.trace(D @ (V @ V.T))

def h(x):
    V = x.reshape(n, r)
    return jnp.sum(V @ V.T, axis=1) - 1

def g(x):
    return jnp.hstack([
        jnp.linalg.norm(x) - jnp.sqrt(s),
        -x
    ])

# initialize within constraints
x0 = random.uniform(random.PRNGKey(42), (n * r,))
x0 = x0 * (jnp.sqrt(s) / jnp.linalg.norm(x0))

nvars = x0.shape[0]
n_eq_cons = h(x0).shape[0]
n_ineq_cons = g(x0).shape[0]

problem = alm.Problem(
    obj=f,
    eq_con=h,
    ineq_con=g
    )

lbda = random.uniform(random.PRNGKey(1234), (n_eq_cons,))
mu0 = random.uniform(random.PRNGKey(1234), (n_ineq_cons,))
rho = 1
nu = 1
tol = 1e-6
max_iter = 1000
lbfgs_options = {'maxls': 20, 'gtol': tol, 'eps': 1.e-8, 'ftol': tol, 'maxfun': max_iter, 'maxcor': 10}
# sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, beta=0.5, alpha=1.01, xi1=1.1, xi2=1.1, start_feas=True, inner_solver="L-BFGS-B", lbfgs_options=lbfgs_options)
sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=True, inner_solver="L-BFGS-B")


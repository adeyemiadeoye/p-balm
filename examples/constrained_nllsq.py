import jax.numpy as jnp
import pbalm
import numpy as np


# \min_{\theta} \quad & \sum_{i=1}^{m} (y_i - f(x_i; \theta))^2 \\
#    \text{s.t.} \quad & \theta_1 + \theta_2 \leq 1 \\
#                      & \theta \geq 0

# Configure JAX
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Generate synthetic data
rng = np.random.default_rng(123)
n_samples = 100
x_data = jnp.linspace(0, 1, n_samples)

# True model: y = theta1 * exp(-theta2 * x) + noise
theta_true = jnp.array([0.5, 2.0])
y_data = theta_true[0] * jnp.exp(-theta_true[1] * x_data) + 0.05 * rng.standard_normal(n_samples)

# Model prediction
def model(x, theta):
    return theta[0] * jnp.exp(-theta[1] * x)

# Objective: sum of squared residuals
def f1(theta):
    predictions = model(x_data, theta)
    residuals = y_data - predictions
    return jnp.sum(residuals**2)

# Constraints
def g1(theta):
    return theta[0] + theta[1] - 5.0  # theta1 + theta2 <= 5

def g2(theta):
    return -theta  # theta >= 0 (element-wise)

# Create problem
problem = pbalm.Problem(
    f1=f1,
    g=[g1, g2],
    jittable=True
)

# Initial guess
theta0 = jnp.array([0.3, 1.0])

# Solve
result = pbalm.solve(
    problem,
    theta0,
    tol=1e-6,
    max_iter=200
)

print(f"True parameters: {theta_true}")
print(f"Estimated parameters: {result.x}")
print(f"Final objective: {f1(result.x):.6f}")
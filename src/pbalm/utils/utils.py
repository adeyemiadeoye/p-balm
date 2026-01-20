import jax.numpy as jnp
import numpy as np
import pbalm

def params_flatten(params):
    return jnp.concatenate([p.flatten() if isinstance(p, jnp.ndarray) else jnp.array([p]) for p in params])
def params_shape(params):
    shapes = [p.shape if isinstance(p, jnp.ndarray) else () for p in params]
    cumsizes = np.cumsum([np.prod(s) for s in shapes])
    # cumsizes = jnp.array(cumsizes, dtype=int)
    return shapes, cumsizes
def params_unflatten(params_flattened, shapes, cumsizes):
    params_split = jnp.split(params_flattened, cumsizes)
    params_unflattened = [array.reshape(shape) for array, shape in zip(params_split, shapes)]
    return params_unflattened

def update_penalties(lbda_sizes, mu_sizes,
                        rho, nu, rho0, nu0,
                        E_x, prev_E, h_x, prev_h,
                        beta, xi1, xi2, phi_i):
    
    m_1 = 0
    m_n = 0
    nu_new = nu
    for mu_i in mu_sizes:
        m_n += mu_i
        nu_i = nu[m_1:m_n]
        nrm_Ei = jnp.linalg.norm(E_x[m_1:m_n], jnp.inf)
        prev_nrm_Ei = jnp.linalg.norm(prev_E[m_1:m_n], jnp.inf)
        if nrm_Ei > beta*prev_nrm_Ei:
            nu_i_new = jnp.maximum(xi2*nu_i, jnp.full_like(nu_i, nu0*phi_i))
            nu_new = nu_new.at[m_1:m_n].set(nu_i_new)
        m_1 = m_n
    
    l_1 = 0
    l_n = 0
    rho_new = rho
    for lbda_i in lbda_sizes:
        l_n += lbda_i
        hxi = h_x[l_1:l_n]
        rho_i = rho[l_1:l_n]
        nrm_hi = jnp.linalg.norm(hxi, jnp.inf)
        prev_nrm_hi = jnp.linalg.norm(prev_h[l_1:l_n], jnp.inf)
        if nrm_hi > beta*prev_nrm_hi:
            rho_i_new = jnp.maximum(xi1*rho_i, jnp.full_like(rho_i, rho0*phi_i))
            rho_new = rho_new.at[l_1:l_n].set(rho_i_new)
        l_1 = l_n

    return rho_new, nu_new

class GradEvalCounter:
    """
    Wrapper class to count the number of gradient evaluations.
    """
    def __init__(self, fn):
        self.fn = fn
        self.count = 0
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.fn(*args, **kwargs)
    def reset(self):
        self.count = 0
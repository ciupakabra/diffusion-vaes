import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp


def logdethess(y):
    zeros = jnp.zeros(y.shape)
    y_ = jnp.concatenate([zeros[..., None], y[..., None]], axis=-1)
    return jnp.sum(y - 2 * logsumexp(y_, axis=-1))
    # return jnp.sum(y - 2 * jnp.log(1 + jnp.exp(y)))
logdethess = jax.vmap(logdethess)

def constrain(V):
    def W(y):
        return V(jax.nn.sigmoid(y)) + logdethess(y)
    return W

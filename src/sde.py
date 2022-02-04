from typing import Generator, Mapping, Tuple, NamedTuple, Sequence, Optional, Callable

import jax
import jax.numpy as jnp
import haiku as hk

def euler_maruyama(
        drift_diff_func: Callable,
        x_0: jnp.ndarray,
        n_steps: int,
        rng,
        ) -> jnp.ndarray:
    """
    Euler-Maruyama for sampling. drift_diff_func should return a tuple with a
    drift, diffusion and an extra output to accumulate over integration.

    """

    dt = 1 / n_steps

    noise_terms = jax.random.normal(rng, (n_steps,) + x_0.shape)

    def em_step(x_i, val):
        noise_term, t = val

        drift, diff, extras = drift_diff_func(x_i, t)
        x_j = x_i + dt * drift + jnp.sqrt(dt) * diff * noise_term

        return x_j, extras

    ts = jnp.linspace(0, 1, n_steps + 1)
    x_1, extras = hk.scan(em_step, x_0, (noise_terms, ts[:-1]))

    return x_1, extras

def ula(
        x_0 : jnp.ndarray,
        grad_U : Callable,
        step_size : float,
        n_steps : int,
        rng,
        ) -> jnp.ndarray:

    noise_terms = jax.random.normal(rng, (n_steps,) + x_0.shape)

    def ula_step(x_i, noise_term):
        x_j = x_i - step_size * grad_U(x_i) + jnp.sqrt(2 * step_size) * noise_term
        return x_j, None

    x_last, _ = jax.lax.scan(ula_step, x_0, noise_terms)

    return x_last

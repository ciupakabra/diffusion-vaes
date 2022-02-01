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

    def em_step(carry, t):
        rng, x_t = carry
        rng, subrng = jax.random.split(rng)

        noise_term = jax.random.normal(subrng, x_t.shape)
        drift, diff, extras = drift_diff_func(x_t, t)

        x_t_ = x_t + dt * drift + jnp.sqrt(dt) * diff * noise_term

        return (rng, x_t_), extras

    ts = jnp.linspace(0, 1, n_steps + 1)
    carry_0 = (rng, x_0)

    (_, x_1), extras = hk.scan(em_step, carry_0, ts[:-1])

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

    x_last, _ = hk.scan(ula_step, x_0, noise_terms)

    return x_last

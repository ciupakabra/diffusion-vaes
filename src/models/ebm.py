from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from einops import reduce

from src import modules
from src import sde

class EBM(hk.Module):

    def __init__(self,
            drift : modules.Drift,
            potential : hk.Module,
            shape : Tuple[int],
            gamma : float = 1.0,
            ):
        super().__init__()

        self.drift = drift
        self.potential = potential
        self.shape = shape
        self.gamma = gamma

    def ula(self,
            n_steps : int,
            n_samples : int,
            step_size : float,
            y_0 : jnp.ndarray = None) -> jnp.ndarray:

        if y_0 is None:
            shape = (n_samples,) + self.shape
            y_0 = jax.random.normal(hk.next_rng_key(), shape)

        grad_U = jax.grad(lambda y : jnp.sum(self.potential(y, False)))

        y = sde.ula(
                y_0,
                grad_U,
                step_size,
                n_steps,
                hk.next_rng_key(),)

        return y

    def sample(self, 
            n_samples : int, 
            n_steps : int, 
            is_training : bool,
            apply_sigmoid : bool) -> jnp.ndarray:

        def drift_diff_func(y, t):
            drift = self.drift(y, t, is_training)
            diff = jnp.sqrt(self.gamma)
            return drift, diff, drift

        shape = (n_samples,) + self.shape
        y_0 = jnp.zeros(shape)
        y_1, us = sde.euler_maruyama(
                drift_diff_func,
                y_0,
                n_steps,
                hk.next_rng_key())

        if apply_sigmoid:
            y_1 = jax.nn.sigmoid(y_1)
        return y_1, us

    def loss(self,
            y_pos : jnp.ndarray,
            y_neg : jnp.ndarray,
            is_training : bool) -> jnp.ndarray:

        E_pos = self.potential(y_pos, is_training)
        E_neg = self.potential(y_neg, is_training)

        return (jnp.mean(E_pos) - jnp.mean(E_neg))

    def relative_entropy_control_cost(self,
            n_steps : int,
            n_samples : int,
            is_training : bool) -> jnp.ndarray:

        y_1, us = self.sample( 
            is_training=is_training,
            n_steps=n_steps,
            n_samples=n_samples,
            apply_sigmoid=True,
        )

        dt = 1 / n_steps
        energy_cost = reduce(us**2, "t b w h c -> b", "sum") * dt / (2 * self.gamma)

        # P(x) = exp(-E(x)) / Z
        log_p = - jnp.squeeze(self.potential(y_1, False))
        terminal_cost = - reduce(y_1**2, "b w h c -> b", "sum") / (2 * self.gamma) - log_p

        return jnp.mean(energy_cost + terminal_cost)

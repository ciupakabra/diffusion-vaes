import jax
import pdb
import jax.numpy as jnp

from src import sde

import haiku as hk

class Follmer(hk.Module):

    def __init__(self, drift, gamma, data_shape):
        super().__init__()

        self.drift = drift
        self.gamma = gamma
        self.data_shape = data_shape

    def sample(self, batch_size, n_steps):
        
        def drift_diff_func(x, t):
            return self.drift(x, t, False), jnp.sqrt(self.gamma), None

        x_0 = jnp.zeros((batch_size,) + self.data_shape)
        x_1, _ = sde.euler_maruyama(
                drift_diff_func,
                x_0,
                n_steps,
                hk.next_rng_key())

        return x_1

    def relative_entropy_control_cost(self, batch_size, n_steps, log_p):
        def drift_diff_func(x, t):
            u = self.drift(x, t, True)
            return u, jnp.sqrt(self.gamma), u

        x_0 = jnp.zeros((batch_size,) + self.data_shape)
        
        x_1, us = sde.euler_maruyama(
                drift_diff_func,
                x_0,
                n_steps,
                hk.next_rng_key())

        dt = 1 / n_steps

        dims_e = (0,) + tuple(range(2, 2 + len(self.data_shape)))
        dims_t = tuple(range(1, 1 + len(self.data_shape)))
        # dims_e = (0, 2, 3, 4)
        # dims_t = (1, 2, 3)

        energy_cost = jnp.sum(us**2, axis=dims_e) * dt / (2 * self.gamma)
        terminal_cost = - jnp.sum(x_1**2, axis=dims_t) / (2 * self.gamma) - log_p(x_1)

        return jnp.mean(energy_cost + terminal_cost)

class ModulatedFollmer(hk.Module):

    def __init__(self, drift, grad_mult, gamma, data_shape):
        super().__init__()

        self.drift = drift
        self.grad_mult = grad_mult
        self.gamma = gamma
        self.data_shape = data_shape

    def sample(self, batch_size, n_steps, log_p):

        grad = jax.grad(lambda x : log_p(x).sum())
        
        def drift_diff_func(x, t):
            m = self.grad_mult(jnp.array([t]))
            u = self.drift(x, t, False) + m * grad(x)
            return u, jnp.sqrt(self.gamma), None

        x_0 = jnp.zeros((batch_size,) + self.data_shape)
        x_1, _ = sde.euler_maruyama(
                drift_diff_func,
                x_0,
                n_steps,
                hk.next_rng_key())

        return x_1

    def relative_entropy_control_cost(self, batch_size, n_steps, log_p):
        grad = jax.grad(lambda x : log_p(x).sum())

        def drift_diff_func(x, t):
            m = self.grad_mult(jnp.array([t]))
            u = self.drift(x, t, True) + m * grad(x)
            return u, jnp.sqrt(self.gamma), u

        x_0 = jnp.zeros((batch_size,) + self.data_shape)
        
        x_1, us = sde.euler_maruyama(
                drift_diff_func,
                x_0,
                n_steps,
                hk.next_rng_key())

        dt = 1 / n_steps

        dims_e = (0,) + tuple(range(2, 2 + len(self.data_shape)))
        dims_t = tuple(range(1, 1 + len(self.data_shape)))
        # dims_e = (0, 2, 3, 4)
        # dims_t = (1, 2, 3)

        energy_cost = jnp.sum(us**2, axis=dims_e) * dt / (2 * self.gamma)
        terminal_cost = - jnp.sum(x_1**2, axis=dims_t) / (2 * self.gamma) - log_p(x_1)

        return jnp.mean(energy_cost + terminal_cost)

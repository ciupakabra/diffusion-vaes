from typing import Generator, Mapping, Tuple, NamedTuple, Sequence, Optional

import haiku as hk
import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from modules import *

BERNOULLI = "bernoulli"
GAUSSIAN = "gaussian"
LIKELIHOODS = [BERNOULLI, GAUSSIAN]

def log_likelihood(logits: jnp.ndarray, y: jnp.ndarray, distribution: str = BERNOULLI) -> jnp.array:
    if distribution == BERNOULLI:
        dist = tfd.Independent(tfd.Bernoulli(logits=logits))
    elif distribution == GAUSSIAN:
        dist = tfd.MultivariateNormalDiag(jax.nn.sigmoid(logits), jnp.ones(logits.shape))
        dist = tfd.Independent(dist)
    return dist.log_prob(y)

class DiffusionVAE(hk.Module):
    """ Diffusion model """

    def __init__(self, 
            latent_size: int,
            control_drift: AmortizedDrift,
            model_drift: Drift,
            decoder: hk.Module,
            gamma: float = 1.0,
            likelihood: str = BERNOULLI,
            ):
        super().__init__()

        self.likelihood = likelihood
        self.latent_size = latent_size
        self.gamma = gamma

        self.control_drift = control_drift
        self.model_drift = model_drift
        self.decoder = decoder

    def em(self, 
            is_training : bool,
            n_steps: int = 20, 
            y: jnp.ndarray = None, 
            batch_size: int = None):
        """ 
        Euler-Maruyama for the latent sampling. If data y is passed,
        then returns a sample from the variational approximation, i.e. controls
        the diffusion with the amortized drift.

        Returns the final sample and the drifts evaluated along the way.
        """

        if y is not None:
            batch_size = y.shape[0]

        dt = 1 / n_steps

        def em_step(carry, t):
            rng, x_t = carry
            rng, subrng = jax.random.split(rng)

            noise_term = jax.random.normal(subrng, x_t.shape)

            model_drift = self.model_drift(x_t, t, is_training)
            x_t_ = x_t + dt * model_drift + jnp.sqrt(dt * self.gamma) * noise_term

            new_val = model_drift

            if y is not None:
                control_drift = self.control_drift(y, x_t, t, is_training)
                x_t_ = x_t_ + dt * control_drift
                new_val = (control_drift, model_drift)

            new_carry = (rng, x_t_)

            return new_carry, new_val

        x_0 = jnp.zeros((batch_size, self.latent_size))
        ts = jnp.linspace(0, 1, n_steps + 1)
        carry_0 = (hk.next_rng_key(), x_0)

        (_, x_1), drifts = hk.scan(em_step, carry_0, ts[:-1])

        return x_1, drifts

    def sample(self, batch_size: int, n_steps: int):
        x_1, _ = self.em(False, batch_size=batch_size, n_steps=n_steps)
        return jax.nn.sigmoid(self.decoder(x_1, False))

    def relative_entropy_control_cost(self, 
            y: jnp.ndarray, 
            n_steps: int,
            is_training: bool
            ) -> jnp.ndarray:
        x_1, (us, _) = self.em(is_training, y=y, n_steps=n_steps)
        logits = self.decoder(x_1, is_training)

        dt = 1 / n_steps
        energy_cost = jnp.sum(us**2, axis=(0, 2)) * dt / (2 * self.gamma)
        terminal_cost = - log_likelihood(logits, y, distribution=self.likelihood)

        return jnp.mean(energy_cost + terminal_cost)

@jax.vmap
def kl(mu, logvar):
  return -0.5 * jnp.sum(1. + logvar - mu**2. - jnp.exp(logvar))

class VanillaVAE(hk.Module):
    """ Vanilla VAE model """

    def __init__(self, 
            latent_size: int,
            encoder: hk.Module,
            decoder: hk.Module,
            likelihood: str = BERNOULLI,
            ):
        super().__init__()

        self.likelihood = likelihood
        self.latent_size = latent_size
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, y : jnp.array, is_training : bool) -> jnp.array:
        enc = self.encoder(y, is_training)
        mu, logvar = jnp.split(enc, 2, axis=1)
        return mu, logvar

    def decode(self, x : jnp.array, is_training : bool) -> jnp.array:
        return self.decoder(x, is_training)

    def sample(self, n: int, is_training : bool) -> jnp.array:
        x = jax.random.normal(hk.next_rng_key(), (n, self.latent_size))
        logits = self.decode(x, is_training)
        return jax.nn.sigmoid(logits)

    def elbo(self, y : jnp.array, is_training : bool) -> jnp.array:
        mu, logvar = self.encode(y, is_training)
        x = mu + jnp.exp(0.5 * logvar) * jax.random.normal(hk.next_rng_key(), mu.shape)
        logits = self.decode(x, is_training)

        kl_loss = kl(mu, logvar)
        recon_loss = log_likelihood(logits, y, distribution=self.likelihood)

        return jnp.mean(recon_loss - kl_loss)

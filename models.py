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
        dist = tfd.Independent(tfd.Bernoulli(logits=logits, validate_args=True))
    elif distribution == GAUSSIAN:
        dist = tfd.MultivariateNormalDiag(jax.nn.sigmoid(logits), jnp.ones(logits.shape), validate_args=True)
    return dist.log_prob(y)

class DiffusionVAE(hk.Module):
    """ Diffusion model """

    def __init__(self, 
            gamma: float = 1.0,
            hidden_size: int = 512,
            latent_size: int = 10,
            decoder: Optional[hk.Module] = None,
            likelihood: str = BERNOULLI):
        super().__init__()

        self.likelihood = likelihood
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # self.control_drift = AmortizedDrift(hidden_size=hidden_size, latent_size=latent_size)
        self.control_drift = CNNAmortizedDrift(hidden_size=hidden_size, latent_size=latent_size)
        self.model_drift = Drift(hidden_size=hidden_size, latent_size=latent_size)
        # self.decoder = Decoder(hidden_size=hidden_size, latent_size=latent_size)
        self.decoder = CNNDecoder()
        self.gamma = gamma

    def em(self, n_steps: int = 20, y: jnp.ndarray = None, batch_size: int = None):
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

            model_drift = self.model_drift(x_t, t)
            x_t_ = x_t + dt * model_drift + jnp.sqrt(dt * self.gamma) * noise_term

            new_val = model_drift

            if y is not None:
                control_drift = self.control_drift(y, x_t, t)
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
        x_1, _ = self.em(batch_size=batch_size, n_steps=n_steps)
        return jax.nn.sigmoid(self.decoder(x_1))

    def relative_entropy_control_cost(self, y: jnp.ndarray, n_steps: int):
        x_1, (us, _) = self.em(y=y, n_steps=n_steps)
        logits = self.decoder(x_1)

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
            latent_size: int = 10,
            likelihood: str = BERNOULLI):
        super().__init__()

        self.likelihood = likelihood

        self.latent_size = latent_size
        self.encoder = CNNEncoder(2 * latent_size)
        self.decoder = CNNDecoder()

    def encode(self, y : jnp.array) -> jnp.array:
        enc = self.encoder(y)
        mu, logvar = jnp.split(enc, 2, axis=1)
        return mu, logvar

    def decode(self, x : jnp.array) -> jnp.array:
        return self.decoder(x)

    def sample(self, n: int) -> jnp.array:
        logits = self.decode(jax.random.normal(hk.next_rng_key(), (n, self.latent_size)))
        return jax.nn.sigmoid(logits)

    def elbo(self, y : jnp.array) -> jnp.array:
        mu, logvar = self.encode(y)
        x = mu + jnp.exp(0.5 * logvar) * jax.random.normal(hk.next_rng_key(), mu.shape)
        logits = self.decode(x)

        kl_loss = kl(mu, logvar)
        recon_loss = log_likelihood(logits, y, distribution=self.likelihood)

        return jnp.mean(recon_loss - kl_loss)

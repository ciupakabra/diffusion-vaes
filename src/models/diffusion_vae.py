import haiku as hk
import jax
import jax.numpy as jnp

from src import utils
from src import modules
from src import sde

class DiffusionVAE(hk.Module):
    """ Diffusion model """

    def __init__(self, 
            latent_size: int,
            control_drift: modules.AmortizedDrift,
            model_drift: modules.Drift,
            decoder: hk.Module,
            gamma: float = 1.0,
            likelihood: str = utils.BERNOULLI,
            ):
        super().__init__()

        self.likelihood = likelihood
        self.latent_size = latent_size
        self.gamma = gamma

        self.control_drift = control_drift
        self.model_drift = model_drift
        self.decoder = decoder

    def decode(self, x : jnp.ndarray, n_steps : int, is_training : bool) -> jnp.ndarray:
        # TODO: this is digusting
        if isinstance(self.decoder, modules.CELEBA_ODE_Decoder) or \
                isinstance(self.decoder, modules.MNIST_ODE_Decoder):
            return self.decoder(x, n_steps, is_training)
        else:
            return self.decoder(x, is_training)

    def sample(self,
            is_training : bool = False,
            n_steps : int = 20,
            y : jnp.ndarray = None,
            batch_size : int = None,
            apply_sigmoid : bool = False) -> jnp.ndarray:
        """

        If data y is passed, then returns a sample from the variational approximation, 
        i.e. controls the diffusion with the amortized drift. Otherwise, returns
        a sample from the model.

        Returns the sampled logits and the control drift evaluated along the way
        (if the diffusion was controlled).

        """

        def drift_diff_func(x_t, t):
            diff = jnp.sqrt(self.gamma)

            drift = self.model_drift(x_t, t, is_training)
            extras = None
            
            if y is not None:
                extras = self.control_drift(y, x_t, t, is_training)
                drift += extras

            return drift, diff, extras

        if y is not None:
            batch_size = y.shape[0]
        x_0 = jnp.zeros((batch_size, self.latent_size))

        x_1, extras = sde.euler_maruyama(
                drift_diff_func,
                x_0,
                n_steps,
                hk.next_rng_key(),)

        logits = self.decode(x_1, n_steps, is_training)

        if apply_sigmoid:
            return jax.nn.sigmoid(logits), extras
        return logits, extras

    def relative_entropy_control_cost(self, 
            y: jnp.ndarray, 
            n_steps: int,
            is_training: bool
            ) -> jnp.ndarray:

        logits, us = self.sample( 
                is_training=is_training,
                n_steps=n_steps,
                y=y,
                apply_sigmoid=False)

        dt = 1 / n_steps
        energy_cost = jnp.sum(us**2, axis=(0, 2)) * dt / (2 * self.gamma)
        terminal_cost = - utils.log_likelihood(logits, y, distribution=self.likelihood)

        return jnp.mean(energy_cost + terminal_cost)

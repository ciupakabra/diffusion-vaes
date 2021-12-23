from typing import Generator, Mapping, Tuple, NamedTuple, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

class Decoder(hk.Module):
    """ Decoder module """

    def __init__(self, hidden_size: int = 512, latent_size: int = 10):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.mlp = hk.nets.MLP([self.latent_size, self.hidden_size, self.hidden_size, 28 * 28], activate_final=False)

    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)

class CNNDecoder(hk.Module):
    """ Decoder module with CNN architecture """

    def __init__(self):
        super().__init__()

        self.cnn = hk.Sequential([
            hk.Linear(7 * 7 * 32), jax.nn.relu,
            hk.Reshape((7, 7, 32)),
            hk.Conv2DTranspose(64, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Conv2DTranspose(32, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Conv2DTranspose(1, 3, stride=1, padding="SAME"),
            hk.Flatten()
        ])

    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        return self.cnn(x)

class CNNEncoder(hk.Module):
    """ Encoder for images (used to condition on an image) """

    def __init__(self, latent_size):
        super().__init__()

        self.cnn = hk.Sequential([
            hk.Reshape((28, 28, 1)),
            hk.Conv2D(32, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Conv2D(64, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Flatten(),
            hk.Linear(latent_size),
        ])

    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        return self.cnn(x)

class Drift(hk.Module):
    """ Drift module """

    def __init__(self, hidden_size: int = 512, latent_size: int = 10):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.mlp = hk.nets.MLP([self.hidden_size, self.hidden_size, self.hidden_size, self.latent_size], activate_final=False)

    def __call__(self, x : jnp.ndarray, t : float) -> jnp.ndarray:
        t = t * jnp.ones(x.shape[0])[:, None]
        inp = jnp.hstack((x, t))
        return self.mlp(inp)

class AmortizedDrift(Drift):
    """ Drift that also depends on data and can be used as amortized VI """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y : jnp.ndarray, x : jnp.ndarray, t : float) -> jnp.ndarray:
        return super().__call__(jnp.hstack((y, x)), t)

class CNNAmortizedDrift(Drift):
    """ Drift that also depends on data and can be used as amortized VI and uses
    CNN for encoding the image.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = CNNEncoder()

    def __call__(self, y : jnp.ndarray, x : jnp.ndarray, t : float) -> jnp.ndarray:
        enc = self.encoder(y)
        return super().__call__(jnp.hstack((enc, x)), t)

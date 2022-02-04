from typing import Generator, Mapping, Tuple, NamedTuple, Sequence, Union, Optional

import haiku as hk
import jax
import jax.numpy as jnp

import math

class Drift(hk.Module):
    """ Drift module """

    def __init__(self, 
            encoder: hk.Module,
            name: str = None,
            ):
        super().__init__(name=name)

        self.encoder = encoder

    def __call__(self, 
            x : jnp.ndarray, 
            t : float, 
            is_training : bool,
            ) -> jnp.ndarray:
        t = t * jnp.ones(x.shape[:-1])[..., None]
        inp = jnp.concatenate((x, t), axis=-1)
        return self.encoder(inp)

class AmortizedDrift(Drift):
    """ Drift that also depends on data and can be used as amortized VI """

    def __init__(self, 
            encoder: hk.Module, 
            data_encoder: hk.Module,
            name: str = None,
            ):
        super().__init__(encoder, name=name)

        self.data_encoder = data_encoder

    def __call__(self, 
            y : jnp.ndarray, 
            x : jnp.ndarray, 
            t : float, 
            is_training : bool,
            ) -> jnp.ndarray:
        y_enc = self.data_encoder(y, is_training)
        return super().__call__(jnp.hstack((y_enc, x)), t, is_training)

class Encoder(hk.Module):
    """ Simple Encoder module """

    def __init__(self, 
            enc_size: int,
            hidden_sizes: Sequence[int],
            name : str = None,
            ):
        super().__init__(name=name)

        hidden_sizes = hidden_sizes + [enc_size]

        self.net = hk.Sequential([
            hk.Reshape((-1,)),
            hk.nets.MLP(hidden_sizes, activate_final=False),
        ])

    def __call__(self, y : jnp.ndarray, is_training : bool) -> jnp.ndarray:
        return self.net(y)

class Decoder(hk.Module):
    """ Simple Decoder module """

    def __init__(self, 
            output_shape: Tuple[int],
            hidden_sizes: Sequence[int],
            ):
        super().__init__()

        hidden_sizes = hidden_sizes + [math.prod(output_shape)]

        self.net = hk.Sequential([
            hk.MLP(hidden_sizes, activate_final=False),
            hk.Reshape(output_shape),
        ])

    def __call__(self, x : jnp.ndarray, is_training : bool) -> jnp.ndarray:
        return self.net(x)

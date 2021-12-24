from typing import Generator, Mapping, Tuple, NamedTuple, Sequence, Union, Optional

import haiku as hk
import jax
import jax.numpy as jnp

import math

class Drift(hk.Module):
    """ Drift module """

    def __init__(self, 
            encoder: hk.Module,
            ):
        super().__init__()

        self.encoder = encoder

    def __call__(self, 
            x : jnp.ndarray, 
            t : float, 
            is_training : bool,
            ) -> jnp.ndarray:
        t = t * jnp.ones(x.shape[0])[:, None]
        inp = jnp.hstack((x, t))
        return self.encoder(inp, is_training)

class AmortizedDrift(Drift):
    """ Drift that also depends on data and can be used as amortized VI """

    def __init__(self, 
            encoder: hk.Module, 
            data_encoder: hk.Module,
            ):
        super().__init__(encoder)

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
            ):
        super().__init__()

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

class MNIST_CNN_Encoder(hk.Module):
    """ CNN Encoder for MNIST (used for VanillaVAE and for amortized drifts) """

    def __init__(self, enc_size: int):
        super().__init__()

        # I think this is also TF tutorial
        self.cnn = hk.Sequential([
            hk.Conv2D(32, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Conv2D(64, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Flatten(),
            hk.Linear(enc_size),
        ])

    def __call__(self, y : jnp.ndarray, is_training : bool) -> jnp.ndarray:
        return self.cnn(y)

class MNIST_CNN_Decoder(hk.Module):
    """ CNN Decoder module for MNIST """

    def __init__(self):
        super().__init__()

        # Taken from TF tutorial on VAEs
        self.cnn = hk.Sequential([
            hk.Linear(7 * 7 * 32), jax.nn.relu,
            hk.Reshape((7, 7, 32)),
            hk.Conv2DTranspose(64, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Conv2DTranspose(32, 3, stride=2, padding="SAME"), jax.nn.relu,
            hk.Conv2DTranspose(1, 3, stride=1, padding="SAME"),
        ])

    def __call__(self, x : jnp.ndarray, is_training : bool) -> jnp.ndarray:
        return self.cnn(x)

# Took these from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
# This wouldn't be so complicated if Sequential allowed passing is_training

class BlockSequential(hk.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def __call__(self, x : jnp.ndarray, is_training : bool = False) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, is_training=is_training)
        return x

class Block(hk.Module):
    def __init__(self,
            h_dim: int,
            kernel: int,
            stride: int,
            padding: Union[str, Tuple[int, int]],
            conv_transpose: bool = False,
            final_convolution: bool = False,
            ):
        super().__init__()

        if not conv_transpose:
            self.conv = hk.Conv2D(h_dim, kernel, stride=stride, padding=padding)
        else:
            self.conv = hk.Conv2DTranspose(h_dim, kernel, stride=stride, padding=padding)

        self.bnorm = hk.BatchNorm(False, False, 0.1)

        self.final_convolution = final_convolution
        if final_convolution:
            self.final_conv = hk.Conv2D(3, 3, padding=(1, 1))

    def __call__(self, y : jnp.ndarray, is_training: bool = False) -> jnp.ndarray:
        y = self.conv(y)
        y = self.bnorm(y, is_training)
        y = jax.nn.leaky_relu(y)

        if self.final_convolution:
            y = self.final_conv(y)
        return y

class CELEBA_CNN_Encoder(hk.Module):
    """ CNN Encoder for MNIST (used for VanillaVAE and for amortized drifts) """

    def __init__(self, enc_size: int):
        super().__init__()

        blocks = []
        hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            blocks.append(Block(h_dim, 3, 2, (1, 1)))

        self.cnn = BlockSequential(blocks)
        self.final_layer = hk.Sequential([hk.Flatten(), hk.Linear(enc_size)])


    def __call__(self, y : jnp.ndarray, is_training : bool) -> jnp.ndarray:
        y = self.cnn(y, is_training=is_training)
        y = self.final_layer(y)
        return y

class CELEBA_CNN_Decoder(hk.Module):
    """ CNN Decoder module for MNIST """

    def __init__(self):
        super().__init__()

        hidden_dims = [32, 64, 128, 256, 512]
        output_shapes = [4, 8, 16, 32, 64]
        hidden_dims.reverse()

        self.initial_layer = hk.Sequential([hk.Linear(hidden_dims[0] * 4), hk.Reshape((2, 2, 512))])

        # (h - 1) * stride - 2 * pad + dil * (ker - 1) + out_pad + 1

        # (h - 1) * 2 - 2 * 1 + 2 + 1 = 2h - 1


        blocks = []

        for i in range(len(hidden_dims) - 1):
            blocks.append(Block(hidden_dims[i + 1], 3, 2, [(1, 2), (1, 2)], conv_transpose=True))
        blocks.append(Block(hidden_dims[-1], 3, 2, [(1, 2), (1, 2)], conv_transpose=True, final_convolution=True))

        self.cnn = BlockSequential(blocks)


    def __call__(self, x : jnp.ndarray, is_training : bool) -> jnp.ndarray:
        y = self.initial_layer(x)
        y = self.cnn(y, is_training=is_training)
        return y

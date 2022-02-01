from typing import Union, Tuple

import haiku as hk
import jax.numpy as jnp
import jax

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

class CELEBA_ODE_Decoder(hk.Module):
    """ ODE Decoder module for CELEBA that uses Resnets for the drift """

    def __init__(self):
        super().__init__()

        self.cnn = CELEBA_CNN_Decoder()
        self.drift = ResBlock(channels=4, stride=1)

    def discretize(self, 
            y_0: jnp.ndarray,
            n_steps: int, 
            is_training : bool,
            ):

        dt = 1 / n_steps
        ts = jnp.linspace(0, 1, n_steps + 1)

        def step(i, y_i):
            drift = self.drift(y_i, is_training)
            y_ip1 = y_i + dt * drift
            return y_ip1

        y_1 = hk.fori_loop(0, n_steps, step, y_0)
        return y_1

    def __call__(self, x : jnp.ndarray, n_steps : int, is_training : bool) -> jnp.ndarray:
        y_0 = self.cnn(x, is_training)
        # For some reason lets keep this commented
        # y_0 = jax.nn.relu(y_0)
        y_1 = self.discretize(y_0, n_steps, is_training)
        return y_1

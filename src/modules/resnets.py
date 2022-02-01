from typing import Union, Sequence, Optional

import haiku as hk
import jax
import jax.numpy as jnp

class ResBlock(hk.Module):

    def __init__(self, 
            channels: int, 
            stride: Union[int, Sequence[int]], 
            name: Optional[str] = None,
            ):
        super().__init__(name=name)

        conv_0 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=3,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="conv_0")

        bn_0 = hk.BatchNorm(False, False, 0.1)

        conv_1 = hk.Conv2D(
            output_channels=channels - 1,
            kernel_shape=1,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv_1")

        bn_1 = hk.BatchNorm(False, False, 0.1)
        layers = ((conv_0, bn_0), (conv_1, bn_1))

        self.layers = layers

    def __call__(self, inputs, is_training):
        x = inputs

        for i, (conv_i, bn_i) in enumerate(self.layers):
          x = bn_i(x, is_training)
          x = jax.nn.relu(x)
          x = conv_i(x)

        return x

import haiku as hk
import jax
import jax.numpy as jnp

from .resnets import ResBlock

class MNIST_CNN_Encoder(hk.Module):
    """ CNN Encoder for MNIST (used for VanillaVAE and for amortized drifts) """

    def __init__(self, enc_size: int):
        super().__init__()

        # I think this is also TF tutorial
        # self.cnn = hk.Sequential([
        #     hk.Conv2D(32, 3, stride=2, padding="SAME"), jax.nn.swish,
        #     hk.Conv2D(64, 3, stride=2, padding="SAME"), jax.nn.swish,
        #     hk.Flatten(),
        #     hk.Linear(enc_size),
        # ])

        self.cnn = hk.Sequential([
            hk.Conv2D(16, 5, stride=2, padding=(4, 4)), jax.nn.swish,
            hk.Conv2D(32, 3, stride=2, padding=(1, 1)), jax.nn.swish,
            hk.Conv2D(64, 3, stride=2, padding=(1, 1)), jax.nn.swish,
            hk.Conv2D(64, 3, stride=2, padding=(1, 1)), jax.nn.swish,
            hk.Flatten(),
            hk.Linear(64), jax.nn.swish,
            hk.Linear(1),
        ])
        
    def __call__(self, y : jnp.ndarray, is_training) -> jnp.ndarray:
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

class MNIST_ODE_Decoder(hk.Module):
    """ ODE Decoder module for MNIST that uses Resnets for the drift """

    def __init__(self):
        super().__init__()

        self.cnn = MNIST_CNN_Decoder()
        # self.drift = Drift(encoder=ResBlock(channels=2, stride=1))
        self.drift = ResBlock(channels=2, stride=1)

    def discretize(self, 
            y_0: jnp.ndarray,
            n_steps: int, 
            is_training : bool,
            ):

        dt = 1 / n_steps
        ts = jnp.linspace(0, 1, n_steps + 1)

        def step(i, y_i):
            # drift = self.drift(y_i, ts[i], is_training)
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

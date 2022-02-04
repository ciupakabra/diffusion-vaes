import jax
import jax.numpy as jnp
import haiku as hk
import optax

from jax import flatten_util

from absl import flags
from absl import app

from src import modules
from src import models
from src import utils
from src import sde
from src import constraints

import pickle

from tqdm import tqdm
import matplotlib.pyplot as plt

flags.DEFINE_float("sigma_n", 0.1, "Likelihood variance")
flags.DEFINE_float("sigma_p", 0.1, "Prior variance")
flags.DEFINE_integer("N_train", 100, "Number of training points")
flags.DEFINE_integer("N_test", 100, "Number of test points")
flags.DEFINE_float("gamma", 0.05**2, "gamma for the follmer SDE")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for the optimizer")
flags.DEFINE_integer("training_iterations", 1000, "Number of iterations to train for")
flags.DEFINE_integer("batch_size_train", 32, "Batch size to use while training")
flags.DEFINE_integer("batch_size_test", 100, "Batch size to use while testing")
flags.DEFINE_integer("n_train_steps", 20, "Number of steps to use in EM while training")
flags.DEFINE_integer("n_test_steps", 100, "Number of steps to use in EM while testing")
flags.DEFINE_integer("seed", 42, "Random seed")

FLAGS = flags.FLAGS

with open("outputs/ebm-replay-buffer/params.npy", "rb") as f:
    params_pot = pickle.load(f)
with open("outputs/ebm-replay-buffer/state.npy", "rb") as f:
    state_pot = pickle.load(f)

def log_p(xs):
    return - modules.MNIST_CNN_Encoder(1)(xs, False)

log_p_fn = hk.without_apply_rng(hk.transform_with_state(log_p))
log_p = constraints.constrain(lambda x : jnp.squeeze(log_p_fn.apply(params_pot, state_pot, x)[0]))
grad_log_p = jax.grad(lambda y : log_p(y).sum())

class Drift(hk.Module):
    def __init__(self):
        super().__init__()
        self.drift = modules.Drift(hk.nets.MLP([300, 300, 300, 784]))
        self.reshape_1 = hk.Reshape((-1,))
        self.reshape_2 = hk.Reshape((28, 28, 1))
        
    def __call__(self, x, t, is_training):
        out = self.reshape_1(x)
        out = self.drift(out, t, is_training)
        out = self.reshape_2(out)
        return out

def build_follmer():
    drift = Drift()
    # follmer = models.Follmer(drift, FLAGS.gamma, (28, 28, 1))
    grad_mult = hk.nets.MLP([512, 512, 1])
    follmer = models.ModulatedFollmer(drift, grad_mult, FLAGS.gamma, (28, 28, 1))
    return follmer

def loss():
    follmer = build_follmer()
    return follmer.relative_entropy_control_cost(
            FLAGS.batch_size_train,
            FLAGS.n_train_steps,
            log_p,
            )

def sample():
    follmer = build_follmer()
    return follmer.sample(
            FLAGS.batch_size_test,
            FLAGS.n_test_steps,
            log_p,
            )

def main(_):
    rng_seq = hk.PRNGSequence(FLAGS.seed)

    # y_0 = jax.random.normal(next(rng_seq), (100, 28, 28, 1))
    # y_1 = sde.ula(
    #         y_0,
    #         lambda y : - grad_log_p(y),
    #         0.1,
    #         1000,
    #         next(rng_seq))

    # x_1 = jax.nn.sigmoid(y_1)

    # print("Average log_p ULA:", jnp.mean(log_p(x_1)))


    # xs = utils.image_grid(10, 10, x_1)
    # print(jnp.max(jnp.abs(xs)))
    # print(jnp.mean(jnp.abs(xs)))

    # xs = jnp.squeeze(jax.lax.clamp(0.0, xs, 1.0))
    # plt.imsave("ula_mnist.png", xs, cmap="gray")

    # exit()

    # x_0 = jax.random.uniform(next(rng_seq), (100, 28, 28, 1))
    # x_1 = sde.ula(
    #         x_0,
    #         lambda x : - grad_log_p(x),
    #         0.00001,
    #         1000,
    #         next(rng_seq))

    # print("Average log_p ULA:", jnp.mean(log_p(x_1)))


    # xs = utils.image_grid(10, 10, x_1)
    # print(jnp.max(jnp.abs(xs)))
    # print(jnp.mean(jnp.abs(xs)))

    # xs = jnp.squeeze(jax.lax.clamp(0.0, xs, 1.0))
    # plt.imsave("ula_mnist.png", xs, cmap="gray")

    # exit()

    global loss
    loss_fn = hk.transform(loss)
    sample_fn = hk.transform(sample)

    params = loss_fn.init(next(rng_seq))
    params = hk.data_structures.map(
            lambda m, n, v: jnp.zeros(v.shape) if m.endswith("linear_4") else v, params)

    opt = optax.rmsprop(FLAGS.learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def update(rng, params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn.apply)(params, rng)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    # for i in tqdm(range(FLAGS.training_iterations)):
    for i in range(FLAGS.training_iterations):
        params, opt_state, loss = update(next(rng_seq), params, opt_state)
        print(i, float(loss))

    xs = sample_fn.apply(params, next(rng_seq))
    # print("Average log_p Follmer:", jnp.mean(log_p(xs)))
    xs = utils.image_grid(10, 10, xs)
    xs = jax.nn.sigmoid(xs)
    # print(jnp.max(jnp.abs(xs)))
    # print(jnp.mean(jnp.abs(xs)))
    # xs = jax.lax.clamp(0.0, xs, 1.0)
    xs = jnp.squeeze(xs)
    plt.imsave("follmer_mnist.png", xs, cmap="gray")

    # x_0 = sample_fn.apply(params, next(rng_seq))
    # x_1 = sde.ula(
    #         x_0,
    #         lambda x : - grad_log_p(x),
    #         0.00001,
    #         1000,
    #         next(rng_seq))

    # print("Average log_p ULA:", jnp.mean(log_p(x_1)))


    # xs = utils.image_grid(10, 10, x_1)
    # print(jnp.max(jnp.abs(xs)))
    # print(jnp.mean(jnp.abs(xs)))

    # # xs = jax.lax.clamp(0.0, xs, 1.0)
    # xs = jnp.squeeze(xs)
    # plt.imsave("ula_mnist.png", xs, cmap="gray")

    # exit()


if __name__ == "__main__":
    app.run(main)

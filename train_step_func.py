import jax
import jax.numpy as jnp
import haiku as hk
import optax

from jax import flatten_util

from absl import flags
from absl import app

from src import modules
from src import models

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

def dnn(x):
    return hk.nets.MLP([100, 100, 1])(x)

dnn_fn = hk.without_apply_rng(hk.transform(dnn))
params_dnn = dnn_fn.init(1, jnp.array([[0.0]]))
params_dnn_flat, unravel_func = flatten_util.ravel_pytree(params_dnn)
dim = params_dnn_flat.shape[0]

dnn_batch_flat_fn = jax.vmap(lambda params, x : dnn_fn.apply(unravel_func(params), x), (0, None))

def log_likelihood(params_dnn, x, y):
    y_pred = dnn_fn.apply(unravel_func(params_dnn), x)
    diff = y - y_pred

    return - jnp.sum(diff**2) / (2 * FLAGS.sigma_n**2)

def log_prior(params_dnn):
    return - jnp.sum(params_dnn**2) / (2 * FLAGS.sigma_p**2)

def log_posterior(params_dnn, x, y):
    return log_prior(params_dnn) + log_likelihood(params_dnn, x, y)

log_posterior_batch = jax.vmap(log_posterior, (0, None, None))


def prep_data(rng):
    rng, subrng = jax.random.split(rng)

    # Test inputs
    X_train = jax.random.uniform(subrng, (FLAGS.N_train, 1))
    X_train = X_train * 7 - 3.5
    X_test = jnp.linspace(-10, 10, FLAGS.N_test).reshape(-1, 1)

    y_train = jnp.heaviside(X_train, 0)
    y_test = jnp.heaviside(X_test, 0)

    y_train = y_train + FLAGS.sigma_n * jax.random.normal(rng, y_train.shape)


    return (X_train, y_train), (X_test, y_test)

def plot(X_train, y_train, X_test, y_test, params_dnn_batch=None, fn=None):
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.plot(X_test, y_test, 'b', label = 'f(x)')
    ax.plot(X_train, y_train, 'kP', ms = 9, label = 'y(x) = f(x) + $\epsilon$')
    ax.legend(loc = 'upper left')

    if params_dnn_batch is not None:
        y_preds = dnn_batch_flat_fn(params_dnn_batch, X_test)
        y_preds = jnp.squeeze(y_preds)

        mean = jnp.mean(y_preds, axis=0)
        std = jnp.std(y_preds, axis=0)

        X_test = jnp.squeeze(X_test)

        ax.plot(X_test, mean, label="mean function")
        ax.fill_between(X_test, mean - 2 * std, mean + 2 * std, alpha=0.2, label="mean function confidence bounds")

    if fn is None:
        plt.show(fig)
    else:
        fig.savefig(fn)
        plt.close(fig)


def build_follmer():
    drift = modules.Drift(hk.nets.MLP([300, 300, 300, 300, dim]))
    follmer = models.Follmer(drift, FLAGS.gamma, (dim,))
    return follmer


def loss(x, y):
    follmer = build_follmer()
    return follmer.relative_entropy_control_cost(
            FLAGS.batch_size_train,
            FLAGS.n_train_steps,
            lambda params_dnn : log_posterior_batch(params_dnn, x, y),
            )

def sample():
    follmer = build_follmer()
    return follmer.sample(
            FLAGS.batch_size_test,
            FLAGS.n_test_steps,
            )


def main(_):
    rng_seq = hk.PRNGSequence(FLAGS.seed)
    (X_train, y_train), (X_test, y_test) = prep_data(next(rng_seq))
    plot(X_train, y_train, X_test, y_test, fn="stepfunc.png")

    global loss
    loss_fn = hk.transform(loss)
    sample_fn = hk.transform(sample)

    params = loss_fn.init(next(rng_seq), X_train, y_train)
    params = hk.data_structures.map(
            lambda m, n, v: jnp.zeros(v.shape) if m.endswith("linear_4") else v, params)

    opt = optax.rmsprop(FLAGS.learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def update(rng, params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn.apply)(params, rng, X_train, y_train)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    # for i in tqdm(range(FLAGS.training_iterations)):
    for i in range(FLAGS.training_iterations):
        params, opt_state, loss = update(next(rng_seq), params, opt_state)
        print(i, float(loss))

    params_dnn_batch = sample_fn.apply(params, next(rng_seq))

    plot(X_train, y_train, X_test, y_test, params_dnn_batch=params_dnn_batch, fn="stepfunc_preds.png")

if __name__ == "__main__":
    app.run(main)

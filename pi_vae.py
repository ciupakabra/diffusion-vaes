from absl import app

from functools import partial

import jax
import jax.numpy as jnp
import numpy
import haiku as hk
import optax

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from tqdm.auto import tqdm

def em(rng, initial_state, drift, gamma, n_steps):
    ts = jnp.linspace(0, 1, n_steps + 1)
    dt = 1 / n_steps

    def em_step(carry, t):
        rng, state = carry
        rng, subrng = jax.random.split(rng)

        noise_term = jax.random.normal(subrng, state.shape)
        drift_term = drift(state, t)
        
        new_state = state + dt * drift_term + jnp.sqrt(dt * gamma) * noise_term
        new_carry = (rng, new_state)

        return new_carry, drift_term

    initial_carry = (rng, initial_state)

    (_, final_state), drifts = jax.lax.scan(em_step, initial_carry, ts[:-1])

    return final_state, drifts

def relative_entropy_control_cost(rng, batch_size, dim, drift, gamma, log_p, n_steps):
    dt = 1 / n_steps

    state_0 = jnp.zeros((batch_size, dim))
    state_1, drifts = em(rng, state_0, drift, gamma, n_steps)

    energy_cost = jnp.sum(drifts**2, axis=(0, 2)) * dt / (2 * gamma)
    terminal_cost = - jnp.sum(state_1**2, axis=1) / (2 * gamma) - log_p(state_1)

    return jnp.mean(energy_cost + terminal_cost)

def log_likelihood(images, zs, f):
    logits = f(zs)
    # dist = tfd.Independent(tfd.Bernoulli(logits=logits, validate_args=True))
    dist = tfd.MultivariateNormalDiag(jax.nn.sigmoid(logits), jnp.ones(logits.shape), validate_args=True)
    return dist.log_prob(images)

def log_prior(zs):
    dist = tfd.MultivariateNormalDiag(jnp.zeros(zs.shape), jnp.ones(zs.shape))
    return dist.log_prob(zs)

def log_posterior(zs, images, f):
    return log_prior(zs) + log_likelihood(images, zs, f)

def drift_fn(images, zs, t):
    t = t * jnp.ones(zs.shape[0])
    inp = jnp.hstack((images, zs, t[:, None]))

    drift_network = hk.Sequential([
        hk.Linear(512), jax.nn.relu,
        hk.Linear(512), jax.nn.relu,
        hk.Linear(10),
    ])

    drift = drift_network(inp)

    return drift

def model_fn(zs):
    mlp = hk.Sequential([
        hk.Linear(512), jax.nn.relu,
        hk.Linear(512), jax.nn.relu,
        hk.Linear(28 * 28),
    ])

    return mlp(zs)

def load_dataset(split, shuffle=False, batch_size=100, binarize=False):
    ds = tfds.load("mnist", split=split).cache().repeat()

    if shuffle:
        ds = ds.shuffle(10 * batch_size, seed=0)

    ds = ds.batch(batch_size).map(lambda x: tf.reshape(tf.cast(x["image"], tf.float32), (-1, 28 * 28)) / 255.)

    if binarize:
        ds = ds.map(lambda x : tf.math.round(x))

    return ds.as_numpy_iterator()

def main(_):
    batch_size = 100
    lr = 1e-4
    iters = 10000
    dt = 0.05
    n_steps = int(1 / dt)

    train_data = load_dataset("train", shuffle=True, batch_size=batch_size)
    train_data_eval = load_dataset("train", shuffle=False, batch_size=10000)
    test_data_eval = load_dataset("test", shuffle=False, batch_size=10000)

    drift = hk.without_apply_rng(hk.transform(drift_fn))
    model = hk.without_apply_rng(hk.transform(model_fn))


    def image_sample(rng, params, nrow, ncol):
      _, model_params = params

      sampled_images = model.apply(model_params, jax.random.normal(rng, (nrow * ncol, 10)))
      sampled_images = jax.nn.sigmoid(sampled_images)
      return image_grid(nrow, ncol, sampled_images, (28, 28))

    def image_grid(nrow, ncol, imagevecs, imshape):
      """Reshape a stack of image vectors into an image grid for plotting."""
      images = iter(imagevecs.reshape((-1,) + imshape))
      return jnp.vstack([jnp.hstack([next(images).T for _ in range(ncol)][::-1])
                        for _ in range(nrow)]).T

    def loss(rng, params, images, gamma, n_steps):
        drift_params, model_params = params

        log_p = lambda zs : log_posterior(zs, images, partial(model.apply, model_params))
        drift_ = partial(drift.apply, drift_params, images)

        return relative_entropy_control_cost(rng, images.shape[0], 10, drift_, gamma, log_p, n_steps)

    rng = jax.random.PRNGKey(42)
    init_drift_params = drift.init(jax.random.PRNGKey(42), next(train_data), jnp.zeros((batch_size, 10)), 0.0)
    init_model_params = model.init(jax.random.PRNGKey(43), jnp.zeros(10))

    params = (init_drift_params, init_model_params)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @partial(jax.jit, static_argnums=(5,))
    def update(rng, params, opt_state, images, gamma, n_steps):
        loss_ = lambda params : loss(rng, params, images, gamma, n_steps)
        loss_, grads = jax.value_and_grad(loss_)(params)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_

    train_losses = []
    test_losses = []

    test_rng = jax.random.PRNGKey(1)

    gammas = jnp.flip(jnp.linspace(0.5, 1.0, iters))**2

    for i in tqdm(range(iters)):
        params, opt_state, loss_ = update(jax.random.PRNGKey(i), params, opt_state, next(train_data), gammas[i], n_steps)
        train_losses.append(loss_)
        # test_losses.append(elbo(test_rng, params, next(test_data_eval)))

    train_losses = jnp.array(train_losses)
    # test_losses = jnp.array(test_losses)

    sampled_images = image_sample(jax.random.PRNGKey(0), params, 10, 10)
    plt.imsave("samples.png", sampled_images, cmap=plt.cm.gray)

    plt.plot(train_losses)
    # plt.plot(test_losses)
    # plt.ylim(-150, -70)
    plt.savefig("losses.png")

if __name__ == "__main__":
    app.run(main)

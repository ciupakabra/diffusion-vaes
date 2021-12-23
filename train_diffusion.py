from typing import Generator, Mapping, Tuple, NamedTuple, Sequence

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import models
import utils

from tqdm.auto import tqdm

tf.config.experimental.set_visible_devices([], "GPU")

flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
flags.DEFINE_integer("em_steps", 20, "Number of steps for Euler-Maruyama (during training)")
flags.DEFINE_integer("em_steps_test", 100, "Number of Euler-Maruyama steps on test")
flags.DEFINE_integer("hidden_size", 512, "Number of neurons in hidden layers of MLPs")
flags.DEFINE_integer("latent_size", 10, "Latent variable dimension")
flags.DEFINE_float("gamma", 0.1, "\sqrt{\gamma} for the diffusion coefficient")
flags.DEFINE_enum("likelihood", models.BERNOULLI, models.LIKELIHOODS, "Likelihood to use in the model")

FLAGS = flags.FLAGS

def main(_):

    binarize = FLAGS.likelihood == models.BERNOULLI

    train_data = utils.load_dataset("train", shuffle=True, batch_size=FLAGS.batch_size, binarize=binarize)
    test_data = utils.load_dataset("test", shuffle=False, batch_size=FLAGS.batch_size, binarize=binarize)

    model_args = (FLAGS.gamma**2, FLAGS.hidden_size, FLAGS.latent_size, FLAGS.likelihood)
    loss_fn = hk.transform(lambda y : models.DiffusionVAE(*model_args).relative_entropy_control_cost(y, FLAGS.em_steps))
    sample_fn = hk.transform(lambda n : models.DiffusionVAE(*model_args).sample(n, FLAGS.em_steps_test))

    rng_seq = hk.PRNGSequence(FLAGS.random_seed)
    params = loss_fn.init(next(rng_seq), next(train_data))

    opt = optax.adam(FLAGS.learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def update(rng, params, opt_state, y):
        loss, grads = jax.value_and_grad(loss_fn.apply)(params, rng, y)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss


    train_losses = []
    for i in tqdm(range(FLAGS.training_steps)):
        params, opt_state, loss = update(next(rng_seq), params, opt_state, next(train_data))
        train_losses.append(loss)

        if (i + 1) % FLAGS.eval_frequency == 0:
            sampled_images = utils.image_sample(next(rng_seq), sample_fn, params, 10, 10)
            plt.imsave("samples.png", sampled_images, cmap=plt.cm.gray)

    train_losses = jnp.array(train_losses)

    plt.plot(train_losses)
    plt.ylim(70, 150)
    plt.savefig("losses.png")

if __name__ == "__main__":
    app.run(main)

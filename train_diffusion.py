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
import modules
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
flags.DEFINE_integer("latent_size", 10, "Latent variable dimension")
flags.DEFINE_float("sqrt_gamma", 0.1, "\sqrt{\gamma} for the diffusion coefficient")
flags.DEFINE_enum("likelihood", models.BERNOULLI, models.LIKELIHOODS, "Likelihood to use in the model")
flags.DEFINE_enum("dataset", utils.MNIST, utils.DATASETS, "Dataset to use")
flags.DEFINE_string("outdir", None, "Output directory")

FLAGS = flags.FLAGS


def main(_):

    binarize = FLAGS.likelihood == models.BERNOULLI

    train_data = utils.load_dataset("train", shuffle=True, batch_size=FLAGS.batch_size, binarize=binarize, dataset=FLAGS.dataset)
    test_data = utils.load_dataset("test", shuffle=False, batch_size=FLAGS.batch_size, binarize=binarize, dataset=FLAGS.dataset)

    def build_model():

        if FLAGS.dataset == utils.MNIST:
            data_encoder=modules.MNIST_CNN_Encoder(FLAGS.latent_size)
            decoder = modules.MNIST_CNN_Decoder()
        elif FLAGS.dataset == utils.CELEBA:
            data_encoder=modules.CELEBA_CNN_Encoder(FLAGS.latent_size)
            decoder = modules.CELEBA_CNN_Decoder()


        control_drift = modules.AmortizedDrift(
                encoder=modules.Encoder(FLAGS.latent_size, [512, 512]),
                data_encoder=data_encoder,
        )
        
        model_drift = modules.Drift(
                encoder=modules.Encoder(FLAGS.latent_size, [512, 512]),
        )

        model = models.DiffusionVAE(
                latent_size=FLAGS.latent_size,
                control_drift=control_drift,
                model_drift=model_drift,
                decoder=decoder,
                gamma=FLAGS.sqrt_gamma**2,
                likelihood=FLAGS.likelihood,
        )

        return model

    loss_fn = hk.transform_with_state(lambda y : build_model().relative_entropy_control_cost(y, FLAGS.em_steps, True))
    sample_fn = hk.transform_with_state(lambda n : build_model().sample(n, FLAGS.em_steps_test))

    rng_seq = hk.PRNGSequence(FLAGS.random_seed)
    params, state = loss_fn.init(next(rng_seq), next(train_data))

    opt = optax.adam(FLAGS.learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def update(rng, params, state, opt_state, y):
        (loss, new_state), grads = jax.value_and_grad(loss_fn.apply, has_aux=True)(params, state, rng, y)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, opt_state, loss


    train_losses = []

    for i in tqdm(range(FLAGS.training_steps)):
        params, state, opt_state, loss = update(next(rng_seq), params, state, opt_state, next(train_data))
        train_losses.append(loss)

        if FLAGS.outdir is not None:
            if (i + 1) % FLAGS.eval_frequency == 0:
                nth_sample = (i + 1) // FLAGS.eval_frequency
                sampled_images, _ = sample_fn.apply(params, state, next(rng_seq), 100)
                sampled_images = utils.image_grid(10, 10, sampled_images)

                if sampled_images.shape[-1] == 1:
                    sampled_images = sampled_images[:, :, 0]

                plt.imsave(FLAGS.outdir + "/sample-{}.png".format(nth_sample), sampled_images, cmap=plt.cm.gray)

                jnp.save(FLAGS.outdir + "/params.npy", params)
                jnp.save(FLAGS.outdir + "/state.npy", state)

                plt.plot(jnp.array(train_losses))
                plt.savefig(FLAGS.outdir + "/losses.png")
                plt.close()

if __name__ == "__main__":
    app.run(main)

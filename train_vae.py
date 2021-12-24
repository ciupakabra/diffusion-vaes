from absl import app
from absl import flags

import jax
import jax.numpy as jnp
import numpy
import haiku as hk
import optax

import models
import modules
import utils

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
flags.DEFINE_integer("latent_size", 10, "Latent variable dimension")
flags.DEFINE_enum("likelihood", models.BERNOULLI, models.LIKELIHOODS, "Likelihood to use in the model")
flags.DEFINE_enum("dataset", utils.MNIST, utils.DATASETS, "Dataset to use")

FLAGS = flags.FLAGS

def main(_):
    binarize = FLAGS.likelihood == models.BERNOULLI

    train_data = utils.load_dataset(
            "train", 
            shuffle=True, 
            batch_size=FLAGS.batch_size, 
            binarize=binarize,
            dataset=FLAGS.dataset)

    test_data = utils.load_dataset(
            "test", 
            shuffle=False, 
            batch_size=FLAGS.batch_size, 
            binarize=binarize,
            dataset=FLAGS.dataset)

    def build_model():

        if FLAGS.dataset == utils.MNIST:
            encoder = modules.MNIST_CNN_Encoder(2 * FLAGS.latent_size)
            decoder = modules.MNIST_CNN_Decoder()
        elif FLAGS.dataset == utils.CELEBA:
            encoder = modules.CELEBA_CNN_Encoder(2 * FLAGS.latent_size)
            decoder = modules.CELEBA_CNN_Decoder()


        model = models.VanillaVAE(
                latent_size=FLAGS.latent_size,
                encoder=encoder,
                decoder=decoder,
                likelihood=FLAGS.likelihood,
        )

        return model

    loss_fn = hk.transform_with_state(lambda y: - build_model().elbo(y, True))
    sample_fn = hk.transform_with_state(lambda n: build_model().sample(n, False))

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

        if (i + 1) % FLAGS.eval_frequency == 0:
            sampled_images, _ = sample_fn.apply(params, state, next(rng_seq), 100)
            sampled_images = utils.image_grid(10, 10, sampled_images)

            if sampled_images.shape[-1] == 1:
                sampled_images = sampled_images[:, :, 0]

            plt.imsave("samples.png", sampled_images, cmap=plt.cm.gray)

    train_losses = jnp.array(train_losses)

    plt.plot(train_losses)
    plt.ylim(70, 150)
    plt.savefig("losses.png")

if __name__ == "__main__":
    app.run(main)

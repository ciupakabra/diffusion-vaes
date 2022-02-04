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

import pickle

from src import utils
from src import modules
from src import models
# from src.replay_buffer import ReplayBuffer
from src import replay_buffer

from src import sde

from datetime import datetime

from tqdm.auto import tqdm

tf.config.experimental.set_visible_devices([], "GPU")

flags.DEFINE_integer("pos_batch_size", 128, "Number of data samples to take while training.")
flags.DEFINE_integer("neg_batch_size", 128, "Number of samples to generate from the model while training.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
flags.DEFINE_integer("ula_steps", 60, "Number of steps for Euler-Maruyama (during training)")
flags.DEFINE_float("step_size", 0.005, "ULA step size")
flags.DEFINE_float("alpha", 0.1, "Regularization coefficient for the loss")
flags.DEFINE_enum("dataset", utils.MNIST, utils.DATASETS, "Dataset to use")
flags.DEFINE_string("outdir", None, "Output directory")
flags.DEFINE_boolean("replay_buffer", False, "Whether to use a replay buffer. Otherwise samples non-persistently")
flags.DEFINE_integer("replay_capacity", 10000, "Capacity of the replay buffer")
flags.DEFINE_float("replay_new_frac", 0.05, "Fraction of new samples from the replay buffer")

FLAGS = flags.FLAGS

def build_model():
    potential = modules.MNIST_CNN_Encoder(1)
    # potential = hk.nets.ResNet18(1)
    return potential

def loss(y_pos, y_neg):
    potential = build_model()

    y = jnp.concatenate([y_pos, y_neg])
    E = potential(y, True)
    E_pos, E_neg = jnp.split(E, 2)

    cdiv_loss = (jnp.mean(E_pos) - jnp.mean(E_neg))
    reg_loss = FLAGS.alpha * (jnp.mean(E_pos**2) + jnp.mean(E_neg**2))

    metrics = {
            "train_loss" : reg_loss + cdiv_loss,
            "reg_loss": reg_loss,
            "cdiv_loss": cdiv_loss,
            "neg_energy": jnp.mean(E_neg),
            "pos_energy": jnp.mean(E_pos),
    }

    return metrics["train_loss"], metrics

def sample(y_0, n_steps):
    potential = build_model()

    grad_U = jax.grad(lambda y : potential(y, False).sum())

    samples = sde.ula(
            y_0,
            grad_U,
            FLAGS.step_size,
            n_steps,
            hk.next_rng_key(),)

    return samples

def main(_):
    train_data = utils.load_dataset("train", shuffle=True, batch_size=FLAGS.pos_batch_size, binarize=False, dataset=FLAGS.dataset)
    test_data = utils.load_dataset("test", shuffle=False, batch_size=FLAGS.pos_batch_size, binarize=False, dataset=FLAGS.dataset)

    logdir = FLAGS.outdir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    # loss_fn = hk.transform(loss)
    loss_fn = hk.transform_with_state(loss)
    sample_fn = hk.transform_with_state(sample)
    # sample_fn = jax.jit(sample_fn.apply, static_argnums=(3,))
    sample_fn = jax.jit(sample_fn.apply, static_argnums=(4,))


    rng_seq = hk.PRNGSequence(FLAGS.random_seed)

    # params = loss_fn.init(next(rng_seq), next(train_data), next(train_data))
    params, state = loss_fn.init(next(rng_seq), next(train_data), next(train_data))

    sn_fn = hk.transform_with_state(lambda p : hk.SNParamsTree(n_steps=10, ignore_regex=".*\/b$")(p))
    sn_fn = hk.without_apply_rng(sn_fn)
    _, sn_state = sn_fn.init(next(rng_seq), params)

    opt = optax.adamw(FLAGS.learning_rate, weight_decay=0.1)
    opt_state = opt.init(params)

    if FLAGS.replay_buffer:
        rb = replay_buffer.create(FLAGS.replay_capacity, (28, 28, 1))
        rb = replay_buffer.store(rb, jax.random.uniform(next(rng_seq), (FLAGS.neg_batch_size, 28, 28, 1)))

    def loss_func(params, state, rng, y_pos, y_neg):
        (loss, metrics), state = loss_fn.apply(params, state, rng, y_pos, y_neg)
        return loss, (metrics, state)

    @jax.jit
    def update(rng, params, y_pos, y_neg, opt_state, state, sn_state):
        (_, (metrics, state)), grads = jax.value_and_grad(loss_func, has_aux=True)(params, state, rng, y_pos, y_neg)

        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # new_params, new_sn_state = sn_fn.apply(None, sn_state, new_params)
        return new_params, opt_state, metrics, state, None

    for i in tqdm(range(FLAGS.training_steps)):

        if FLAGS.replay_buffer:
            y_neg = replay_buffer.sample(next(rng_seq), rb, FLAGS.neg_batch_size, FLAGS.replay_new_frac, (28, 28, 1))
        else:
            y_neg = jax.random.uniform(next(rng_seq), (FLAGS.neg_batch_size,) + (28, 28, 1))
            
        y_neg, _ = sample_fn(params, state, next(rng_seq), y_neg, FLAGS.ula_steps)
        y_pos = next(train_data)

        params, opt_state, metrics, state, sn_state = update(
                next(rng_seq),
                params,
                y_pos,
                y_neg,
                opt_state,
                state,
                sn_state,
                )

        for k, v in metrics.items():
            tf.summary.scalar("train_metrics/" + k, data=v, step=i)

        for m, n, v in hk.data_structures.traverse(params):
            tf.summary.scalar("parameter_norms/" + m + "_" + n, data=jnp.linalg.norm(v), step=i)

        if FLAGS.replay_buffer:
            rb = replay_buffer.store(rb, y_neg)

        if FLAGS.outdir is not None:
            if (i + 1) % FLAGS.eval_frequency == 0:
                nth_sample = (i + 1) // FLAGS.eval_frequency

                sampled_images = jax.random.uniform(next(rng_seq), (100, 28, 28, 1))
                sampled_images, _ = sample_fn(params, state, next(rng_seq), sampled_images, 500)
                sampled_images = utils.image_grid(10, 10, sampled_images)
                sampled_images = jax.lax.clamp(0.0, sampled_images, 1.0)
                sampled_images = sampled_images[None, ...]
                tf.summary.image("samples/sample", data=sampled_images, step=i)

                if FLAGS.replay_buffer:
                    buffer_images = replay_buffer.sample(next(rng_seq), rb, 100, 0.0, (28, 28, 1))
                    buffer_images = utils.image_grid(10, 10, buffer_images)
                    buffer_images = jax.lax.clamp(0.0, buffer_images, 1.0)
                    buffer_images = buffer_images[None, ...]
                    tf.summary.image("samples/buffer", data=buffer_images, step=i)

                
                with open(FLAGS.outdir + "/params.npy", "wb") as f:
                    pickle.dump(params, f)
                with open(FLAGS.outdir + "/state.npy", "wb") as f:
                    pickle.dump(state, f)

if __name__ == "__main__":
    app.run(main)

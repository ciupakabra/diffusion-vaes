from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle

import matplotlib.pyplot as plt

import tensorflow as tf

from src import utils
from src import modules
from src import models
# from src.replay_buffer import ReplayBuffer
from src import replay_buffer
from src import constraints
from src import sde

from datetime import datetime

from tqdm.auto import tqdm

tf.config.experimental.set_visible_devices([], "GPU")

flags.DEFINE_integer("pos_batch_size", 128, "Number of data samples to take while training.")
flags.DEFINE_integer("neg_batch_size", 128, "Number of samples to generate from the model while training.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 1000, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
flags.DEFINE_integer("n_steps", 20, "Number of steps for Euler-Maruyama (during training)")
flags.DEFINE_integer("every_n_update", 20, "Number of steps to train Follmer for one step of potential")
flags.DEFINE_float("alpha", 0.0, "Regularization coefficient for the loss")
flags.DEFINE_float("gamma", 1.0, "gamma for Follmer")
flags.DEFINE_enum("dataset", utils.MNIST, utils.DATASETS, "Dataset to use")
flags.DEFINE_string("outdir", None, "Output directory")

FLAGS = flags.FLAGS

def build_ebm():
    potential = modules.MNIST_CNN_Encoder(1)
    return potential

def log_p(y):
    potential = build_ebm()
    # return - potential(y, False)
    return - jnp.squeeze(potential(y, False))

log_p_fn = hk.without_apply_rng(hk.transform_with_state(log_p))

class Drift(hk.Module):
    def __init__(self):
        super().__init__()
        self.drift = modules.Drift(hk.nets.MLP([512, 512, 784]))
        self.reshape_1 = hk.Reshape((-1,))
        self.reshape_2 = hk.Reshape((28, 28, 1))
        
    def __call__(self, x, t, is_training):
        out = self.reshape_1(x)
        out = self.drift(out, t, is_training)
        out = self.reshape_2(out)
        return out

def build_follmer():
    drift = Drift()
    grad_mult = hk.nets.MLP([512, 512, 1])
    follmer = models.ModulatedFollmer(drift, grad_mult, FLAGS.gamma, (28, 28, 1))
    return follmer

def loss_potential(y_pos, y_neg):
    potential = build_ebm()

    y = jnp.concatenate([y_pos, y_neg])
    E = potential(y, True)
    E_pos, E_neg = jnp.split(E, 2)

    cdiv_loss = (jnp.mean(E_pos) - jnp.mean(E_neg))
    norm = jnp.mean(jnp.abs(E))
    reg_loss = FLAGS.alpha * (jnp.mean(E_pos**2) + jnp.mean(E_neg**2))

    metrics = {
            "train_loss" : reg_loss + cdiv_loss,
            "reg_loss": reg_loss,
            "cdiv_loss": cdiv_loss,
            "neg_energy": jnp.mean(E_neg),
            "pos_energy": jnp.mean(E_pos),
            "norm": norm,
    }

    return metrics["train_loss"], metrics

def sample(batch_size, n_steps):
    follmer = build_follmer()
    x_samples = follmer.sample(batch_size, n_steps)
    return jax.nn.sigmoid(x_samples)
    # return follmer.sample(batch_size, n_steps)

def loss_follmer(params_potential, state_potential):
    follmer = build_follmer()

    V = lambda y : log_p_fn.apply(params_potential, state_potential, y)[0]
    W = constraints.constrain(V)

    return follmer.relative_entropy_control_cost(
            FLAGS.neg_batch_size,
            FLAGS.n_steps,
            W,
            # lambda y : log_p_fn.apply(params_potential, state_potential, y)[0]
            )

def main(_):
    train_data = utils.load_dataset("train", shuffle=True, batch_size=FLAGS.pos_batch_size, binarize=False, dataset=FLAGS.dataset)
    test_data = utils.load_dataset("test", shuffle=False, batch_size=FLAGS.pos_batch_size, binarize=False, dataset=FLAGS.dataset)

    logdir = FLAGS.outdir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    loss_pot_fn = hk.transform_with_state(loss_potential)
    loss_fol_fn = hk.transform_with_state(loss_follmer)
    sample_fn = hk.transform_with_state(sample)
    sample_fn = jax.jit(sample_fn.apply, static_argnums=(3, 4,))

    rng_seq = hk.PRNGSequence(FLAGS.random_seed)

    # params_pot, state_pot = loss_pot_fn.init(next(rng_seq), next(train_data), next(train_data))
    with open("outputs/ebm-replay-buffer/params.npy", "rb") as f:
        params_pot = pickle.load(f)
    with open("outputs/ebm-replay-buffer/state.npy", "rb") as f:
        state_pot = pickle.load(f)
    params_fol, state_fol = loss_fol_fn.init(next(rng_seq), params_pot, state_pot)
    params_fol = hk.data_structures.map(lambda m, n, v: jnp.zeros(v.shape) if m.endswith("linear_2") else v, params_fol)

    opt_pot = optax.adamw(FLAGS.learning_rate, weight_decay=0.1)
    opt_pot_state = opt_pot.init(params_pot)

    opt_fol = optax.adamw(FLAGS.learning_rate, weight_decay=0.0)
    opt_fol_state = opt_fol.init(params_fol)

    def loss_pot(params_pot, state_pot, rng, y_pos, y_neg):
        (loss, metrics), state_pot = loss_pot_fn.apply(
                params_pot, 
                state_pot, 
                rng, 
                y_pos, 
                y_neg,
                )
        return loss, (metrics, state_pot)

    @jax.jit
    def update_pot(rng, params_pot, state_pot, opt_pot_state, y_pos, y_neg):
        (_, (metrics, state_pot)), grads = jax.value_and_grad(loss_pot, has_aux=True)(
                params_pot, 
                state_pot, 
                rng, 
                y_pos, 
                y_neg,
                )

        updates, opt_pot_state = opt_pot.update(grads, opt_pot_state, params_pot)
        params_pot = optax.apply_updates(params_pot, updates)
        return params_pot, state_pot, opt_pot_state, metrics

    @jax.jit
    def update_fol(rng, params_fol, state_fol, params_pot, state_pot, opt_fol_state):
         (loss, state_fol), grads = jax.value_and_grad(loss_fol_fn.apply, has_aux=True)(
                 params_fol, 
                 state_fol,
                 rng,
                 params_pot,
                 state_pot,
                 )

         updates, opt_fol_state = opt_fol.update(grads, opt_fol_state, params_fol)
         params_fol = optax.apply_updates(params_fol, updates)
         return params_fol, state_fol, opt_fol_state, loss

    ebm_updates = 0

    for i in tqdm(range(FLAGS.training_steps)):
        # y_neg, _ = sample_fn(
        #         params_fol, 
        #         state_fol, 
        #         next(rng_seq),
        #         FLAGS.neg_batch_size, 
        #         FLAGS.n_steps)

        # y_pos = next(train_data)

        # _, (metrics, _) = loss_pot(params_pot, state_pot, next(rng_seq), y_pos, y_neg)

        # for k, v in metrics.items():
        #     tf.summary.scalar("train_metrics/" + k, data=v, step=i)

        # if (metrics["neg_energy"] - metrics["pos_energy"]) / jnp.abs(metrics["norm"]) < 0.1:
        if i % FLAGS.every_n_update == 0:
            y_neg, _ = sample_fn(
                    params_fol, 
                    state_fol, 
                    next(rng_seq),
                    FLAGS.neg_batch_size, 
                    FLAGS.n_steps)

            y_pos = next(train_data)

            params_pot, state_pot, opt_pot_state, metrics = update_pot(
                    next(rng_seq),
                    params_pot,
                    state_pot,
                    opt_pot_state,
                    y_pos,
                    y_neg,
                    )

            ebm_updates += 1

            for k, v in metrics.items():
                tf.summary.scalar("train_metrics/" + k, data=v, step=i)

            for m, n, v in hk.data_structures.traverse(params_pot):
                tf.summary.scalar("parameter_norms/" + m + "_" + n, data=jnp.linalg.norm(v), step=i)
        else:
            params_fol, state_fol, opt_fol_state, loss = update_fol(
                    next(rng_seq),
                    params_fol,
                    state_fol,
                    params_pot,
                    state_pot,
                    opt_fol_state,
                    )

            tf.summary.scalar("train_metrics/follmer_loss", data=loss, step=i)

        tf.summary.scalar("train_metrics/ebm_updates", data=ebm_updates, step=i)

        if FLAGS.outdir is not None:
            if (i + 1) % FLAGS.eval_frequency == 0:
                sampled_images, _ = sample_fn(
                        params_fol, 
                        state_fol, 
                        next(rng_seq), 
                        100,
                        100,)
                sampled_images = utils.image_grid(10, 10, sampled_images)
                sampled_images = jax.lax.clamp(0.0, sampled_images, 1.0)
                sampled_images = sampled_images[None, ...]
                tf.summary.image("samples/sample", data=sampled_images, step=i)

                sampled_images, _ = sample_fn(
                        params_fol, 
                        state_fol, 
                        next(rng_seq), 
                        100,
                        100,)
                sampled_images = utils.image_grid(10, 10, sampled_images)
                sampled_images = jax.lax.clamp(0.0, sampled_images, 1.0)
                sampled_images = sampled_images[None, ...]
                tf.summary.image("samples/extra_sample", data=sampled_images, step=i)

if __name__ == "__main__":
    app.run(main)

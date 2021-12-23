import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp

def load_dataset(split, shuffle=False, batch_size=100, binarize=False):
    ds = tfds.load("mnist", split=split).cache().repeat()

    if shuffle:
        ds = ds.shuffle(10 * batch_size, seed=0)

    ds = ds.batch(batch_size)
    ds = ds.map(lambda y: tf.reshape(tf.cast(y["image"], tf.float32), (-1, 28 * 28)) / 255.)

    if binarize:
        ds = ds.map(lambda y : tf.math.round(y))

    return ds.as_numpy_iterator()

def image_grid(nrow, ncol, imagevecs, imshape):
  """Reshape a stack of image vectors into an image grid for plotting."""
  images = iter(imagevecs.reshape((-1,) + imshape))
  return jnp.vstack([jnp.hstack([next(images).T for _ in range(ncol)][::-1])
                    for _ in range(nrow)]).T

def image_sample(rng, sample_fn, params, nrow, ncol):
    sampled_images = sample_fn.apply(params, rng, nrow * ncol)
    return image_grid(nrow, ncol, sampled_images, (28, 28))

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp

MNIST_SHAPE = (28, 28, 1)
CELEBA_SHAPE = (64, 64, 3)

MNIST = "mnist"
CELEBA = "celeb_a"
DATASETS = [MNIST, CELEBA]

def center_crop(batch, target_w):
    w, h = batch.shape[1], batch.shape[2]

    offset_w = (w - target_w) // 2
    offset_h = (h - target_w) // 2

    return tf.image.crop_to_bounding_box(batch, offset_w, offset_h, target_w, target_w)

def load_dataset(
        split: str, 
        shuffle: bool = False, 
        batch_size: int = 100, 
        binarize: bool = False,
        dataset: str = "mnist"
        ):
    ds = tfds.load(dataset, split=split).cache().repeat()

    if shuffle:
        ds = ds.shuffle(10 * batch_size, seed=0)

    ds = ds.batch(batch_size)
    ds = ds.map(lambda y: y["image"])
    ds = ds.map(lambda y: tf.cast(y, tf.float32))
    ds = ds.map(lambda y: y / 255.0)

    if binarize:
        ds = ds.map(lambda y: tf.math.round(y))

    if dataset == "celeb_a":
        ds = ds.map(lambda y: center_crop(y, 148))
        ds = ds.map(lambda y: tf.image.resize(y, CELEBA_SHAPE[:2]))

    return ds.as_numpy_iterator()

def image_grid(nrow, ncol, imagevecs):
  """Reshape a stack of image vectors into an image grid for plotting."""
  images = iter(imagevecs)
  return jnp.concatenate([jnp.concatenate([next(images) for _ in range(ncol)], axis=0)
                    for _ in range(nrow)], axis=1)

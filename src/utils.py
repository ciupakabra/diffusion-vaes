import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_probability.substrates.jax import distributions as tfd

import jax
import jax.numpy as jnp

from einops import rearrange

MNIST_SHAPE = (28, 28, 1)
CELEBA_SHAPE = (64, 64, 3)

MNIST = "mnist"
CELEBA = "celeb_a"
DATASETS = [MNIST, CELEBA]

BERNOULLI = "bernoulli"
GAUSSIAN = "gaussian"
LIKELIHOODS = [BERNOULLI, GAUSSIAN]

def log_likelihood(logits: jnp.ndarray, y: jnp.ndarray, distribution: str = BERNOULLI) -> jnp.array:
    if distribution == BERNOULLI:
        dist = tfd.Independent(tfd.Bernoulli(logits=logits))
    elif distribution == GAUSSIAN:
        dist = tfd.MultivariateNormalDiag(jax.nn.sigmoid(logits), jnp.ones(logits.shape))
        dist = tfd.Independent(dist)
    return dist.log_prob(y)

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
  return rearrange(imagevecs, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=nrow, b2=ncol)



def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)

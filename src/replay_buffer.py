from typing import Tuple
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
from functools import partial

from collections import namedtuple

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]

ReplayBuffer = namedtuple("ReplayBuffer", ["buffer", "size", "index", "capacity"])

@jax.jit
def store_(rb, data):
    new_buffer = jax.lax.dynamic_update_slice(rb.buffer, data, (rb.index, 0, 0, 0))
    return ReplayBuffer(
            buffer=new_buffer, 
            index=(rb.index + data.shape[0]) % rb.capacity, 
            size=jnp.min(jnp.array([rb.size + data.shape[0], rb.capacity])),
            capacity=rb.capacity)

def store(rb, data):
    data = jax.device_put(data, cpu)
    rows = data.shape[0]
    rows_to_top = rb.capacity - rb.index

    if rows_to_top < rows:
        rb = store_(rb, data[:rows_to_top])
        return store(rb, data[rows_to_top:])
    return store_(rb, data)

def create(capacity, data_shape):
    buffer = jax.device_put(jnp.full((capacity,) + data_shape, jnp.nan, jnp.float32), cpu)
    return ReplayBuffer(
            buffer=buffer, 
            index=0,
            size=0,
            capacity=capacity)

@partial(jax.jit, static_argnums=(2,))
def sample_(rng, rb, batch_size):
    idx = jax.random.randint(rng, (batch_size,), 0, rb.size)
    data = jnp.take(rb.buffer, idx, axis=0)
    return data

def sample(rng, rb, batch_size, new_frac, data_shape):
    n_new = round(batch_size * new_frac)
    n_old = batch_size - n_new

    rng, subrng = jax.random.split(rng)

    old_data = sample_(rng, rb, n_old)
    new_data = jax.random.uniform(rng, (n_new,) + data_shape)

    data = jnp.concatenate([old_data, new_data], axis=0)
    return jax.device_put(data, gpu)


import math
from typing import Any, Dict, Optional, TypeVar

import logging
import flax
import numpy as np
import jax
import jax.numpy as jnp
from flax.metrics import tensorboard
from PIL import Image
import tensorflow as tf
from jax import numpy as jnp
from tqdm import tqdm

T = TypeVar("T")

def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)

def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)

class SimpleLogger:
  def __init__(self, log_dir) -> None:
    self.writer = tensorboard.SummaryWriter(log_dir) 
    pass

  def log_loss(self, loss_val, step):
    logging.info("step: %d, training_loss: %.5e" % (step, loss_val))
    self.writer.scalar('training_loss', loss_val, step=step)

  def log_eval_loss(self, eval_loss_val, step):
    logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss_val))
    self.writer.scalar('eval_loss', eval_loss_val, step=step)


def load_training_state(filepath, state):
  with tf.io.gfile.GFile(filepath, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
  """Make a grid of images and save it into an image file.

  Pixel values are assumed to be within [0, 1].

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """
  if not (isinstance(ndarray, jnp.ndarray) or
          (isinstance(ndarray, list) and
           all(isinstance(t, jnp.ndarray) for t in ndarray))):
    raise TypeError("array_like of tensors expected, got {}".format(
      type(ndarray)))

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                       padding)
  num_channels = ndarray.shape[3]
  grid = jnp.full(
    (height * ymaps + padding, width * xmaps + padding, num_channels),
    pad_value).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[y * height + padding:(y + 1) * height,
                     x * width + padding:(x + 1) * width].set(ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
  im = Image.fromarray(np.array(ndarr).copy())
  im.save(fp, format=format)


def flatten_dict(config):
  """Flatten a hierarchical dict to a simple dict."""
  new_dict = {}
  for key, value in config.items():
    if isinstance(value, dict):
      sub_dict = flatten_dict(value)
      for subkey, subvalue in sub_dict.items():
        new_dict[key + "/" + subkey] = subvalue
    elif isinstance(value, tuple):
      new_dict[key] = str(value)
    else:
      new_dict[key] = value
  return new_dict


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    grad_fn = lambda data: jnp.sum(fn(data, t) * eps)
    grad_fn_eps = jax.grad(grad_fn)(x)
    return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

  return div_fn


def get_value_div_fn(fn):
  """Return both the function value and its estimated divergence via Hutchinson's trace estimator."""

  def value_div_fn(x, t, eps):
    def value_grad_fn(data):
      f = fn(data, t)
      return jnp.sum(f * eps), f
    grad_fn_eps, value = jax.grad(value_grad_fn, has_aux=True)(x)
    return value, jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

  return value_div_fn
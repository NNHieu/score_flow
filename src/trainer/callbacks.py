from typing import Any, Callable, Optional
from .trainer import CustomTrainState, PRNGKey, Trainer

import os
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import utils

STEP_OUTPUT = Any

class Callback:
  def setup(self, trainer: Trainer, stage: Optional[str] = None) -> None:
    """Called when fit, validate, test, predict, or tune begins."""

  def teardown(self, trainer: Trainer, stage: Optional[str] = None) -> None:
    """Called when fit, validate, test, predict, or tune ends."""

  def on_fit_start(self, trainer: Trainer) -> None:
    """Called when fit begins."""
  
  def on_fit_end(self, trainer: Trainer) -> None:
    """Called when fit ends."""
  
  def on_train_batch_start(
        self, trainer: Trainer, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
  
  def on_train_batch_end(
        self, trainer: Trainer, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends."""
  
  def on_validation_batch_start(
        self, trainer: Trainer, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch begins."""

  def on_validation_batch_end(
        self,
        trainer: Trainer,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""

  def on_save_checkpoint(
      self, trainer: Trainer, pstate: Any, saved_state: CustomTrainState,
    ) -> None:
        r"""
        Called when saving a checkpoint to give you a chance to store anything else you might want to save.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pstate: the current (replicate) :class:`~src.trainer.CustomTrainState` instance.
            saved_state: the current :class:`~src.trainer.CustomTrainState` that have been saved.
            rng: The current PRNGKey that is shared between callbacks

        Returns: None.
        """

  def on_exception(self, trainer: Trainer, exception: BaseException) -> None:
    """Called when any trainer execution is interrupted by an exception."""

class SamplingCallback(Callback):
  def __init__(self, sampling_fn: Callable, sample_dir: Any, rng: PRNGKey) -> None:
    self.sampling_fn = sampling_fn
    self.sample_dir = sample_dir
    tf.io.gfile.makedirs(sample_dir)
    self.rng = rng

  def on_save_checkpoint(self, trainer: Trainer, pstate: Any, saved_state: CustomTrainState):
    if trainer.is_multi_gpu:
      self.rng, *sample_rng = jax.random.split(self.rng, jax.local_device_count() + 1)
      sample_rng = jnp.asarray(sample_rng)
      sample, n = self.sampling_fn(sample_rng, pstate)
      this_sample_dir = os.path.join(
          self.sample_dir, "iter_{}_host_{}".format(step_idx, jax.process_index()))
      tf.io.gfile.makedirs(this_sample_dir)
      image_grid = sample.reshape((-1, *sample.shape[2:]))
      nrow = int(np.sqrt(image_grid.shape[0]))
      sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
      with tf.io.gfile.GFile(
          os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
        np.save(fout, sample)
      with tf.io.gfile.GFile(
          os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
        utils.save_image(image_grid, fout, nrow=nrow, padding=2)
    else:
        pass

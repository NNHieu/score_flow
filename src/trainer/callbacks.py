
from typing import Any, Callable, Optional

from .trainer import CustomTrainState, PRNGKey, Trainer

import os
import io
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import utils.utils as utils
from src import utils as sutils
from flax.jax_utils import replicate


log = sutils.get_pylogger(__name__)

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
        self, 
        trainer: Trainer, 
        state,
        outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends."""
  
  def on_validation_batch_start(
        self, trainer: Trainer, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch begins."""

  def on_validation_batch_end(
        self,
        trainer: Trainer,
        state,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
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
    if not trainer.strategy.is_multi_gpu:
      pstate = replicate(pstate)
    
    self.rng, *sample_rng = jax.random.split(self.rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
    sample, n = self.sampling_fn(sample_rng, pstate)
    this_sample_dir = os.path.join(
        self.sample_dir, "iter_{}_host_{}".format(trainer.global_step, jax.process_index()))
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
    log.info(f'Saved sampling at {this_sample_dir}')

class SaveBpd(Callback):
  def __init__(self, eval_dir, prefix) -> None:
    self.eval_dir = eval_dir
    tf.io.gfile.makedirs(eval_dir)
    self.prefix = prefix
  
  def on_validation_batch_end(self, trainer: Trainer, state, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    bpd = outputs.reshape(-1)
    # log.info("ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, jnp.mean(jnp.asarray(bpds))))
    # Save bits/dim to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join(self.eval_dir,
                                        f"{self.prefix}_bpd_{batch_idx}.npz"),
                            "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, bpd)
      fout.write(io_buffer.getvalue())
      

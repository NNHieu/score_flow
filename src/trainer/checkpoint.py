from enum import Enum
from typing import Any, Callable, Optional
from xml.etree.ElementInclude import include

from utils.run.rank_zero import rank_zero_only
from .callbacks import STEP_OUTPUT

import os
import tensorflow as tf
import numpy as np
from flax.training import checkpoints

from .callbacks import Callback

# from src.utils.run.pylogger import get_pylogger
# log = get_pylogger(__name__)
STEP_OUTPUT = Any

class CheckpointMode(Enum):
  NONE = 0
  NORMAL = 1
  PREEMPTION = 2
  NORMAL_PREEMPTION = 3

  def savable(self):
    return self.value > 0
  
  def include(self, mode):
    return (self.value & mode.value) > 0

class CheckpointCallback(Callback):
  def __init__(self, save_dir: os.PathLike, snapshot_freq, snapshot_freq_for_preemption) -> None:
    self.save_dir = save_dir
    tf.io.gfile.makedirs(self.save_dir)
    self.snapshot_freq = snapshot_freq
    self.snapshot_freq_for_preemption = snapshot_freq_for_preemption
    self.checkpoint_meta_dir = os.path.join(self.save_dir, "checkpoints_meta")
    self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")

  def save(self, mode: CheckpointMode, pstate: Any, saved_state, step_idx):
    # Preemption
    if mode == CheckpointMode.PREEMPTION:
      checkpoints.save_checkpoint(self.checkpoint_meta_dir, saved_state,
                                    step= step_idx // self.snapshot_freq_for_preemption,
                                    keep=1, overwrite=True)
    elif mode == CheckpointMode.NORMAL:
      checkpoints.save_checkpoint(self.checkpoint_dir, saved_state,
                                step = step_idx // self.snapshot_freq,
                                keep=np.inf, overwrite=True)
    
  @rank_zero_only
  def on_train_batch_end(self, trainer):
    mode = self._should_save(trainer.global_step, trainer.num_train_steps)
    if mode.savable():
      train_runner = trainer._get_train_runner()
      saved_state = train_runner.get_state()
      if mode.include(CheckpointMode.PREEMPTION):
        self.save(CheckpointMode.PREEMPTION, None, saved_state, trainer.global_step)
      if mode.include(CheckpointMode.NORMAL):
        self.save(CheckpointMode.NORMAL, None, saved_state, trainer.global_step)
        trainer._call_callbacks_on_save_checkpoint(train_runner.get_pstate(), train_runner.get_state())
    
  def _should_save(self, step_idx, num_train_steps):
    # Preemption
    mode = 0
    if step_idx > 0:
      if step_idx % self.snapshot_freq_for_preemption == 0:
        mode = mode | CheckpointMode.PREEMPTION.value
      if step_idx % self.snapshot_freq == 0 or step_idx == num_train_steps:
        mode = mode | CheckpointMode.NORMAL.value
    return CheckpointMode(mode)
  
  def _should_save_mode(self, step_idx, num_train_steps, mode: CheckpointMode):
    return self._should_save(step_idx, num_train_steps).include(CheckpointMode.PREEMPTION)

  def restore(self, state):
    # log.debug(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {self.checkpoint_meta_dir}")
    return checkpoints.restore_checkpoint(self.checkpoint_meta_dir, state)
from lib2to3 import pytree
import os
from sched import scheduler
from tkinter.tix import Tree
from typing import Callable, Any

import jax
import jax.random as jrnd
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import numpy as np
import tensorflow as tf
import functools
from flax.metrics import tensorboard
# import wandb
from flax.training import checkpoints
from tqdm import tqdm
# Keep the import below for registering all model definitions
from models import ncsnpp
import losses
import sde_lib
import utils
from models import utils as mutils
import src.datasets as datasets
from absl import flags
import sampling

import logging
# logger = logging.getLogger()
# logging.disable(logging.CRITICAL)

FLAGS = flags.FLAGS
PRNGKey = Any

class CustomTrainState(train_state.TrainState):
  rng: PRNGKey
  model_states: Any = flax.struct.field(pytree_node=False)

  def update_model(self, *, new_model_states, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        model_states = new_model_states,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, model_states, rng, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        rng = rng,
        model_states = model_states,
        **kwargs,
    )

def get_dataset(config):
  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)
  return train_ds, eval_ds

def get_optimizer(config, beta2=0.999):
  grad_clip = config.optim.grad_clip
  warmup = config.optim.warmup
  lr = config.optim.lr
  if config.optim.optimizer == 'Adam':
    schedule = lr
    if warmup > 0:
      schedule = optax.linear_schedule(1./ warmup, lr, warmup)
    opt = optax.chain(
      optax.clip(grad_clip),
      optax.adamw(learning_rate=schedule, b1=config.optim.beta1, b2=beta2, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
    )
    
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return opt


class Trainer(object):
  def __init__(self, config, workdir, train_step_fn, eval_step_fn, end_step = None) -> None:
    self.config = config
    self.workdir = workdir
    self.setup_dir()
    self.setup_logger()
    self.train_step_fn = jax.pmap(train_step_fn, axis_name='batch', donate_argnums=1)
    self.eval_step_fn = jax.pmap(eval_step_fn, axis_name='batch', donate_argnums=1)
    self.end_step = end_step

  def setup_dir(self):
    self.checkpoint_meta_dir = os.path.join(self.workdir, "checkpoints_meta")
    self.checkpoint_dir = os.path.join(self.workdir, "checkpoints")
    self.tb_dir = os.path.join(self.workdir, "tensorboard")
    tf.io.gfile.makedirs(self.tb_dir)
    if self.config.training.snapshot_sampling:
      self.sample_dir = os.path.join(self.workdir, "samples")
      tf.io.gfile.makedirs(self.sample_dir)
    
  def setup_logger(self):
    self.logger = utils.SimpleLogger(self.tb_dir)

  def save(self, saved_state, step, preemtion=False):
    if preemtion:
      checkpoints.save_checkpoint(self.checkpoint_meta_dir, saved_state,
                                    step=step // self.config.training.snapshot_freq_for_preemption,
                                    keep=1, overwrite=True)
    else:
      checkpoints.save_checkpoint(self.checkpoint_dir, saved_state,
                                step = step // self.config.training.snapshot_freq,
                                keep=np.inf, overwrite=True)
    
  def load(self, state):
    return checkpoints.restore_checkpoint(self.checkpoint_meta_dir, state)

  def fit(self, rng, state, train_ds, eval_ds):
    config = self.config
    state = self.load(state)
    initial_step = int(state.step)
    num_train_steps = config.training.n_iters
    # Replicate the training state to run on multiple devices
    pstate = flax.jax_utils.replicate(state)
    del state
    
    train_iter, eval_iter = iter(train_ds), iter(eval_ds)
    
    
    # pbar = tqdm(range(initial_step, num_train_steps + 1))
    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    if jax.host_id() == 0:
      logging.info("Starting training loop at step %d." % (initial_step,))
    rng = jax.random.fold_in(rng, jax.host_id())
    # JIT multiple training steps together for faster training
    n_jitted_steps = config.training.n_jitted_steps
    # Must be divisible by the number of steps jitted together
    assert config.training.log_freq % n_jitted_steps == 0 and \
          config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
          config.training.eval_freq % n_jitted_steps == 0 and \
          config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

    # for step in pbar:
    for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
      batch = next(train_iter)['image']._numpy()
      # rng, next_rng = jax.random.split(rng)
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, pstate), loss = self.train_step_fn((next_rng, pstate), batch)
      loss = flax.jax_utils.unreplicate(loss)
      self.logger.log_loss(loss.mean(), step)

      if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
        if jax.host_id() == 0:
          saved_state = flax.jax_utils.unreplicate(pstate)
          saved_state = saved_state.replace(rng=rng)
          self.save(saved_state, step, preemtion=True)
          del saved_state
      
      # Report the loss on an evaluation dataset periodically
      if step % config.training.eval_freq == 0:
        eval_batch = next(eval_iter)['image']._numpy()  # pylint: disable=protected-access
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        (_, _), eval_loss = self.eval_step_fn((next_rng, pstate), eval_batch)
        eval_loss = flax.jax_utils.unreplicate(eval_loss)
        self.logger.log_eval_loss(eval_loss.mean(), step)
      
      # Save a checkpoint periodically and generate samples if needed
      if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
        # Save the checkpoint.
          saved_state = flax.jax_utils.unreplicate(pstate)
          saved_state = saved_state.replace(rng=rng)
          self.save(saved_state, step, preemtion=False)
          del saved_state

      rng, end_step_rng = jax.random.split(rng)
      if self.end_step is not None: self.end_step(num_train_steps, step, pstate, end_step_rng)
    return state

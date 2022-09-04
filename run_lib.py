from lib2to3 import pytree
import os
from typing import Callable, Any

import jax
import jax.random as jrnd
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import numpy as np
import tensorflow as tf
import logging
import functools
from flax.metrics import tensorboard
# import wandb
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ncsnpp
import losses
import sde_lib
import utils
from models import utils as mutils
import datasets
from absl import flags
from utils import GenModel

FLAGS = flags.FLAGS
PRNGKey = Any

class EMATrainState(train_state.TrainState):
  ema_rate: float
  params_ema: flax.core.FrozenDict[str, Any]
  rng: PRNGKey
  data_scaler: Callable = flax.struct.field(pytree_node=False)
  data_inv_scaler: Callable = flax.struct.field(pytree_node=False)
  model_states: Any = flax.struct.field(pytree_node=False)

  def update_model(self, *, new_model_states, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    new_params_ema = jax.tree_util.tree_map(
        lambda p_ema, p: p_ema * self.ema_rate + p * (1. - self.ema_rate),
        self.params_ema, new_params
      )
    return self.replace(
        step=self.step + 1,
        params=new_params,
        params_ema = new_params_ema,
        opt_state=new_opt_state,
        model_states = new_model_states,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, ema_rate, data_scaler, data_inv_scaler, model_states, rng, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        ema_rate = ema_rate,
        params_ema = params,
        rng = rng,
        data_scaler = data_scaler,
        data_inv_scaler = data_inv_scaler,
        model_states = model_states,
        **kwargs,
    )


def get_dataset(config):
  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              # additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)
  return train_ds, eval_ds

def get_optimizer(config, beta2=0.999):
  if config.optim.optimizer == 'Adam':
    optimizer = optax.adamw(learning_rate=config.optim.lr, b1=config.optim.beta1, b2=beta2, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def get_step_fn(training_config, gen_model, is_training, is_parallel=False, **kwargs):
  reduce_mean = training_config.reduce_mean
  likelihood_weighting = training_config.likelihood_weighting
  importance_weighting = training_config.importance_weighting
  smallest_time = training_config.smallest_time

  loss_fn = losses.get_loss_fn(sde, score_apply_fn, 
                              is_reduce_mean=reduce_mean,
                              likelihood_weighting=likelihood_weighting,
                              importance_weighting=importance_weighting,
                              smallest_time=smallest_time)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """
    rng = carry_state[0]
    state: EMATrainState = carry_state[1]
    rng, step_rng = jax.random.split(rng)
    params = state.params
    model_states = state.model_states
    if is_training:
      (loss, new_model_states), grads = grad_fn(params, model_states, batch, step_rng)
      if is_parallel:
        grad = jax.lax.pmean(grad, axis_name='batch')
      state = state.update_model(new_model_states=new_model_states, grads=grads)
    else:
      loss, _ = loss_fn(params, model_states, batch, step_rng)
    
    return (rng, state), loss
  
  return step_fn

class GenModel:
  def __init__(self, sde, model, continuous, smallest_time, data_scaler, data_inv_scaler) -> None:
    self.continuous = continuous
    self.smallest_time = smallest_time
    self.sde = sde
    self.model = model
    self.data_scaler = data_scaler
    self.data_inv_scaler = data_inv_scaler

  def get_score_fn(self, train=False, return_state=False):
    model_apply_fn = mutils.get_model_fn(self.model, train=train)
    score_apply_fn = mutils.get_score_fn(self.sde, model_apply_fn, continuous = self.continuous, return_state=return_state)
    def scale_score_apply_fn(params, states, images, labels, rng=None):
      images = jax.tree_map(lambda x: self.data_scaler(x), images)
      return score_apply_fn(params, states, images, labels, rng=rng)
    return scale_score_apply_fn
  
  def get_sampling_fn(self, sampler_name, sampling_shape, noise_removal):
    score_apply_fn = self.get_score_fn(train=False, return_state=False)
    wrap_score_apply_fn = lambda s, x, t : score_apply_fn(s.params, s.model_states, x, t)
    
    if sampler_name.lower() == 'ode':
      sampling_fn = sampling.get_ode_sampler(sde=self.sde,
                                  score_apply_fn=wrap_score_apply_fn,
                                  shape=sampling_shape,
                                  inverse_scaler=self.data_inv_scaler,
                                  denoise=noise_removal,
                                  eps=self.smallest_time)
    else:
      raise ValueError(f"Sampler name {sampler_name} unknown.")
    return sampling_fn
  

  def get_step_fn(training_config, gen_model, is_training, is_parallel=False, **kwargs):
    reduce_mean = training_config.reduce_mean
    likelihood_weighting = training_config.likelihood_weighting
    importance_weighting = training_config.importance_weighting
    smallest_time = training_config.smallest_time

    loss_fn = losses.get_loss_fn(sde, score_apply_fn, 
                                is_reduce_mean=reduce_mean,
                                likelihood_weighting=likelihood_weighting,
                                importance_weighting=importance_weighting,
                                smallest_time=smallest_time)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def step_fn(carry_state, batch):
      """Running one step of training or evaluation.

      This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
      for faster execution.

      Args:
        carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
        batch: A mini-batch of training/evaluation data.

      Returns:
        new_carry_state: The updated tuple of `carry_state`.
        loss: The average loss value of this state.
      """
      rng = carry_state[0]
      state: EMATrainState = carry_state[1]
      rng, step_rng = jax.random.split(rng)
      params = state.params
      model_states = state.model_states
      if is_training:
        (loss, new_model_states), grads = grad_fn(params, model_states, batch, step_rng)
        if is_parallel:
          grad = jax.lax.pmean(grad, axis_name='batch')
        state = state.update_model(new_model_states=new_model_states, grads=grads)
      else:
        loss, _ = loss_fn(params, model_states, batch, step_rng)
      
      return (rng, state), loss
    
    return step_fn


class Trainer(object):
  def __init__(self, config, workdir) -> None:
    self.config = config
    self.workdir = workdir
    self.setup_dir()
    self.setup_logger()

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
    pass

  def init_train_state(self, rng):
    config = self.config
    sde = sde_lib.get_sde(config.training.sde, config.model)
    
    # Init model
    rng, model_rng = jrnd.split(rng)
    model, init_model_states, initial_params = mutils.init_model(model_rng, config)
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    score_model = GenModel(sde, model, config.training.continuous, config.sampling.smallest_time, scaler, inverse_scaler)
    score_apply_fn = score_model.get_score_fn(train=True, return_state=True)

    tx = get_optimizer(config)
    state = EMATrainState.create(apply_fn=score_apply_fn, 
                                params=initial_params, 
                                tx=tx,
                                ema_rate=config.model.ema_rate,
                                model_states=init_model_states,
                                data_scaler = scaler, data_inv_scaler = inverse_scaler,
                                rng=rng)

    return score_model, state
  
  def build_step_funcs(self, score_model: GenModel, state):
    config = self.config
    
    train_step = functools.partial(jax.lax.scan, get_step_fn(config.training, score_model.sde, state.apply_fn, is_training=True))
    eval_step  = functools.partial(jax.lax.scan, get_step_fn(config.training, score_model.sde, state.apply_fn, is_training=False))
    train_step = jax.jit(train_step)
    eval_step = jax.jit(eval_step)

    sampling_shape = (config.training.batch_size, config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = score_model.get_sampling_fn(config, sampling_shape)

    return train_step, eval_step, sampling_fn

  def save(self, saved_state, step, preemtion=False):
    if preemtion:
      checkpoints.save_checkpoint(self.checkpoint_meta_dir, saved_state,
                                    step=step // self.config.training.snapshot_freq_for_preemption,
                                    keep=1, overwrite=True)
    else:
      checkpoints.save_checkpoint(self.checkpoint_dir, saved_state,
                                step = step // self.config.training.snapshot_freq,
                                keep=np.inf, overwrite=True)
    
  def load(self):
    pass

  def sample(self):
    pass

  def fit(self, seed, train_ds, eval_ds):
    config = self.config
    train_iter, eval_iter = iter(train_ds), iter(eval_ds)
    rng = jax.random.PRNGKey(seed)

    rng, init_model_rng = jrnd.split(rng)
    score_model, state = self.init_train_state(init_model_rng)

    train_step, eval_step, sampling_fn = self.build_step_funcs(score_model, state)

    initial_step = int(state.step)
    num_train_steps = config.training.n_iters

    for step in range(initial_step, num_train_steps + 1):
      batch = next(train_iter)['image']._numpy()
      rng, next_rng = jax.random.split(rng)
      (_, state), loss = train_step((next_rng, state), batch)
      self.logger.log_loss(loss.mean(), step)

      if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
        saved_state = state.replace(rng=rng)
        self.save(saved_state, step, preemtion=True)
      
      # Report the loss on an evaluation dataset periodically
      if step % config.training.eval_freq == 0:
        eval_batch = jax.tree_map(lambda x: state.data_scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
        rng, next_rng = jax.random.split(rng)
        (_, _), eval_loss = eval_step((next_rng, state), eval_batch['image'])
        
        self.logger.log_eval_loss(eval_loss.mean(), step)
      
      # Save a checkpoint periodically and generate samples if needed
      if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
        # Save the checkpoint.
        saved_state = state.replace(rng=rng)
        self.save(saved_state, step)
        # Generate and save samples
        if config.training.snapshot_sampling:
          rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
          sample_rng = jnp.asarray(sample_rng)
          pstate = flax.jax_utils.replicate(state)
          sample, n = sampling_fn(sample_rng, pstate)
          this_sample_dir = os.path.join(
             self.sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
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


def train(config, workdir):
  train_ds, eval_ds = get_dataset(config)
  trainer = Trainer(config, workdir)
  trainer.fit(config.seed, train_ds, eval_ds)



                
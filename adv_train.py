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
import datasets
from absl import flags
import sampling
import flax.linen as nn
from jax.nn.initializers import normal as normal_init

import logging
# logger = logging.getLogger()
# logging.disable(logging.CRITICAL)

FLAGS = flags.FLAGS
PRNGKey = Any

class EMATrainState(train_state.TrainState):
  ema_rate: float
  params_ema: flax.core.FrozenDict[str, Any]
  rng: PRNGKey
  # data_scaler: Callable = flax.struct.field(pytree_node=False)
  # data_inv_scaler: Callable = flax.struct.field(pytree_node=False)
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
  def create(cls, *, apply_fn, params, tx, ema_rate, model_states, rng, **kwargs):
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
        # data_scaler = data_scaler,
        # data_inv_scaler = data_inv_scaler,
        model_states = model_states,
        **kwargs,
    )

class GenModel:
  def __init__(self, config, continuous, smallest_time) -> None:
    self.config = config
    self.continuous = continuous
    self.smallest_time = smallest_time

    # Create data normalizer and its inverse
    self.data_scaler = datasets.get_data_scaler(config)
    self.data_inv_scaler = datasets.get_data_inverse_scaler(config)

    self.sde = sde_lib.get_sde(config.training.sde, config.model)
    self.model = mutils.get_model(config.model.name)(config=config)
    
  def init_params(self, image_size, num_channels,  rng):
    input_shape = (1, image_size, image_size, num_channels)
    label_shape = input_shape[:1]
    init_input = jnp.zeros(input_shape)
    init_label = jnp.zeros(label_shape, dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    model = self.model
    variables = model.init({'params': params_rng, 'dropout': dropout_rng}, init_input, init_label)
    # Split state and params (which are updated by optimizer).
    init_model_state, initial_params = variables.pop('params')
    del variables # Delete variables to avoid wasting resources
    return init_model_state, initial_params
    
  def get_score_fn(self, train=False, return_state=False):
    model_apply_fn = mutils.get_model_fn(self.model, train=train)
    score_apply_fn = mutils.get_score_fn(self.sde, model_apply_fn, continuous = self.continuous, return_state=return_state)
    def scale_score_apply_fn(params, states, images, labels, rng=None):
      images = jax.tree_map(lambda x: self.data_scaler(x), images)
      return score_apply_fn(params, states, images, labels, rng=rng)
    return scale_score_apply_fn
  
  def get_sampling_fn(self, image_size, num_channels, sampler_name):
    sampling_shape = (1, image_size, image_size, num_channels)
    score_apply_fn = self.get_score_fn(train=False, return_state=False)
    wrap_score_apply_fn = lambda s, x, t : score_apply_fn(s.params, s.model_states, x, t)
    
    if sampler_name.lower() == 'ode':
      sampling_fn = sampling.get_ode_sampler(sde=self.sde,
                                  score_apply_fn=wrap_score_apply_fn,
                                  shape=sampling_shape,
                                  inverse_scaler=self.data_inv_scaler,
                                  denoise=self.config.sampling.noise_removal,
                                  eps=self.smallest_time)
    else:
      raise ValueError(f"Sampler name {sampler_name} unknown.")
    return sampling_fn
  

  def get_step_fn(self, is_training, is_parallel=False, **kwargs):
    reduce_mean = self.config.training.reduce_mean
    likelihood_weighting = self.config.training.likelihood_weighting
    importance_weighting = self.config.training.importance_weighting
    smallest_time = self.config.training.smallest_time

    loss_fn = losses.get_loss_fn(self.sde, self.get_score_fn(train=is_training, return_state=True), 
                                 is_reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting, 
                                 importance_weighting=importance_weighting, smallest_time=smallest_time)
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

class Discriminator(nn.Module):
  features: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True):
    conv = functools.partial(nn.Conv, kernel_size=[4, 4], strides=[2, 2], padding='VALID',
                   kernel_init=normal_init(0.02), dtype=self.dtype)
    batch_norm = functools.partial(nn.BatchNorm, use_running_average=not train, axis=-1,
                         scale_init=normal_init(0.02), dtype=self.dtype)
        
    x = conv(self.features)(x)
    x = batch_norm()(x)
    x = nn.leaky_relu(x, 0.2)
    x = conv(self.features*2)(x)
    x = batch_norm()(x)
    x = nn.leaky_relu(x, 0.2)
    x = conv(1)(x)
    x = x.reshape((args['batch_size_p'], -1))
    return x
  
  def get_train_step(self):
    def loss_fn(params, generated_data: jnp.ndarray, real_data: jnp.ndarray):
      logits_real, mutables = discriminator_state.apply_fn(
          {'params': params, 'batch_stats': discriminator_state.batch_stats},
          real_data, mutable=['batch_stats'])
          
      logits_generated, mutables = discriminator_state.apply_fn(
          {'params': params, 'batch_stats': mutables['batch_stats']},
          generated_data, mutable=['batch_stats'])
      
      real_loss = optax.sigmoid_binary_cross_entropy(
          logits_real, args['true_label']).mean()
      generated_loss = optax.sigmoid_binary_cross_entropy(
          logits_generated, args['false_label']).mean()
      
      loss = (real_loss + generated_loss) / 2

      return loss, mutables

    # Critique real and generated data with the Discriminator.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mutables), grads = grad_fn(discriminator_state.params)

    # Average cross the devices.
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    # Update the discriminator through gradient descent.
    new_discriminator_state = discriminator_state.apply_gradients(
        grads=grads, batch_stats=mutables['batch_stats'])
    
    return new_discriminator_state, loss

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


def train(config, workdir):
  train_ds, eval_ds = get_dataset(config)
  gen_model = GenModel(config, config.training.continuous, config.training.smallest_time)
  tx = get_optimizer(config)
  
  rng = jax.random.PRNGKey(config.seed)
  rng, init_model_rng = jax.random.split(rng)

  init_model_states, initial_params = gen_model.init_params(config.data.image_size, config.data.num_channels, init_model_rng)
  state = EMATrainState.create(apply_fn=gen_model.get_score_fn(train=True, return_state=True), 
                              params=initial_params, 
                              tx=tx,
                              ema_rate=config.model.ema_rate,
                              model_states=init_model_states,
                              rng=rng)
  del init_model_states, initial_params
  
  train_step = functools.partial(jax.lax.scan, gen_model.get_step_fn(is_training=True))
  eval_step  = functools.partial(jax.lax.scan, gen_model.get_step_fn(is_training=False))
  train_step = jax.jit(train_step)
  eval_step = jax.jit(eval_step)

  # if config.training.snapshot_sampling:
  sampling_fn = gen_model.get_sampling_fn(config.data.image_size, config.data.num_channels, config.sampling.method)
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)
  def end_step_fn(num_train_steps, step_idx, pstate, rng):
    if step_idx != 0 and step_idx % config.training.snapshot_freq == 0 or step_idx == num_train_steps and config.training.snapshot_sampling:
      rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
      sample_rng = jnp.asarray(sample_rng)
      # pstate = flax.jax_utils.replicate(state)
      sample, n = sampling_fn(sample_rng, pstate)
      this_sample_dir = os.path.join(
          sample_dir, "iter_{}_host_{}".format(step_idx, jax.host_id()))
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


  trainer = Trainer(config, workdir, train_step, eval_step, end_step=end_step_fn)
  state = trainer.fit(rng, state, train_ds, eval_ds)



                
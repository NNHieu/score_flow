from typing import Any, Callable

import jax
import jax.numpy as jnp
from matplotlib import image
# import wandb
# Keep the import below for registering all model definitions
import losses
import sde_lib
from models import utils as mutils
import src.datasets as datasets
import sampling

PRNGKey = Any

class GenModel:
  def __init__(self, sde, score_model, continuous, image_size, num_channels) -> None:
    self.continuous = continuous
    self.score_model = score_model
    self.sde = sde
    # self.image_size = image_size
    # self.num_channels = num_channels
    self.input_shape = (1, image_size, image_size, num_channels)
    
  def init_params(self, rng):
    label_shape = self.input_shape[:1]
    init_input = jnp.zeros(self.input_shape)
    init_label = jnp.zeros(label_shape, dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    variables = self.score_model.init({'params': params_rng, 'dropout': dropout_rng}, init_input, init_label)
    # Split state and params (which are updated by optimizer).
    init_model_state, initial_params = variables.pop('params')
    del variables # Delete variables to avoid wasting resources
    return init_model_state, initial_params
    
  def get_score_fn(self, train=False, return_state=False):
    model_apply_fn = mutils.get_model_fn(self.score_model, train=train)
    score_apply_fn = mutils.get_score_fn(self.sde, model_apply_fn, continuous = self.continuous, return_state=return_state)
    return score_apply_fn
  
  def get_sampling_fn(self, method, noise_removal: bool, data_inv_scaler: Callable):
    score_apply_fn = self.get_score_fn(train=False, return_state=False)
    wrap_score_apply_fn = lambda s, x, t : score_apply_fn(s.params, s.model_states, x, t)
    
    if method.lower() == 'ode':
      sampling_fn = sampling.get_ode_sampler(sde=self.sde,
                                  score_apply_fn=wrap_score_apply_fn,
                                  shape=self.input_shape,
                                  inverse_scaler=data_inv_scaler,
                                  denoise=noise_removal,
                                  eps=self.sde.smallest_time)
    else:
      raise ValueError(f"Sampler name {method} unknown.")
    return sampling_fn

  def get_step_fn(self, update_params_fn, loss_config, is_training, is_parallel=False, **kwargs):
    reduce_mean = loss_config.reduce_mean
    likelihood_weighting = loss_config.likelihood_weighting
    importance_weighting = loss_config.importance_weighting
    smallest_time = self.sde.smallest_time
    score_fn = self.get_score_fn(train=is_training, return_state=True)
    loss_fn = losses.get_loss_fn(self.sde, score_fn, 
                                 is_reduce_mean=reduce_mean, 
                                 likelihood_weighting=likelihood_weighting, 
                                 importance_weighting=importance_weighting, 
                                 smallest_time=smallest_time)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def step_fn(rng, params, model_state, opt_state, batch):
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
      rng, step_rng = jax.random.split(rng)
      if is_training:
        (loss, model_state), grads = grad_fn(params, model_state, batch, step_rng)
        if is_parallel:
          grads = jax.lax.pmean(grads, axis_name='batch')
        params, opt_state = update_params_fn(params, opt_state, grads)
      else:
        loss, _ = loss_fn(params, model_state, batch, step_rng)
      return rng, params, model_state, opt_state, loss
    return step_fn
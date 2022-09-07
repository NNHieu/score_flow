from typing import Any

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
  
  def get_sampling_fn(self, sampler_config):
    score_apply_fn = self.get_score_fn(train=False, return_state=False)
    wrap_score_apply_fn = lambda s, x, t : score_apply_fn(s.params, s.model_states, x, t)
    
    if sampler_config.method.lower() == 'ode':
      sampling_fn = sampling.get_ode_sampler(sde=self.sde,
                                  score_apply_fn=wrap_score_apply_fn,
                                  shape=self.input_shape,
                                  inverse_scaler=self.data_inv_scaler,
                                  denoise=sampler_config.noise_removal,
                                  eps=self.smallest_time)
    else:
      raise ValueError(f"Sampler name {sampler_config.method} unknown.")
    return sampling_fn

  def get_step_fn(self, loss_config, is_training, is_parallel=False, **kwargs):
    reduce_mean = loss_config.reduce_mean
    likelihood_weighting = loss_config.likelihood_weighting
    importance_weighting = loss_config.importance_weighting
    smallest_time = self.sde.smallest_time

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
      state = carry_state[1]
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
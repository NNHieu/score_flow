import functools
from typing import Callable
import jax.numpy as jnp
import jax.random as jrnd
import models.utils as mutils
from utils import batch_mul
from icecream import ic

def get_sde_loss_fn(sde, score_apply_fn, is_reduce_mean=True, likelihood_weighting=True,
                    importance_weighting=True, eps=1e-5):
  reduce_op = jnp.mean if is_reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
  def importance_weight_loss(data, t, z, score, std):
    losses = jnp.square(batch_mul(score, std) + z)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    return losses
  
  def uniform_weight_loss(data, t, z, score, std):
    g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
    losses = jnp.square(score + batch_mul(z, 1. / std))
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2
    return losses
  
  weighting_loss = importance_weight_loss
  if likelihood_weighting and not importance_weighting:
    weighting_loss = uniform_weight_loss 

  if likelihood_weighting and importance_weighting:
    time_sampler = functools.partial(sde.sample_importance_weighted_time_for_likelihood, eps=eps)
  else:
    time_sampler = functools.partial(jrnd.uniform, minval=eps, maxval=sde.T)

  def loss_fn(params, states, data, rng):
    # Sampling timestep
    rng, step_rng = jrnd.split(rng)
    t = time_sampler(step_rng, (data.shape[0],))
    
    # Generate perturbed data
    rng, step_rng = jrnd.split(rng)
    z = jrnd.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    
    # Calculate scores at perturbed data and sampled time step
    rng, step_rng = jrnd.split(rng)
    score, new_model_state = score_apply_fn(params, states, perturbed_data, t, rng=step_rng)

    # Weighting losses
    losses =  weighting_loss(data, t, z, score, std)
    loss = jnp.mean(losses)
    return loss, new_model_state
  
  return loss_fn
  
def get_loss_fn(sde, score_apply_fn, is_reduce_mean=False, continuous=True, likelihood_weighting=False,
                importance_weighting=False, smallest_time=1e-5):
  if continuous:
    return get_sde_loss_fn(sde, score_apply_fn, is_reduce_mean=is_reduce_mean,
                              likelihood_weighting=likelihood_weighting,
                              importance_weighting=importance_weighting,
                              eps=smallest_time)
  else:
    # assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    # if isinstance(sde, VESDE):
    #   loss_fn = get_smld_loss_fn(sde, model, is_training, reduce_mean=is_reduce_mean)
    # elif isinstance(sde, VPSDE):
    #   loss_fn = get_ddpm_loss_fn(sde, model, is_training, reduce_mean=is_reduce_mean)
    # else:
    raise ValueError(f"Discrete training for {sde.__class__.__name__} is not implimented.")


import jax
import jax.numpy as jnp
import flax
# import wandb
# from src.trainer.checkpoint import CheckpointCallback
# Keep the import below for registering all model definitions
from icecream import ic
ic.configureOutput(includeContext=True)


class TrainStateRunner():
  def __init__(self, trainer, trainstate, train_update_fn, eval_fn, is_multi_gpu=False) -> None:
    self.is_multi_gpu = is_multi_gpu
    self.trainer = trainer
    self.rng = jax.random.fold_in(trainstate.rng, jax.process_index())
    self._state = trainstate
    self._sync_state = True
    if is_multi_gpu:
      self.train_update_fn = self._parallel_wrapper_step_fn(train_update_fn)
      self._eval_fn = self._parallel_wrapper_step_fn(eval_fn)
      self._trainstate = flax.jax_utils.replicate(self._state)
    else:
      self.train_update_fn = self._non_parallel_wrapper_train_step(train_update_fn)
      self._eval_fn = self._non_parallel_wrapper_train_step(eval_fn)
      self._trainstate = self._state

  @classmethod
  def _parallel_wrapper_step_fn(cls, step_fn):
    def wrap_fn(pstate, batch, rng):
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, pstate), loss = step_fn((next_rng, pstate), batch)
      loss = flax.jax_utils.unreplicate(loss)
      return pstate, loss, rng
    return wrap_fn
  
  @classmethod
  def _non_parallel_wrapper_train_step(cls, step_fn):
    def wrap_fn(state, batch, rng):
      rng, next_rng = jax.random.split(rng)
      (_, state), loss = step_fn((next_rng, state), batch)
      return state, loss, rng
    return wrap_fn

  def train_step(self, batch):
    _trainstate, loss, self.rng = self.train_update_fn(self._trainstate, batch, self.rng)
    # _trainstate = _trainstate.replace(rng=rng)
    self._update_trainstate(_trainstate)
    return loss, None

  def eval_step(self, batch):
    _, loss, self.rng = self._eval_fn(self._trainstate, batch, self.rng)
    return loss, None

  def get_pstate(self):
    return self._trainstate
  
  def _update_trainstate(self, trainstate):
    self._trainstate = trainstate
    self._sync_state = False

  def get_state(self):
    if not self._sync_state:
      if self.is_multi_gpu:
        self._state = flax.jax_utils.unreplicate(self._trainstate).replace(rng=self.rng)
      else:
        self._state = self._trainstate.replace(rng=self.rng)
      self._sync_state = True
    return self._state

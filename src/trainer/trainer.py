from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
# import wandb
from .runner import TrainStateRunner
# from src.trainer.checkpoint import CheckpointCallback
# Keep the import below for registering all model definitions
import src.datasets as datasets
from src import utils as sutils
from . import states

log = sutils.get_pylogger(__name__)

from icecream import ic
ic.configureOutput(includeContext=True)

PRNGKey = Any

# @functools.partial(jax.jit,static_argnums=(0, ), donate_argnums=(1, 2))
def update_params_fn(tx, params, opt_state, grads):
  updates, opt_state = tx.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state


class CustomTrainState(train_state.TrainState):
  rng: PRNGKey
  # model_states: Any = flax.struct.field(pytree_node=False)
  model_states: Any

  def update_model(self, *, new_model_states, grads, **kwargs):
    # updates, new_opt_state = self.tx.update(
    #     grads, self.opt_state, self.params)
    # new_params = optax.apply_updates(self.params, updates)
    params, opt_state = update_params_fn(self.tx, self.params, self.opt_state, grads)
    return self.replace(
        step=self.step + 1,
        params=params,
        opt_state=opt_state,
        model_states = new_model_states,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, model_states, rng, **kwargs):
    opt_state = tx.init(params)
    return cls(
      step=jnp.zeros((1,)),
      apply_fn=apply_fn,
      params=params,
      tx=tx,
      opt_state=opt_state,
      rng = rng,
      model_states = model_states,
      **kwargs,
    )

class Trainer(object):
  def __init__(self, 
              train_step_fn: Callable, 
              eval_step_fn: Callable, 
              n_iters: int,
              eval_freq: int,
              log_freq: int,
              is_multigpu: bool =False,
              checkpoint_cb: "CheckpointCallback" = None,
              callbacks: Optional[Sequence["Callback"]] = None,
              restore_checkpoint_after_setup = False) -> None:
    # self.setup_dir()
    # self.setup_logger()
    self.is_multi_gpu = is_multigpu
    self.num_train_steps = n_iters
    self.eval_freq = eval_freq
    self.restore_checkpoint_after_setup = restore_checkpoint_after_setup
    self.log_freq = log_freq
    
    self._train_step = train_step_fn
    self._eval_step = eval_step_fn
    
    self.checkpoint_cb = checkpoint_cb
    self.callbacks = callbacks
    if callbacks is None:
      self.callbacks = []
    
    self.state = states.TrainerState()

  # def setup_dir(self):
  #   self.checkpoint_meta_dir = os.path.join(self.workdir, "checkpoints_meta")
  #   self.checkpoint_dir = os.path.join(self.workdir, "checkpoints")
  #   self.tb_dir = os.path.join(self.workdir, "tensorboard")
  #   tf.io.gfile.makedirs(self.tb_dir)
    
  def log(self, name, val, step_idx):
    log.info(f"Training - step {step_idx} | {name}: {val}")

  def setup_logger(self):
    # self.logger = utils.SimpleLogger(self.tb_dir)
    pass

  def create_saved_state(self, pstate, rng):
    if jax.process_index() == 0:
      if self.is_multi_gpu:
        saved_state = flax.jax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng)
      return saved_state
  
  def _get_train_runner(self):
    assert self.state.fn == states.TrainerFn.FITTING and self.state.status == states.TrainerStatus.RUNNING
    return self.train_runner


  def load(self, state):
    return self.checkpoint_cb.restore(state)

  def fit(self, state, datamodule: datasets.DataModule):
    self._call_and_handle_interrupt(self._fit, state, datamodule)

  def _fit(self, trainstate, datamodule: datasets.DataModule):
    self.state.fn = states.TrainerFn.FITTING
    self.state.status = states.TrainerStatus.RUNNING
    
    # ----------------------------
    # SET UP TRAINING
    # ----------------------------
    n_jitted_steps = datamodule.additional_dim
    if n_jitted_steps is None: n_jitted_steps = 1
    train_ds, eval_ds = datamodule.train_ds(), datamodule.test_ds()
    train_iter, eval_iter = iter(train_ds), iter(eval_ds)

    # ----------------------------
    # TRAIN
    # ----------------------------
    # hook
    # self._call_callback_hooks("on_fit_start", self)
    # self._log_hyperparams()
    if self.restore_checkpoint_after_setup:
      log.debug(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {self.checkpoint_cb.checkpoint_meta_dir}")
      # self._restore_modules_and_callbacks(ckpt_path)
      trainstate = self.load(trainstate)
    
    initial_step_idx = int(trainstate.step)
    num_train_steps = self.num_train_steps

    self.train_runner = TrainStateRunner(self, trainstate, self._train_step, self._eval_step, self.is_multi_gpu)
    del trainstate

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    log.info("Starting training loop at step %d." % (initial_step_idx,))
    for step in range(initial_step_idx, num_train_steps + 1, n_jitted_steps):
      self._step_idx = step
      batch = next(train_iter)['image']._numpy()
      # # TODO: consider move this scale step to preprocess 
      batch = jax.tree_map(lambda x: datamodule.scaler(x), batch)
      # ic(batch.shape)
      loss, outputs = self.train_runner.train_step(batch)
      if step % self.log_freq == 0:
        self.log('trai_loss', loss.mean(), step)      
      self._call_callback_hooks("on_train_batch_end", self.train_runner, outputs, batch, step)
      del batch

      # Report the loss on an evaluation dataset periodically
      if step % self.eval_freq == 0:
        eval_batch = next(eval_iter)['image']._numpy()  # pylint: disable=protected-access
        # TODO: consider move this scale step to preprocess 
        eval_batch = jax.tree_map(lambda x: datamodule.scaler(x), eval_batch)
        eval_loss, eval_outputs  = self.train_runner.eval_step(eval_batch)
        self.log("eval_loss", eval_loss.mean(), step)
        self._call_callback_hooks("on_validation_batch_end", self.train_runner, eval_outputs, eval_batch, step, dataloader_idx=0)
        del eval_batch
    
    # hook
    # self._call_callback_hooks("on_fit_end", self)
    self.train_runner = None
    self.state.fn = states.TrainerFn.FITTING
    self.state.status = states.TrainerStatus.FINISHED
    return self.train_runner.get_state()
  
  @property
  def global_step(self):
    if self.state.fn == states.TrainerFn.FITTING and \
       self.state.status == states.TrainerStatus.RUNNING:
      return self._step_idx
    return -1
    
  def _teardown(self):
    """This is the Trainer's internal teardown, unrelated to the `teardown` hooks in LightningModule and
        Callback; those are handled by :meth:`_call_teardown_hook`."""
    # self._logger_connector.teardown()
    # self._signal_connector.teardown()
    pass

  def _call_callback_hooks(
    self,
    hook_name: str,
    *args: Any,
    **kwargs: Any,
  ) -> None:
    log.debug(f"{self.__class__.__name__}: calling callback hook: {hook_name}")

    if hook_name in ('on_train_batch_end', ):
      checkpoint_fn = getattr(self.checkpoint_cb, hook_name)
      if callable(checkpoint_fn):
        checkpoint_fn(self)
    for callback in self.callbacks:
      fn = getattr(callback, hook_name)
      if callable(fn):
        fn(self, *args, **kwargs)
    
  def _call_callbacks_on_save_checkpoint(self, pstate, state):
    log.debug(f"{self.__class__.__name__}: calling on_save_checkpoint callback hook")
    for callback in self.callbacks:
      fn = getattr(callback, "on_save_checkpoint")
      if callable(fn):
        fn(self, pstate, state)


  def _call_and_handle_interrupt(self, trainer_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    r"""
    Error handling, intended to be used only for main trainer function entry points (fit, validate, test, predict)
    as all errors should funnel through them
    Args:
        trainer_fn: one of (fit, validate, test, predict)
        *args: positional arguments to be passed to the `trainer_fn`
        **kwargs: keyword arguments to be passed to `trainer_fn`
    """
    try:
      return trainer_fn(*args, **kwargs)
    except KeyboardInterrupt as exception:
      log.warn("Detected KeyboardInterrupt, attempting graceful shutdown...")
      # user could press Ctrl+c many times... only shutdown once
      self.state.status = states.TrainerStatus.INTERRUPTED
      self._call_callback_hooks("on_exception", exception)
    except BaseException as exception:
      self.state.status = states.TrainerStatus.INTERRUPTED
      self._call_callback_hooks("on_exception", exception)
      self._teardown()
      # teardown might access the stage so we reset it after
      self.state.stage = None
      raise


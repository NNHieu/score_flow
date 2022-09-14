import functools
import os
import numpy as np
import jax
import jax.numpy as jnp
import flax
import tensorflow as tf
# import wandb
# from src.trainer.checkpoint import CheckpointCallback
# Keep the import below for registering all model definitions
from icecream import ic

from utils.run.pylogger import get_pylogger
ic.configureOutput(includeContext=True)

log = get_pylogger(__name__)

class Strategy():
  def __init__(self, multi_jitted_step=None, multi_gpu=None, jit_step_fn=True) -> None:
    self._multi_gpu = multi_gpu
    self._multi_jitted_step = multi_jitted_step
    self._jit_step_fn = jit_step_fn
  
  @property
  def is_multi_gpu(self):
    return self._multi_gpu is not None
    
  @property
  def n_jitted_step(self):
    return self._multi_jitted_step if self._multi_jitted_step is not None else 1
  
  def prepair_ds(self, dataset: tf.data.Dataset, prefetch_buffer_sz=tf.data.AUTOTUNE):
    if self._multi_jitted_step is not None:
      dataset = dataset.batch(self._multi_jitted_step)
    if self.is_multi_gpu:
      dataset = dataset.batch(self._multi_gpu)
    return dataset.prefetch(prefetch_buffer_sz)

  def prepair_trainstate(self, state):
    if self.is_multi_gpu:
      state = flax.jax_utils.replicate(state)
    return state
  
  def prepair_step_fn(self, step_fn, donate_argnums=()):
    if self._multi_jitted_step is not None:
      log.info(f"Wrap scan functions")
      step_fn = functools.partial(jax.lax.scan, step_fn)
    if self._jit_step_fn:
      if self.is_multi_gpu:
        log.info(f"Wrap pmap step functions")
        step_fn = jax.pmap(step_fn, axis_name='batch', donate_argnums=donate_argnums)
      else:
        log.info(f"Wrap jit step functions")
        step_fn = jax.jit(step_fn, donate_argnums=donate_argnums)
    return step_fn
  
  def get_rng_split_fn(self):
    if self.is_multi_gpu:
      def rng_split(rng):
        rng, *next_rng = jax.random.split(rng, num=self._multi_gpu + 1)
        next_rng = jnp.asarray(next_rng)
        return rng, next_rng
    else:
      rng_split = lambda rng: jax.random.split(rng)
    return rng_split

  def get_state(self, state):
    if self.is_multi_gpu:
      return flax.jax_utils.unreplicate(state)
    return state

class EvalLiklihoodRunner:
  def __init__(self, 
              likelihood_fn, 
              trainstate, 
              num_samples, 
              batch_size, 
              is_multi_gpu, 
              prefix, 
              p_dequantizer=None,
              scaler = None,
              inverse_scaler=None) -> None:
    num_sampling_rounds = num_samples // batch_size + 1
    self._state = trainstate
    if is_multi_gpu:
      self._trainstate = flax.jax_utils.replicate(self._state)
    else:
      self._trainstate = self._state
    
    self.likelihood_fn = likelihood_fn
    self.prefix = prefix
    self.p_dequantizer = p_dequantizer
    self.inverse_scaler = inverse_scaler
    self.scaler = scaler
    self.dequantizer = self.p_dequantizer is not None

  def get_eval_bpd_step(self, batch):
    def wraped_fn(state, batch, rng):
      data = batch['image']
      if self.dequantizer:
        rng, step_rng = jax.random.split(rng)
        u = jax.random.normal(step_rng, data.shape)
        noise, logpd = self.p_dequantizer(u, self.inverse_scaler(data))
        data = self.scaler((self.inverse_scaler(data) * 255. + noise) / 256.)
        bpd_d = -logpd / np.log(2.)
        dim = np.prod(noise.shape[2:])
        bpd_d = bpd_d / dim
      rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
      step_rng = jnp.asarray(step_rng)
      bpd = self.likelihood_fn(step_rng, self._trainstate, batch)[0]
      return bpd
    
  def eval_bpd(self, begin_bpd_round, ds_bpd, bpd_num_repeats):
    bpds = []
    begin_repeat_id = begin_bpd_round // len(ds_bpd)
    begin_batch_id = begin_bpd_round % len(ds_bpd)
    # Repeat multiple times to reduce variance when needed
    for repeat in range(begin_repeat_id, bpd_num_repeats):
      bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
      for _ in range(begin_batch_id):
        next(bpd_iter)
      for batch_id in range(begin_batch_id, len(ds_bpd)):
        bpd_round_id = batch_id + len(ds_bpd) * repeat
        if tf.io.gfile.exists(f"{self.prefix}_bpd_{bpd_round_id}.npz"):
          continue
        batch = next(bpd_iter)
        eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
        

        rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        bpd = likelihood_fn(step_rng, pstate, data)[0]
        if self.dequantizer:
          bpd = bpd + bpd_d
        bpd = bpd.reshape(-1)
        bpds.extend(bpd)
        logging.info(
          "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, jnp.mean(jnp.asarray(bpds))))
        # Save bits/dim to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(eval_dir,
                                            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, bpd)
          fout.write(io_buffer.getvalue())

        eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
        # Save intermediate states to resume evaluation after pre-emption
        checkpoints.save_checkpoint(
          eval_dir,
          eval_meta,
          step=ckpt * (num_sampling_rounds + num_bpd_rounds) + bpd_round_id,
          keep=1,
          prefix=f"meta_{jax.host_id()}_", overwrite=True)
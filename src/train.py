import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import Optional, Any

import os
import functools

import hydra
from omegaconf import DictConfig, OmegaConf

import jax
import optax

from src import datasets
# Keep the import below for registering all model definitions
from models import ncsnpp
import models.utils as mutils
from src import sde_lib
from src.model import GenModel
from src.trainer.callbacks import SamplingCallback
from src.trainer.checkpoint import CheckpointCallback
from src.trainer.trainer import Trainer, CustomTrainState

from icecream import ic
ic.configureOutput(includeContext=True)

from src import utils
log = utils.get_pylogger(__name__)
from utils.run.utils import task_wrapper

PRNGKey = Any

@task_wrapper
def train(cfg: OmegaConf, workdir):
  main_cfg = cfg.main
  # Init lightning datamodule
  log.info(f"Current working directory : {workdir}")
  log.info(f"Random seed : {main_cfg.training.seed}")
  rng = jax.random.PRNGKey(main_cfg.training.seed)
  parallel_training = main_cfg.training.get("parallel", False)
  assert not parallel_training or (parallel_training and main_cfg.training.get('jit', False))

  n_jitted_steps = main_cfg.training.get('n_jitted_steps', None) if parallel_training else None
  if n_jitted_steps is None or n_jitted_steps <= 1: 
    n_jitted_steps = None
  else:
    # JIT multiple training steps together for faster training
    # Must be divisible by the number of steps jitted together
    assert main_cfg.training.log_freq % n_jitted_steps == 0 and \
          main_cfg.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
          main_cfg.training.eval_freq % n_jitted_steps == 0 and \
          main_cfg.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  # Init data module
  log.info(f"Instantiating datamodule <{main_cfg.dataset._target_}>")
  multi_gpu = jax.local_device_count() if parallel_training else None
  datamodule: datasets.DataModule = hydra.utils.instantiate(main_cfg.dataset, 
                                      additional_dim=n_jitted_steps,
                                      multi_gpu = multi_gpu)

  # Init model
  log.info(f"Instantiating model <{main_cfg.model.name}>")
  score_model = mutils.get_model(main_cfg.model.name)(config=main_cfg)
  sde: sde_lib.SDE = sde_lib.get_sde(main_cfg.sde)
  gen_model = GenModel(sde, score_model, main_cfg.sde.continuous, main_cfg.dataset.image_size, main_cfg.dataset.num_channels)
  
  # Init optimizer
  log.info(f"Instantiating optimizer")
  tx = hydra.utils.instantiate(main_cfg.training.optim)

  log.info(f"Instantiating model parameters and states")
  rng, init_model_rng = jax.random.split(rng)
  init_model_states, initial_params = gen_model.init_params(init_model_rng)
  
  log.info(f"Instantiating train state")
  rng, trainstate_rng = jax.random.split(rng)
  state = CustomTrainState.create(apply_fn=gen_model.get_score_fn(train=True, return_state=True), 
                                  params=initial_params, 
                                  model_states=init_model_states,
                                  tx=tx,
                                  rng=trainstate_rng)
  del init_model_states, initial_params

  log.info(f"Get train and eval step")
  train_step = gen_model.get_step_fn(main_cfg.loss, is_training=True, is_parallel=parallel_training)
  eval_step  = gen_model.get_step_fn(main_cfg.loss, is_training=False, is_parallel=parallel_training)
  
  if n_jitted_steps is not None:
    log.info(f"Wrap scan functions")
    train_step = functools.partial(jax.lax.scan, train_step)
    eval_step  = functools.partial(jax.lax.scan, eval_step)

  # def unroll_wrapper(step_fn):
  #   @functools.wraps(step_fn)
  #   def wrapped_fn(rng, pstate, batch):
  #     (_, pstate), loss = step_fn((rng, pstate), batch)
  #     return pstate, loss
    
  #   return wrapped_fn
  # train_step = unroll_wrapper(train_step)
  # eval_step = unroll_wrapper(eval_step)

  if parallel_training:
    log.info(f"Wrap pmap step functions")
    train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))
    eval_step  = jax.pmap(eval_step, axis_name='batch')
  elif main_cfg.training.get('jit'):
    log.info(f"Wrap jit step functions")
    train_step = jax.jit(train_step, donate_argnums=(0,))
    eval_step = jax.jit(eval_step)
  
  log.info(f"Setup checkpoint callback")
  checkpoint_cb = CheckpointCallback(workdir, 
                                     main_cfg.training.snapshot_freq, 
                                     main_cfg.training.snapshot_freq_for_preemption)

  log.info(f"Setup sampling callback")
  sampling_fn = gen_model.get_sampling_fn(main_cfg.sde.sampling.method, 
                                          main_cfg.sde.sampling.noise_removal, 
                                          datamodule.inv_scaler,
                                          datamodule.per_device_batch_size)
  sample_dir = os.path.join(workdir, "samples")
  rng, sampling_rng = jax.random.split(rng)
  sampling_rng = jax.random.fold_in(sampling_rng, jax.process_index())
  
  callbacks = [
    SamplingCallback(sampling_fn, sample_dir, sampling_rng)
  ]
  trainer = Trainer(train_step, eval_step, 
                    main_cfg.training.n_iters,
                    main_cfg.training.eval_freq,
                    main_cfg.training.log_freq,
                    is_multigpu=parallel_training,
                    checkpoint_cb=checkpoint_cb, 
                    callbacks=callbacks,
                    restore_checkpoint_after_setup=True)
  # ic.disable()
  state = trainer.fit(state, datamodule)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(config: DictConfig) -> Optional[float]:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    return train(config)

if __name__ == "__main__":
    main()
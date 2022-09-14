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
import jax.numpy as jnp
import flax
import optax

from src import datasets
# Keep the import below for registering all model definitions
from models import ncsnpp
import models.utils as mutils
from src import sde_lib
from src.model import GenModel
from src.trainer.callbacks import SamplingCallback, SaveBpd
from src.trainer.checkpoint import CheckpointCallback
from src.trainer.trainer import Trainer, CustomTrainState
import bound_likelihood
import likelihood

from icecream import ic
ic.configureOutput(includeContext=True)

from src import utils
log = utils.get_pylogger(__name__)
from utils.run.utils import task_wrapper

PRNGKey = Any

# A data class for storing intermediate results to resume evaluation after pre-emption
@flax.struct.dataclass
class EvalMeta:
  ckpt_id: int
  sampling_round_id: int
  bpd_round_id: int
  rng: Any

def prepair_eval_bpd(main_cfg, multi_gpu, gen_model: GenModel):
  config_likelihood = main_cfg.eval.likelihood
  if main_cfg.eval.data.dequantizer:
    raise NotImplemented

  datamodule: datasets.DataModule = hydra.utils.instantiate(main_cfg.dataset, 
                                        additional_dim=None,
                                        uniform_dequantization = not main_cfg.eval.data.dequantizer,
                                        multi_gpu=multi_gpu)
  if main_cfg.eval.data.split.lower() == "train":
    ds_bpd = datamodule.eval_train_ds()
    bpd_num_repeats = 1
  elif main_cfg.eval.data.split.lower() == "test":
    ds_bpd = datamodule.eval_test_ds()
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {main_cfg.eval.data.split} recognized.")


  # Add one additional round to get the exact number of samples as required.
  # rng = jax.random.fold_in(rng, jax.host_id())
  # num_sampling_rounds = main_cfg.num_samples // main_cfg.dataset.batch_size + 1
  num_bpd_rounds = len(ds_bpd) * bpd_num_repeats
  step_fn = gen_model.get_likelihood_fn(main_cfg.eval.data.dequantizer, 
                                        config_likelihood.bound, 
                                        datamodule.inv_scaler, 
                                        config_likelihood.dsm, 
                                        config_likelihood.offset)

  return datamodule, ds_bpd, step_fn, num_bpd_rounds, bpd_num_repeats

@task_wrapper
def eval(cfg: OmegaConf, workdir):
  main_cfg = cfg.main
  # Init lightning datamodule
  log.info(f"Current working directory : {workdir}")
  log.info(f"Random seed : {main_cfg.training.seed}")
  rng = jax.random.PRNGKey(main_cfg.training.seed)
  parallel_training = main_cfg.training.get("parallel", False)
  assert not parallel_training or (parallel_training and main_cfg.training.get('jit', False))
  multi_gpu = jax.local_device_count() if parallel_training else None

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
  
  log.info(f"Setup checkpoint callback")
  checkpoint_cb = CheckpointCallback(workdir, 
                                     main_cfg.training.snapshot_freq, 
                                     main_cfg.training.snapshot_freq_for_preemption)
  state = checkpoint_cb.restore(state)

  # if main_cfg.likelihood.enable:
  log.info(f"Get eval bpd step")
  dm, ds_bpd, step_fn, num_bpd_rounds, bpd_num_repeats = prepair_eval_bpd(main_cfg, multi_gpu, gen_model)
  bpd_num_repeats = 1
  for i in range(bpd_num_repeats):
    eval_dir = os.path.join(workdir, f"bpds/{main_cfg.eval.data.split}_{i}")
    callbacks = [
      SaveBpd(eval_dir, f"{main_cfg.dataset.ds_name}_ckpt_last")
    ]
    trainer = Trainer(None, step_fn, 
                      main_cfg.training.n_iters,
                      main_cfg.training.eval_freq,
                      main_cfg.training.log_freq,
                      multigpu=multi_gpu,
                      jit_step_fn=False,
                      callbacks=callbacks,
                      restore_checkpoint_after_setup=True)
    # ic.disable()
    trainer.eval(state, ds_bpd, dm.scaler)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(config: DictConfig) -> Optional[float]:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    return eval(config)

if __name__ == "__main__":
    main()
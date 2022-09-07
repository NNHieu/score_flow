import functools
import os
from typing import Any
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src import utils
from src import datasets

# Keep the import below for registering all model definitions
from models import ncsnpp
import models.utils as mutils
from src import sde_lib
from src.model import GenModel
from src.trainer import Trainer, CustomTrainState
import jax
import flax
import optax
from flax.training import train_state

log = utils.get_logger(__name__)
PRNGKey = Any

def optax_chain(args):
  return optax.chain(*args)

def train(config: OmegaConf, workdir):
    # Init lightning datamodule
    log.info(f"Current working directory : {workdir}")
    log.info(f"Random seed : {config.training.seed}")
    rng = jax.random.PRNGKey(config.training.seed)
    log.info(f"Instantiating datamodule <{config.dataset._target_}>")
    datamodule: datasets.DataModule = hydra.utils.instantiate(config.dataset)
    train_ds, eval_ds = datamodule.train_ds(), datamodule.test_ds()

    # Init lightning model
    log.info(f"Instantiating model <{config.model.name}>")
    score_model = mutils.get_model(config.model.name)(config=config)
    sde: sde_lib.SDE = sde_lib.get_sde(config.sde)
    gen_model = GenModel(sde, score_model, config.sde.continuous, config.dataset.image_size, config.dataset.num_channels)
    
    tx = hydra.utils.instantiate(config.training.optim)

    rng, init_model_rng = jax.random.split(rng)
    init_model_states, initial_params = gen_model.init_params(init_model_rng)
    
    log.info(f"Instantiating train state")
    state = CustomTrainState.create(apply_fn=gen_model.get_score_fn(train=True, return_state=True), 
                                    params=initial_params, 
                                    model_states=init_model_states,
                                    tx=tx,
                                    rng=rng)
    del init_model_states, initial_params

    log.info(f"Get train and eval step")
    train_step = functools.partial(jax.lax.scan, gen_model.get_step_fn(config.loss, is_training=True, is_parallel=True))
    eval_step  = functools.partial(jax.lax.scan, gen_model.get_step_fn(config.loss, is_training=False, is_parallel=True))
    train_step = jax.jit(train_step)
    eval_step = jax.jit(eval_step)

    trainer = Trainer(config, workdir, train_step, eval_step, end_step=lambda *args: None)
    state = trainer.fit(rng, state, train_ds, eval_ds)




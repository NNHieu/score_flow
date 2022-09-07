import logging
import os
import warnings
from typing import Any, Callable, List, Optional, Sequence
from functools import wraps
import inspect
import jax

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        # if rank_zero_only.rank == 0:
        if jax.process_index() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    output_dir: os.PathLike,
    fields: Sequence[str] = (
        "training",
        "model",
        "dataset",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open(os.path.join(output_dir, "config_tree.log"), "w") as fp:
        rich.print(tree, file=fp)


# @rank_zero_only
# def log_hyperparameters(
#     config: DictConfig,
#     model: pl.LightningModule,
#     datamodule: pl.LightningDataModule,
#     trainer: pl.Trainer,
#     callbacks: List[pl.Callback],
#     logger: List[pl.loggers.LightningLoggerBase],
# ) -> None:
#     """This method controls which parameters from Hydra config are saved by Lightning loggers.
#     Additionaly saves:
#         - number of model parameters
#     """

#     hparams = {}

#     # choose which parts of hydra config will be saved to loggers
#     hparams["trainer"] = config["trainer"]
#     hparams["model"] = config["model"]
#     hparams["datamodule"] = config["datamodule"]

#     if "seed" in config:
#         hparams["seed"] = config["seed"]
#     if "callbacks" in config:
#         hparams["callbacks"] = config["callbacks"]

#     # save number of model parameters
#     hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
#     hparams["model/params/trainable"] = sum(
#         p.numel() for p in model.parameters() if p.requires_grad
#     )
#     hparams["model/params/non_trainable"] = sum(
#         p.numel() for p in model.parameters() if not p.requires_grad
#     )

#     # send hparams to all loggers
#     trainer.logger.log_hyperparams(hparams)


# def finish(
#     config: DictConfig,
#     model: pl.LightningModule,
#     datamodule: pl.LightningDataModule,
#     trainer: pl.Trainer,
#     logger: List[pl.loggers.LightningLoggerBase],
#     callbacks: List[pl.Callback] = None,
# ) -> None:
#     """Makes sure everything closed properly."""

#     # without this sweeps with wandb logger might crash!
#     for lg in logger:
#         if isinstance(lg, pl.loggers.wandb.WandbLogger):
#             import wandb

#             wandb.finish()

#https://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
def autoargs(*include, **kwargs):
    def _autoargs(func):
        # attrs, varargs, varkw, defaults = inspect.signature(func)
        signature = inspect.signature(func)

        def sieve(attr):
            if kwargs and attr in kwargs['exclude']:
                return False
            if not include or attr in include:
                return True
            else:
                return False

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            bound_arguments = signature.bind(self, *args, **kwargs)
            for attr, val in bound_arguments.arguments.items():
                if sieve(attr):
                    setattr(self, attr, val)
            return func(self, *args, **kwargs)
        return wrapper
    return _autoargs
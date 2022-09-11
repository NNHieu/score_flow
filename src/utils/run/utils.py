import os
from pathlib import Path
from importlib.util import find_spec
import time
import warnings
from typing import Any, Callable, List, Optional, Sequence
from functools import wraps
import inspect

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from .rank_zero import rank_zero_only
from src.utils.run import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)
    
def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.
    Makes multirun more resistant to failure.
    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap

@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
      log.warning("Extras config not found! <cfg.extras=null>")
      return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    # if cfg.extras.get("enforce_tags"):
    #     log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
    #     rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)



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

def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
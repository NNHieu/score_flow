import os
import dotenv
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

@hydra.main(version_base="1.2", config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.actions import train
    from src import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    # utils.extras(config)
    
    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, config.rwd, resolve=True, fields=(
            "training",
            "dataset",
            "model",
            "loss",
            "sde",
            "logger",
            "name",
        ))

    # if config.get("only_test"):
    #     return test(config)
    # # Train model
    # return train(config)
    workdir = config.rwd
    if config.get("experiment_mode"):
      workdir = config.exp_dir
    return train(config, workdir)



if __name__ == "__main__":
    main()
from train import make_dataclass_from_dict, Config, setup_logging, get_dataloader
import os
import hydra
import omegaconf
import dataclasses
from logging import getLogger
import yaml

logger = getLogger(__name__)


@hydra.main(config_path="configs", config_name="slurm_debug", version_base="1.2")
def main(raw_config: omegaconf.DictConfig):
    os.environ["JAX_PLATFORMS"] = "cpu"
    config = make_dataclass_from_dict(Config, raw_config)
    # Using `jax.distributed.initialize` shows an import warning in VsCode+Pylance saying
    # it isn't defined there.
    # BUG: Seems like it's necessary to pass this list otherwise the GPUS aren't split correctly.
    # jax_distributed_initialize(
    #     local_device_ids=list(range(distributed_env.gpus_per_task))
    # )
    setup_logging(0, 1, 3)

    logger.info("Initialized.")
    print("Config:")
    print(yaml.dump(dataclasses.asdict(config), indent=2))

    # what is this? Why is it here?
    # maybe this? https://github.com/jax-ml/jax/issues/17982

    dataloader, max_token_id = get_dataloader(config, distributed_env=None)


if __name__ == "__main__":
    main()

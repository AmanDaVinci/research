import wandb
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from image_classification.src.utils import import_class


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    dict_config = OmegaConf.to_container(config, resolve=True)
    with wandb.init(project="image-classification", config=dict_config):
        config.data_dir = f"{get_original_cwd()}/{config.data_dir}"
        trainer_class = import_class(config.trainer.module, 
                                     config.trainer.class_name)
        trainer = trainer_class(config)
        trainer.run()

if __name__ == '__main__':
    main()
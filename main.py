import hydra
from omegaconf import DictConfig
from src.experiment.run import run_experiment

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run_experiment(cfg)

if __name__ == "__main__":
    main()
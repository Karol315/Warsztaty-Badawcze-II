import logging
import os
import numpy as np
import torch
import torch.nn as nn
import utils
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(config: DictConfig):
    utils.hydra.preprocess_config(config)
    logger = utils.wandb.setup_logger(config)
    device = utils.training.setup_device(config)

    # Inicjalizujemy model RAZ - tak jak w Twojej komórce w notatniku
    model = instantiate(config.model).to(device)
    env = instantiate(config.dataset)
    acquisition = instantiate(config.acquisition, start_pos=env.entrance, maze_size=env.size)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    output_dir = HydraConfig.get().runtime.output_dir
    np.save(os.path.join(output_dir, "true_maze.npy"), env.maze)

    for round_idx in range(11):
        log.info(f"--- Runda {round_idx}/10 ---")
        acquisition.observe(env.maze)

        if len(acquisition.observed_coords) > 0:
            norm_coords = [env.normalize_coords(c[0], c[1]) for c in acquisition.observed_coords]
            coords_tensor = torch.tensor(norm_coords, dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(acquisition.observed_labels, dtype=torch.float32).to(device)

            model.train()
            # Dłuższy trening na starcie był w Twoim kodzie kluczowy
            epochs = 1200 if round_idx == 0 else 200

            for _ in range(epochs):
                optimizer.zero_grad()
                outputs = model(coords_tensor)

                # TO JEST TO: Wagowanie błędu z notatnika
                loss_raw = nn.functional.binary_cross_entropy(outputs, labels_tensor, reduction='none')
                weights = torch.where(loss_raw > 0.05, torch.tensor(10.0).to(device), torch.tensor(1.0).to(device))
                loss = (loss_raw * weights).mean()

                loss.backward()
                optimizer.step()

        # Ewaluacja
        model.eval()
        all_x, all_y = np.meshgrid(np.arange(env.size), np.arange(env.size), indexing="ij")
        grid = torch.tensor([env.normalize_coords(x, y) for x, y in zip(all_x.flatten(), all_y.flatten())],
                            dtype=torch.float32).to(device)

        with torch.no_grad():
            mean_pred = model(grid).cpu().numpy().reshape(env.size, env.size)

            # Wariancja: Skoro w modelu nie ma dropoutu, symulujemy niepewność
            # na podstawie maski eksploracji (tak jak to robiły Twoje wykresy)
            variance = np.where(acquisition.explored_mask, 0.0, 1.0)

            np.save(os.path.join(output_dir, f"mean_pred_round_{round_idx}.npy"), mean_pred)
            np.save(os.path.join(output_dir, f"variance_round_{round_idx}.npy"), variance)
            np.save(os.path.join(output_dir, f"explored_mask_round_{round_idx}.npy"), acquisition.explored_mask)
            np.save(os.path.join(output_dir, f"pos_round_{round_idx}.npy"), np.array(acquisition.pos))

        if round_idx < 10:
            acquisition.pos = acquisition.get_next_move(variance, 8)
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
from scipy.ndimage import distance_transform_edt
from src.environment.base import BaseEnv
from src.agent.memory import EpisodicMemory

log = logging.getLogger(__name__)

class Agent:
    def __init__(self, memory: EpisodicMemory, perception: Any, strategy: Any, uncertainty_module: Any, movement: Any, max_steps: int = 10, mode: str = "classification"):
        self.memory = memory
        self.perception = perception
        self.strategy = strategy
        self.uncertainty_module = uncertainty_module
        self.movement = movement
        self.max_steps = max_steps
        self.mode = mode
        self.sdf_map = None

    def run_episode(self, env: BaseEnv, model: Any):
        log.info(f"Agent: Zaczynam eksplorację! Tryb SIREN: {self.mode.upper()}")
        obs = env.reset()
        done = False

        history_pos = []
        history_memory = []
        history_visible = []
        history_pred = []

        ground_truth = env.get_ground_truth()

        # --- TWORZYMY SDF TYLKO W TLE ---
        if self.mode == "regression":
            self.sdf_map = distance_transform_edt(ground_truth == 0)
            self.sdf_map = self.sdf_map / np.max(self.sdf_map)
            loss_fn = nn.MSELoss()
            log.info("Wygenerowano ciągłą mapę SDF dla środowiska.")
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        for step in range(self.max_steps):
            log.info(f"--- MAKRO-KROK {step + 1} / {self.max_steps} ---")

            # 1. ZOBACZ: Normalna fizyka promieni (tylko zera i jedynki!)
            visible_coords, visible_vals = self.perception.observe(env.agent_pos, ground_truth)

            latest_visible_mask = np.zeros_like(ground_truth, dtype=bool)
            if len(visible_coords) > 0:
                latest_visible_mask[visible_coords[:, 0], visible_coords[:, 1]] = True

            # 2. ZAPAMIĘTAJ: Zapisuje ZAWSZE fizyczne przeszkody (0 i 1) do pamięci
            # Dzięki temu Matplotlib na 100% nie oszaleje z kolorami!
            self.memory.update(env.agent_pos, visible_coords, visible_vals)

            # 3. UCZ SIĘ
            if model is not None:
                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                self._train_model(model, optimizer, loss_fn, epochs=200)

            # 4. ZAPISZ DO HISTORII
            history_pos.append(env.agent_pos)
            history_memory.append(np.copy(self.memory.explored_map))
            history_visible.append(np.copy(latest_visible_mask))

            pred = self._get_model_prediction(model) if model else np.zeros_like(self.memory.explored_map)
            history_pred.append(pred)

            # 5. PĘTLA RUCHU
            while True:
                uncertainty_map = self.uncertainty_module.estimate(model, self.memory.explored_map)
                reachable_mask = self.movement.get_reachable_mask(self.memory.explored_map, env.agent_pos)

                for past_pos in history_pos:
                    reachable_mask[past_pos[0], past_pos[1]] = False
                reachable_mask[env.agent_pos[0], env.agent_pos[1]] = False

                if not np.any(reachable_mask):
                    log.info("Brak nowych, sensownych celów (wszystko zasłonięte ścianami lub odwiedzone). Koniec!")
                    done = True
                    break

                target_pos = self.strategy.select_action(
                    current_pos=env.agent_pos,
                    uncertainty_map=uncertainty_map,
                    reachable_mask=reachable_mask,
                    memory_map=self.memory.explored_map
                )

                obs, env_done, info = env.step(target_pos)

                if info.get("hit_wall", False):
                    log.warning(f"Zderzenie ze ścianą na {target_pos}! Oznaczam w pamięci.")
                    # Nawigacja ZAWSZE musi widzieć ścianę jako 1.0!
                    self.memory.explored_map[target_pos[0], target_pos[1]] = 1.0
                else:
                    log.info(f"Agent przemieszcza się pomyślnie na pozycję {target_pos}.")
                    if env_done:
                        done = True
                    break

            if done:
                break

        log.info("Agent: Zakończyłem eksplorację.")
        return history_pos, history_memory, history_visible, history_pred

    def _train_model(self, model: Any, optimizer: Any, loss_fn: Any, epochs: int):
        model.train()
        coords, labels = self.memory.get_dataset_for_model()
        if len(coords) == 0:
            return

        # --- MAGIA ROZDZIELENIA NAWIGACJI OD UCZENIA ---
        # Dopiero TUTAJ, tuż przed wstrzyknięciem danych do sieci, podmieniamy
        # chamskie jedynki i zera na piękne gradienty SDF. Pamięć na ekranie zostaje normalna.
        if self.mode == "regression" and self.sdf_map is not None:
            labels = self.sdf_map[coords[:, 0], coords[:, 1]]

        map_size = self.memory.map_size
        tensor_coords = torch.tensor(coords, dtype=torch.float32) / (map_size - 1) * 2.0 - 1.0
        tensor_labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(tensor_coords)
            loss = loss_fn(preds, tensor_labels)
            loss.backward()
            optimizer.step()

        log.info(f"Model zaktualizowany na {len(coords)} punktach (Strata: {loss.item():.4f})")

    def _get_model_prediction(self, model: Any) -> np.ndarray:
        model.eval()
        map_size = self.memory.map_size

        y, x = np.meshgrid(np.arange(map_size), np.arange(map_size), indexing='ij')
        all_coords = np.stack([y.flatten(), x.flatten()], axis=-1)
        tensor_coords = torch.tensor(all_coords, dtype=torch.float32) / (map_size - 1) * 2.0 - 1.0

        with torch.no_grad():
            logits = model(tensor_coords).squeeze()
            if self.mode == "classification":
                preds = torch.sigmoid(logits).numpy()
            else:
                preds = logits.numpy()

        return preds.reshape(map_size, map_size)
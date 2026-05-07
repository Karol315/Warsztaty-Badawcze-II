import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
from src.env.base import BaseEnv
from src.agent.memory import EpisodicMemory

log = logging.getLogger(__name__)


class Agent:
    def __init__(self, memory: EpisodicMemory, perception: Any, strategy: Any, uncertainty_module: Any, movement: Any):
        self.memory = memory
        self.perception = perception
        self.strategy = strategy
        self.uncertainty_module = uncertainty_module
        self.movement = movement

    def run_episode(self, env: BaseEnv, model: Any):
        log.info("Agent: Zaczynam eksplorację!")
        obs = env.reset()
        done = False

        history_pos = []
        history_memory = []
        history_visible = []
        history_pred = []

        loss_fn = nn.BCEWithLogitsLoss()

        for step in range(10):
            log.info(f"--- MAKRO-KROK {step + 1} ---")

            ground_truth = env.get_ground_truth()

            # 1. ZOBACZ
            visible_coords, visible_vals = self.perception.observe(env.agent_pos, ground_truth)

            latest_visible_mask = np.zeros_like(ground_truth, dtype=bool)
            if len(visible_coords) > 0:
                latest_visible_mask[visible_coords[:, 0], visible_coords[:, 1]] = True

            # 2. ZAPAMIĘTAJ
            self.memory.update(env.agent_pos, visible_coords, visible_vals)

            # 3. UCZ SIĘ
            if model is not None:
                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                self._train_model(model, optimizer, loss_fn, epochs=200)

            # 4. ZAPISZ DO HISTORII (Zapisujemy klatkę dopiero po zebraniu wiedzy)
            history_pos.append(env.agent_pos)
            history_memory.append(np.copy(self.memory.explored_map))
            history_visible.append(np.copy(latest_visible_mask))

            pred = self._get_model_prediction(model) if model else np.zeros_like(self.memory.explored_map)
            history_pred.append(pred)

            # 5. PĘTLA RUCHU (Odporna na zderzenia ze ścianami)
            while True:
                uncertainty_map = self.uncertainty_module.estimate(model, self.memory.explored_map)
                reachable_mask = self.movement.get_reachable_mask(self.memory.explored_map, env.agent_pos)

                target_pos = self.strategy.select_action(uncertainty_map, reachable_mask, self.memory.explored_map)

                obs, done, info = env.step(target_pos)

                # FIZYKA: Reakcja na niewidzialne ściany w środowisku
                if info.get("hit_wall", False):
                    log.warning(
                        f"Zderzenie ze ścianą na {target_pos}! Oznaczam w pamięci jako 1 i natychmiast szukam innej drogi.")
                    self.memory.explored_map[target_pos[0], target_pos[1]] = 1
                    # Pętla while kręci się dalej - agent szuka nowego celu BEZ zużywania makro-kroku
                else:
                    log.info(f"Agent przemieszcza się pomyślnie na pozycję {target_pos}.")
                    break  # Sukces! Zakończyliśmy makro-krok, wychodzimy z pętli while.

            if done:
                break

        log.info("Agent: Zakończyłem eksplorację.")
        return history_pos, history_memory, history_visible, history_pred

    def _train_model(self, model: Any, optimizer: Any, loss_fn: Any, epochs: int):
        model.train()
        coords, labels = self.memory.get_dataset_for_model()
        if len(coords) == 0:
            return

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
            preds = torch.sigmoid(logits).numpy()

        return preds.reshape(map_size, map_size)
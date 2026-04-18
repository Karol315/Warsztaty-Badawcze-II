import logging
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils

log = logging.getLogger(__name__)


# --- AGENT AKTYWNEGO UCZENIA (Połączenie logiki Kuby i Karola) ---
class ActiveAgent:
    def __init__(self, start_pos, maze_size, vision_radius=8):
        self.pos = start_pos
        self.size = maze_size
        self.vision_radius = vision_radius
        self.explored_mask = np.zeros((maze_size, maze_size), dtype=bool)
        # map: 0 to nieznane, 1 to ściana, -1 to puste korytarze
        self.map = np.zeros((maze_size, maze_size))
        self.observed_coords = []
        self.observed_labels = []

    def normalize_coords(self, x, y):
        return (x / (self.size - 1)) * 2 - 1, (y / (self.size - 1)) * 2 - 1

    def observe(self, true_maze):
        cx, cy = self.pos
        angles = np.linspace(0, 2 * np.pi, 120)
        for angle in angles:
            for r in range(1, self.vision_radius + 1):
                nx = int(round(cx + r * np.cos(angle)))
                ny = int(round(cy + r * np.sin(angle)))

                # Granice mapy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    break

                # Dodaj do pamięci, jeśli jeszcze tego nie widzieliśmy
                if not self.explored_mask[nx, ny]:
                    self.explored_mask[nx, ny] = True
                    norm_x, norm_y = self.normalize_coords(nx, ny)
                    self.observed_coords.append([norm_x, norm_y])
                    self.observed_labels.append([true_maze[nx, ny]])
                    self.map[nx, ny] = 1 if true_maze[nx, ny] == 1 else -1

                # Wzrok nie przenika przez ściany
                if true_maze[nx, ny] == 1:
                    break

                    # Agent zawsze widzi pole, na którym stoi
        if not self.explored_mask[cx, cy]:
            self.explored_mask[cx, cy] = True
            norm_x, norm_y = self.normalize_coords(cx, cy)
            self.observed_coords.append([norm_x, norm_y])
            self.observed_labels.append([true_maze[cx, cy]])
            self.map[cx, cy] = -1

    def get_maxvar_move(self, variance_map, R):
        """Strategia MaxVar: Idź w stronę granicy z największą niepewnością modelu"""
        queue = deque([self.pos])
        visited = {self.pos}
        parents = {self.pos: None}
        frontiers = []

        # Przeszukiwanie BFS po znanych i pustych korytarzach
        while queue:
            curr = queue.popleft()
            cx, cy = curr

            # Sprawdź czy pole obok jest nieznane (granica)
            is_frontier = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.map[nx, ny] == 0:  # 0 = pole nieznane
                        is_frontier = True

            if is_frontier:
                frontiers.append((curr, variance_map[cx, cy]))

            # Kontynuuj BFS tylko wzdłuż pustych korytarzy
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    # -1 to odkryty pusty korytarz
                    if (nx, ny) not in visited and self.map[nx, ny] == -1:
                        visited.add((nx, ny))
                        parents[(nx, ny)] = curr
                        queue.append((nx, ny))

        if not frontiers:
            return self.pos  # Cały labirynt zwiedzony

        # Wybieramy cel z największą wariancją z MCDropoutu
        best_target = max(frontiers, key=lambda x: x[1])[0]

        # Odtwarzamy ścieżkę do celu
        path = []
        curr = best_target
        while curr is not None:
            path.append(curr)
            curr = parents[curr]
        path.reverse()

        # Ruszamy się maksymalnie o budżet R
        steps = min(R, len(path) - 1)
        return path[steps] if steps > 0 else self.pos


# --- ORKIESTRACJA EKSPERYMENTU ---
def run(config: DictConfig):
    utils.hydra.preprocess_config(config)
    logger = utils.wandb.setup_logger(config)
    device = utils.training.setup_device(config)
    utils.training.set_seed(config.exp.seed)

    log.info(f"Running Active Learning PoC on device: {device}")

    # 1. Inicjalizacja komponentów
    env = instantiate(config.dataset)
    model = instantiate(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()  # Sigmoid na wyjściu + BCELoss

    # Parametry z tabelki projektu
    T = 10  # 10 rund eksploracji
    R = 8  # Zasięg ruchu w jednej rundzie

    agent = ActiveAgent(start_pos=env.entrance, maze_size=env.size, vision_radius=R)
    output_dir = HydraConfig.get().runtime.output_dir

    for round_idx in range(T + 1):
        log.info(f"--- Runda {round_idx}/{T} ---")

        # 2. Agent rozgląda się
        agent.observe(env.maze)

        # 3. Model uczy się na podstawie tego, co widział agent
        if len(agent.observed_coords) > 0:
            coords_tensor = torch.tensor(agent.observed_coords, dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(agent.observed_labels, dtype=torch.float32).to(device)

            # Standardowy trening SIREN
            model.train()  # Ważne: Dropout włączony w trakcie nauki
            epochs = 300 if round_idx == 0 else 100
            for _ in range(epochs):
                optimizer.zero_grad()
                outputs = model(coords_tensor)
                loss = criterion(outputs, labels_tensor)
                loss.backward()
                optimizer.step()
            logger.log({"metrics/loss": loss.item(), "round": round_idx})

        # 4. MCDropout - Agent "myśli", czego nie jest pewien (UQ: MCDrop)
        model.train()  # KRYTYCZNE: Model musi pozostać w trybie train, by wymusić zrzucanie neuronów!
        all_x, all_y = np.meshgrid(np.arange(env.size), np.arange(env.size), indexing="ij")
        inputs = torch.tensor(
            [agent.normalize_coords(x, y) for x, y in zip(all_x.flatten(), all_y.flatten())],
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            mc_samples = 20
            # Losujemy 20 różnych predykcji, z racji dropoutu każda będzie inna
            preds = torch.stack([model(inputs) for _ in range(mc_samples)])
            # Liczymy wariancję (do strategii MaxVar)
            variance = preds.var(dim=0).squeeze().cpu().numpy().reshape(env.size, env.size)
            # Liczymy średnią predykcję (do metryki PSNR)
            mean_pred = preds.mean(dim=0).squeeze().cpu().numpy().reshape(env.size, env.size)

            # 5. Zapisujemy wyniki (Accuracy i % Odkrycia)
            # Zamieniamy prawdopodobieństwo (0-1) na twarde decyzje: puste(0) lub ściana(1)
            binary_pred = (mean_pred >= 0.5).astype(float)

            # Dokładność: ile pikseli model zgadł poprawnie w całym labiryncie
            accuracy = np.mean(binary_pred == env.maze)

            # Procent odkrycia: jak dużą część mapy agent fizycznie zobaczył
            pct_explored = agent.explored_mask.sum() / (env.size * env.size)

            logger.log(
                {
                    "metrics/accuracy": accuracy,
                    "metrics/pct_explored": pct_explored,
                    "round": round_idx,
                }
            )
            log.info(f"Dokładność: {accuracy:.1%} | Odkryto: {pct_explored:.1%}")

        # 6. Wizualizacja (Oryginał, Wiedza, Predykcja Modelu)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(env.maze, cmap="binary")
        axes[0].plot(agent.pos[1], agent.pos[0], "ro", markersize=8, label="Agent")
        axes[0].set_title("Prawdziwy Labirynt")

        explored_map = np.full((env.size, env.size), 0.5)
        explored_map[agent.explored_mask] = env.maze[agent.explored_mask]
        axes[1].imshow(explored_map, cmap="gray_r", vmin=0, vmax=1)
        axes[1].plot(agent.pos[1], agent.pos[0], "ro", markersize=8)
        axes[1].set_title("Wiedza Agenta")

        im = axes[2].imshow(mean_pred, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[2].set_title(f"Świat SIREN (Acc: {accuracy:.1%})")  # <--- O, TUTAJ ZMIANA
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"runda_{round_idx}.png"))
        plt.close()

        # 7. Agent decyduje, dokąd idzie w następnej rundzie (MaxVar Acquisition)
        if round_idx < T:
            agent.pos = agent.get_maxvar_move(variance, R)

    log.info("Zakończono Proof of Concept pomyślnie!")

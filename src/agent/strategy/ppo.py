import numpy as np
import logging
from typing import Tuple, Any, Optional
from .base import BaseStrategy

log = logging.getLogger(__name__)

# Stable-Baselines3 jest wymagane: pip install stable-baselines3
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3.common.env_util import make_vec_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    log.warning("stable-baselines3 nie jest zainstalowane! Uruchom: pip install stable-baselines3 gymnasium")


# ---------------------------------------------------------------------------
# Wrapper środowiska: zamienia DiscreteMaze w format Gymnasium dla SB3
# ---------------------------------------------------------------------------
class MazeGymEnv(gym.Env):
    """
    Gymnasium wrapper wokół DiscreteMaze.
    Obserwacja: spłaszczona mapa pamięci agenta (wartości -1, 0, 1) + pozycja agenta (2 liczby).
    Akcja: indeks komórki docelowej (y * size + x) — dyskretna przestrzeń.
    Nagroda: +1 za odkrycie nowej komórki, -0.1 za uderzenie w ścianę, -0.01 za krok.
    """

    def __init__(self, map_size: int = 32):
        super().__init__()
        self.map_size = map_size
        self.n_cells = map_size * map_size

        # Przestrzeń obserwacji: mapa (spłaszczona) + pozycja (y, x) znormalizowana
        obs_dim = self.n_cells + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Przestrzeń akcji: każda komórka mapy jako możliwy cel
        self.action_space = spaces.Discrete(self.n_cells)

        self._reset_state()

    def _reset_state(self):
        """Generuje nowy labirynt i resetuje stan wewnętrzny."""
        from mazelib import Maze
        from mazelib.generate.Prims import Prims

        m = Maze()
        m.generator = Prims(self.map_size // 2, self.map_size // 2)
        m.generate()
        self.grid = m.grid.astype(np.float32)

        # Upewnij się że grid ma dokładnie map_size x map_size
        if self.grid.shape[0] != self.map_size or self.grid.shape[1] != self.map_size:
            # mazelib generuje (2*(n//2)+1) x (2*(n//2)+1), przycinamy lub paddujemy
            self.grid = self.grid[:self.map_size, :self.map_size]

        self.agent_pos = np.array([1, 1], dtype=np.int32)
        # Mapa pamięci: -1 = nieznane, 0 = wolne, 1 = ściana
        self.memory_map = np.full((self.map_size, self.map_size), -1.0, dtype=np.float32)
        # Agent widzi swoją startową komórkę
        self.memory_map[1, 1] = 0.0
        self.visited = set()
        self.visited.add((1, 1))
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        flat_map = self.memory_map.flatten()
        pos_norm = self.agent_pos.astype(np.float32) / (self.map_size - 1) * 2.0 - 1.0
        return np.concatenate([flat_map, pos_norm], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        self.step_count += 1
        target_y = action // self.map_size
        target_x = action % self.map_size

        # Ogranicz do granic mapy
        target_y = np.clip(target_y, 0, self.map_size - 1)
        target_x = np.clip(target_x, 0, self.map_size - 1)

        reward = -0.01  # koszt każdego kroku

        if self.grid[target_y, target_x] == 1:
            # Uderzenie w ścianę
            self.memory_map[target_y, target_x] = 1.0
            reward -= 0.1
        else:
            self.agent_pos = np.array([target_y, target_x], dtype=np.int32)

            # Odkrycie nowych komórek w promieniu widzenia (uproszczone: tylko sąsiedztwo)
            newly_discovered = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = target_y + dy, target_x + dx
                    if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                        if self.memory_map[ny, nx] == -1.0:
                            self.memory_map[ny, nx] = float(self.grid[ny, nx])
                            newly_discovered += 1

            reward += newly_discovered * 0.1

            if (target_y, target_x) not in self.visited:
                self.visited.add((target_y, target_x))
                reward += 0.5  # bonus za nową komórkę

        # Epizod kończy się po max_steps lub odkryciu całej mapy
        max_steps = self.map_size * 4
        terminated = self.step_count >= max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Strategia PPO — kompatybilna z BaseStrategy
# ---------------------------------------------------------------------------
class PPOStrategy(BaseStrategy):
    """
    Strategia eksploracji oparta na PPO (Proximal Policy Optimization).

    Trenuje politykę RL na losowych labiryntach przed eksploracją,
    następnie używa jej zero-shot do wyboru akcji.

    Parametry:
        map_size:        Rozmiar labiryntu (musi zgadzać się ze środowiskiem).
        total_timesteps: Liczba kroków treningowych PPO (domyślnie 200_000).
        n_envs:          Liczba równoległych środowisk treningowych.
        model_path:      Opcjonalna ścieżka do zapisanego/wczytanego modelu.
                         Jeśli podana i plik istnieje, pomija trening.
    """

    def __init__(
        self,
        map_size: int = 32,
        total_timesteps: int = 200_000,
        n_envs: int = 4,
        model_path: Optional[str] = None,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "Zainstaluj stable-baselines3: pip install stable-baselines3 gymnasium"
            )

        self.map_size = map_size
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.model_path = model_path
        self.policy: Optional[SB3_PPO] = None

        self._load_or_train()

    def _load_or_train(self):
        """Wczytuje istniejący model lub trenuje nowy."""
        import os

        if self.model_path and os.path.exists(self.model_path + ".zip"):
            log.info(f"PPO: Wczytuję wytrenowany model z {self.model_path}")
            self.policy = SB3_PPO.load(self.model_path)
            return

        log.info(
            f"PPO: Rozpoczynam trening na {self.total_timesteps} krokach "
            f"({self.n_envs} równoległych środowisk, map_size={self.map_size})..."
        )

        def make_env():
            return MazeGymEnv(map_size=self.map_size)

        vec_env = make_vec_env(make_env, n_envs=self.n_envs)

        self.policy = SB3_PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.01,  # lekka entropia = zachęta do eksploracji
            tensorboard_log=None,
        )

        self.policy.learn(total_timesteps=self.total_timesteps)

        if self.model_path:
            self.policy.save(self.model_path)
            log.info(f"PPO: Model zapisany do {self.model_path}")

        vec_env.close()
        log.info("PPO: Trening zakończony!")

    def _build_obs(
        self,
        current_pos: Tuple[int, int],
        memory_map: np.ndarray,
    ) -> np.ndarray:
        """Buduje wektor obserwacji zgodny z MazeGymEnv."""
        flat_map = memory_map.flatten().astype(np.float32)
        pos_norm = (
            np.array(current_pos, dtype=np.float32) / (self.map_size - 1) * 2.0 - 1.0
        )
        return np.concatenate([flat_map, pos_norm])

    def select_action(
        self,
        current_pos: Tuple[int, int],
        uncertainty_map: Any,
        reachable_mask: np.ndarray,
        memory_map: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Wybiera cel zgodnie z polityką PPO, ale tylko spośród osiągalnych komórek.

        Jeśli PPO proponuje nieosiągalną komórkę, fallback do komórki osiągalnej
        z największą niepewnością (zachowanie greedy jako bezpieczna siatka).
        """
        obs = self._build_obs(current_pos, memory_map)

        # SB3 oczekuje batcha obserwacji
        action, _ = self.policy.predict(obs[np.newaxis, :], deterministic=True)
        action = int(action[0])

        target_y = action // self.map_size
        target_x = action % self.map_size

        # Sprawdź czy PPO wybrało osiągalną komórkę
        if (
            0 <= target_y < self.map_size
            and 0 <= target_x < self.map_size
            and reachable_mask[target_y, target_x]
        ):
            return (target_y, target_x)

        # Fallback: spośród osiągalnych wybierz tę z największą niepewnością
        log.debug("PPO wybrał nieosiągalną komórkę — fallback do max uncertainty.")
        valid_coords = np.argwhere(reachable_mask)
        if len(valid_coords) == 0:
            return current_pos

        uncertainties = uncertainty_map[valid_coords[:, 0], valid_coords[:, 1]]
        best_idx = np.argmax(uncertainties)
        best = valid_coords[best_idx]
        return (int(best[0]), int(best[1]))

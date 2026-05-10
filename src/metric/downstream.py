import numpy as np
from src.planning.astar import astar

class DownstreamMetrics:
    def __init__(self, wall_threshold: float = 0.5):
        self.wall_threshold = wall_threshold

    def compute(self, pred: np.ndarray, ground_truth: np.ndarray) -> dict:
        # 1. Klasyczne metryki
        mse = np.mean((pred - ground_truth) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        # 2. Ewaluacja A*
        binary_pred = (pred > self.wall_threshold).astype(int)
        
        max_y, max_x = ground_truth.shape
        start = (1, 1) # Punkt startowy
        goal = (max_y - 2, max_x - 2) # Meta

        # Planowanie na wyobrażeniu z SIREN
        path = astar(binary_pred, start, goal)

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        max_dist = heuristic(start, goal)
        last_safe_node = start
        strict_success = 0.0  # Wersja 0/100
        
        if path is not None and len(path) > 0:
            for y, x in path:
                if ground_truth[y, x] == 1:
                    break # Zderzenie z PRAWDZIWĄ ścianą
                last_safe_node = (y, x)
                
            # Jeśli ostatni bezpieczny punkt to meta, mamy pełen sukces
            if last_safe_node == goal:
                strict_success = 100.0

        # Wersja progresywna (dystans)
        current_dist = heuristic(last_safe_node, goal)
        prog_success = max(0.0, ((max_dist - current_dist) / max_dist) * 100.0)

        return {
            "mse": mse,
            "psnr": psnr,
            "path_success_strict": strict_success,
            "path_success_prog": prog_success,
            "path_length": len(path) if path else 0,
            "path": path
        }
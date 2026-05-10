import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import random
import torch
from hydra.core.hydra_config import HydraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def set_all_seeds(seed: int):
    """Zamraża losowość w całym środowisku (Python, Numpy, Pytorch) dla pełnej reprodukowalności."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Dla konfiguracji multi-GPU

    # Wymusza deterministyczne algorytmy w CuDNN (może lekko spowolnić, ale gwarantuje ten sam wynik)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(cfg: DictConfig):
    # 1. ZAMROŻENIE LOSOWOŚCI Z CONFIGU
    seed = cfg.get("seed", 42)
    set_all_seeds(seed)

    log.info(f"Rozpoczynam eksperyment: {cfg.experiment_name} (Seed: {seed})")

    env = hydra.utils.instantiate(cfg.environment)
    model = hydra.utils.instantiate(cfg.model)
    agent = hydra.utils.instantiate(cfg.agent)
    metric_module = hydra.utils.instantiate(cfg.metric)

    log.info("Start głównej pętli eksperymentu!")

    # 2. Eksperyment (Twarda logika eksploracji)
    history_pos, history_memory, history_visible, history_pred = agent.run_episode(env=env, model=model)

    log.info("Eksperyment zakończony. Obliczanie metryk jakości...")
    ground_truth = env.get_ground_truth()

    psnr_scores = []
    history_paths = []
    
    # 3. Ewaluacja matematyczna (PSNR wyliczany w tle i logowany w terminalu)
    for step, pred in enumerate(history_pred):
        metrics = metric_module.compute(pred, ground_truth)
        psnr_scores.append(metrics["psnr"])
        history_paths.append(metrics.get("path", None))
        
        # Pobranie obu wartości A* (Progresywnej i Restrykcyjnej)
        strict_val = metrics.get("path_success_strict", -1)
        prog_val = metrics.get("path_success_prog", -1)
        
        success_str = ""
        if strict_val != -1 and prog_val != -1:
            success_str = f" | A* Prog: {prog_val:02.0f}% | A* Strict: {strict_val:3.0f}%"
        elif "path_success" in metrics: # Zabezpieczenie dla starej metryki, gdybyś jej użył
            path_success = metrics.get("path_success", -1)
            success_str = f" | A* Success: {path_success:.0f}%" if path_success != -1 else ""
        
        log.info(f"Krok {step + 1:2d}: MSE = {metrics['mse']:.4f}, PSNR = {metrics['psnr']:.2f} dB{success_str}")

    # Ustalenie folderu wyjściowego z dynamicznej ścieżki Hydry
    output_dir = HydraConfig.get().runtime.output_dir
    plots_dir = os.path.join(output_dir, "plots")

    log.info("Generowanie i zapisywanie wykresów...")
    
    # WYCIĄGAMY TRYB Z AGENTA
    mode = getattr(agent, "mode", "classification")
    visualize_results(env, history_pos, history_memory, history_visible, history_pred, history_paths, plots_dir, mode)


def visualize_results(env, history_pos, history_memory, history_visible, history_pred, history_paths, plots_dir, mode):
    num_steps = len(history_pos)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.2)

    ax_gt, ax_mem, ax_pred = axes

    im_gt = ax_gt.imshow(env.get_ground_truth(), cmap='binary')
    pt_gt, = ax_gt.plot([], [], 'ro', markersize=8, label='Agent')

    # Oryginalne i jedyne słuszne kolory: -1 to szary, 0 to biały, 1 to czarny
    cmap_memory = ListedColormap(['gray', 'white', 'black'])
    im_mem = ax_mem.imshow(history_memory[-1], cmap=cmap_memory, vmin=-1.5, vmax=1.5)

    # --- MAGIA WIZUALIZACJI ZALEŻNA OD TRYBU ---
    if mode == "regression":
        im_pred = ax_pred.imshow(history_pred[-1], cmap='plasma', vmin=0.0, vmax=1.0)
        label_text = "SDF (Dystans do ściany)"
    else:
        im_pred = ax_pred.imshow(history_pred[-1], cmap='RdBu_r', vmin=0.0, vmax=1.0)
        label_text = "Prawdopodobieństwo ściany"

    divider = make_axes_locatable(ax_pred)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_pred, cax=cax, label=label_text)

    contour_refs = {"gt": None, "mem": None}

    def draw_step(idx):
        pos_y, pos_x = history_pos[idx]
        mask = history_visible[idx]

        ax_gt.set_title(f"Krok {idx + 1}: Prawdziwa Mapa")
        ax_mem.set_title(f"Krok {idx + 1}: Pamięć Agenta")
        ax_pred.set_title(f"Krok {idx + 1}: Predykcja SIREN")

        pt_gt.set_data([pos_x], [pos_y])
        im_mem.set_data(history_memory[idx])
        im_pred.set_data(history_pred[idx])

        # Bezpieczne czyszczenie konturów (odporne na różne wersje Matplotlib)
        if contour_refs["gt"] is not None:
            try:
                contour_refs["gt"].remove()
            except AttributeError:
                for c in contour_refs["gt"].collections: c.remove()
        if contour_refs["mem"] is not None:
            try:
                contour_refs["mem"].remove()
            except AttributeError:
                for c in contour_refs["mem"].collections: c.remove()

        if np.any(mask):
            contour_refs["gt"] = ax_gt.contour(mask, levels=[0.5], colors='cyan', linewidths=1.5)
            contour_refs["mem"] = ax_mem.contour(mask, levels=[0.5], colors='cyan', linewidths=1.5)
        else:
            contour_refs["gt"] = None
            contour_refs["mem"] = None

        # Rysowanie wyobrażonej trasy A* na predykcji SIREN
        if hasattr(draw_step, "line_astar") and draw_step.line_astar:
            draw_step.line_astar.remove()
            draw_step.line_astar = None
            
        path = history_paths[idx]
        if path is not None and len(path) > 0:
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            # Zielona linia jeśli zgadł trasę
            draw_step.line_astar, = ax_pred.plot(path_x, path_y, color='lime', linewidth=2, linestyle='--')

        fig.canvas.draw_idle()

    # --- ZAPIS DO PLIKÓW ---
    #os.makedirs(plots_dir, exist_ok=True)
    #for i in range(num_steps):
        #draw_step(i)
        #plt.savefig(os.path.join(plots_dir, f"step_{i + 1:02d}.png"), bbox_inches='tight')

    #log.info(f"Wykresy PNG bezpiecznie zapisane w: {os.path.abspath(plots_dir)}")

    # --- ZAPIS DO PLIKÓW (ZMIENIONE: Zapisuje TYLKO ostatni krok) ---
    os.makedirs(plots_dir, exist_ok=True)
    draw_step(num_steps - 1) # Rysuj tylko finałową klatkę
    plt.savefig(os.path.join(plots_dir, "final_exploration.png"), bbox_inches='tight')

    log.info(f"Zapisano TYLKO JEDEN finałowy wykres PNG w: {os.path.abspath(plots_dir)}")

    # --- INTERAKTYWNE GUI ---
    def update_slider(val):
        idx = int(slider.val) - 1
        draw_step(idx)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, 'Krok Eksploracji', 1, num_steps, valinit=num_steps, valstep=1, color='blue')
    slider.on_changed(update_slider)
    fig.slider = slider

    draw_step(num_steps - 1)
    plt.show()
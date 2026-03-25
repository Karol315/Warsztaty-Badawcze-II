# Exploration Strategies

## Overall Goal
The goal is to:
- Determine whether active INR-based world model learning provides better exploration than established robotics and reinforcement learning baselines when the agent cannot teleport (i.e., it must physically navigate to observation locations within a movement budget).

- Test whether the INR world model enables downstream task performance (path planning, collision avoidance) beyond mere reconstruction quality.

- Compare these principled active strategies against RL-native methods, including PPO and SAC, to see if end-to-end long-horizon learning outperforms uncertainty-driven heuristics.

## Proposition

Subprojects on the topic of Uncertainty Quantification (Team 1) and Acquisition Functions (Team 2) assumes that the agent can observe any coordinate instantly. This is unrealistic for physical agents. Two constraints emerge that do not exist in unconstrained active learning:

- **Reachability constraint:** At each round, the agent can only query coordinates within a reachable set determined by its current position and movement budget. The globally best coordinate may be inaccessible this round.

- **Temporal credit assignment:** Moving toward a less uncertain region may open access to a highly uncertain region in the following round. A purely greedy acquisition function is myopic in the constrained setting in a way it is not in the unconstrained case.

**Research Question**: Do INR-based active strategies (designed for the unconstrained setting) still outperform exploration methods specifically built for the embodied case? Furthermore, do RL-trained policies, which can learn long-horizon strategies to overcome myopia outperform both?

## Proof of Concept

Here we describe a first deliverable that will be requested. Table 1 outlines the experimental parameters for this initial validation run.

| Parameter | Details |
| :--- | :--- |
| **Environment** | Single procedurally generated maze (64×64, binary occupancy) |
| **Architecture** | SIREN |
| **Acquisition** | MaxVar (Unconstrained), MaxVar (Constrained), Frontier |
| **Budget** | $T=10$ rounds, Movement radius $R=8$ cells/round |
| **UQ** | MCDrop |

## Methods Compared

| Strategy | Description | Origin / Key Test |
| :--- | :--- | :--- |
| **MaxVar (Constrained)** | Candidate set restricted to reachable coordinates, then top-$k$ by $\sigma^2(x)$. Greedy and myopic. | Adapted from SP-2. |
| **BatchDPP (Constrained)** | DPP batch selection within the reachable set. Diversity expands the exploration frontier. | Adapted from SP-2. |
| **Frontier Exploration** | Always move to the nearest boundary between explored and unexplored space. Purely geometric. | Classical robotics (Yamauchi, 1997). Guarantees coverage. |
| **InfoGainRRT** | Rapidly-exploring Random Tree biased toward high-uncertainty regions. | Robotics / motion planning. Blends planning with information theory. |

## Baselines

| Framework | Details |
| :--- | :--- |
| **PPO** | On-policy baseline across all levels. Default hyperparameters, 1M training steps, zero-shot evaluation on held-out environments. |
| **SAC** | Off-policy, max-entropy. Used at Levels 3–4 where entropy regularisation aids risk-sensitive and continuous-action exploration. |
| **PPO-Lagrangian** | Constrained on-policy baseline for Level 4. Augments PPO with a Lagrange multiplier on cumulative constraint cost. From Safety-Baselines3. |
| **IQL** | Offline RL baseline for AntMaze (D4RL datasets). Avoids out-of-distribution action queries; provides direct comparison against the published AntMaze leaderboard. |

## Datasets & Environments

| Environment | Details |
| :--- | :--- |
| **Gaussian Random Field** | GP with spatially varying RBF kernel. Heterogeneous smoothness via length scale variation. Downstream task: threshold detection (IoU of exceedance region). |
| **Perlin Terrain Height Field** | Multi-octave Perlin noise scalar field. Multi-scale heterogeneity without GP overhead. Downstream task: cost-optimal path planning via A\*. |
| **PointMaze (Open, UMaze)** | Level 1 RL benchmark. Maze hidden; agent observes position + INR probe outputs + uncertainty. Open = sanity check; UMaze tests greedy acquisition myopia. |
| **Continuous SDF from Smooth Obstacles** | SDF over $[0,1]^2$ from ellipse/spline obstacles. Sharp boundaries require task-aware acquisition. Downstream task: collision-free planning with safety margin $d_\text{safe}$. |
| **Procedural Maze as Continuous SDF** | Recursive-subdivision maze as smooth SDF. Preserves corridor/junction/room topology without binary occupancy discontinuities. |
| **PointMaze-Large / AntMaze** | Level 2 RL benchmarks. PointMaze-Large for complex layouts; AntMaze adds 8-DoF locomotion and non-trivial low-level control. |
| **Wind Field (Potential Flow)** | Analytical 2D velocity field (stream + cylinders + vortices). Vector output $f:[0,1]^2\to\mathbb{R}^2$, divergence-free. No RL benchmark. Downstream task: minimum headwind path. |
| **Radiation Mapping** | Inverse-square emitters with rectangular attenuators. Field value couples traversal cost to acquisition; path planning and acquisition are non-separable. |
| **Safety Gymnasium (PointGoal, CarGoal)** | Level 4 RL benchmark. Hazard zones = radiation sources; constraint cost = dose budget. CarGoal adds non-holonomic dynamics. |

## Evaluation

Outline of the metrics used to evaluate exploration performance, focusing on embodied sample efficiency and the viability of the world model for downstream planning.

### 1. NRMSE / PSNR vs Exploration Steps

Measures the raw quality of the reconstructed world model against physical movement steps rather than abstract acquisition rounds. All signals are normalised to $[0,1]$ before computing reconstruction metrics.

First, calculate the Mean Squared Error (MSE):

$$\text{MSE}=\frac{1}{n}\sum_{i=1}^n(Y_i-\hat{Y}_i)^2$$

For normalised signals (binary occupancy, SDF clamped to $[0,1]$), report PSNR:

$$\text{PSNR}=10\cdot\log_{10}\left(\frac{1}{\text{MSE}}\right)$$

For unnormalised physical fields (terrain height, radiation intensity), report NRMSE instead:

$$\text{NRMSE}=\frac{\sqrt{\text{MSE}}}{\max(g)-\min(g)}$$

**Parameters:**
- $n$: Total number of evaluation coordinates (dense held-out grid).
- $Y_i$: Ground-truth field value at coordinate $i$.
- $\hat{Y}_i$: INR prediction at coordinate $i$.
- $g$: Ground-truth field values over the full domain (used to normalise NRMSE).

In addition to the full learning curve, three scalar milestone metrics are extracted:

$$t_{p} = \min\{ t \mid \text{PSNR}(t) \geq p \cdot \text{PSNR}_{\text{oracle}} \}, \quad p \in \{0.80,\ 0.90,\ 0.95\}$$

where $\text{PSNR}_{\text{oracle}}$ is the score achieved by the Oracle (upper bound) strategy at the end of the budget. These report how many exploration steps each strategy needs to reach a usable reconstruction quality.

---

### 2. Path Planning Success Rate

Evaluated after a fixed exploration budget. Tests whether the learned world model is accurate enough to safely navigate the environment. A path is planned using A\* on the INR reconstruction and then executed in the ground-truth environment.

$$\text{Success Rate}=\frac{1}{N_{\text{trials}}}\sum_{j=1}^{N_{\text{trials}}}\mathbf{1}[\text{CollisionFree}(\pi_j, d_{\text{safe}})]$$

where a path $\pi_j$ is considered collision-free if every point along it satisfies $\text{SDF}_{\text{true}}(\mathbf{x}) > d_{\text{safe}}$ when executed in the ground-truth environment. For binary occupancy environments, $d_{\text{safe}} = 0$ (no penetration into occupied cells).

**Parameters:**
- $N_{\text{trials}}$: Total number of random start-to-goal path planning queries.
- $\pi_j$: The path planned for query $j$ using the INR world model.
- $d_{\text{safe}}$: Safety margin; environment-specific (see Datasets & Environments).

---

### 3. Path Cost Ratio (Suboptimality)

For paths that successfully reach the goal, measures how efficient the planned path is relative to the true optimum. The definition of cost $C(\pi)$ is environment-dependent:

- **Maze / SDF environments:** $C(\pi)$ is the Euclidean length of the path.
- **Terrain environments:** $C(\pi)$ is the cumulative elevation change along the path, $\int_\pi |\nabla g| \, ds$.

$$\text{Cost Ratio}=\frac{1}{N_{\text{success}}}\sum_{j \in \text{Successes}}\frac{C(\pi_j)}{C(\pi_j^*)}$$

The optimal path cost $C(\pi_j^*)$ is computed by running A\* on the full ground-truth field with fine discretisation. A cost ratio of 1.0 is optimal; higher values indicate suboptimal planning caused by world model inaccuracies.

**Parameters:**
- $N_{\text{success}}$: Number of trials in which the path reached the goal without collision.
- $C(\pi_j)$: Cost of the path planned on the INR world model.
- $C(\pi_j^*)$: Cost of the optimal path computed on the ground-truth oracle.

---

### 4. Collision Rate

Fraction of all planned paths (including failures) that collide with at least one obstacle when executed in the ground-truth environment. Complements the success rate by quantifying how severely the world model misrepresents obstacle geometry.

$$\text{Collision Rate}=\frac{1}{N_{\text{trials}}}\sum_{j=1}^{N_{\text{trials}}}\mathbf{1}[\exists\, \mathbf{x} \in \pi_j : \text{SDF}_{\text{true}}(\mathbf{x}) \leq d_{\text{safe}}]$$

Note that Success Rate + Collision Rate $\leq 1$; paths that neither collide nor reach the goal (e.g., exceed step budget) account for the remainder.

---

### 5. RL-Specific Diagnostics

**Sample Efficiency.** PSNR evaluated against *total* environment interactions, defined as:

$$\text{Steps}_{\text{total}} = \text{Steps}_{\text{train}} + \text{Steps}_{\text{test}}$$

where $\text{Steps}_{\text{train}}$ is the number of environment interactions consumed during PPO/SAC training across all training environments, and $\text{Steps}_{\text{test}}$ is the exploration budget at evaluation time. This places RL methods on the same axis as zero-shot heuristics and makes the upfront training cost explicit.

**Zero-Shot Generalisation Drop.**

$$\Delta\text{PSNR} = \text{PSNR}_{\text{train dist}} - \text{PSNR}_{\text{held out}}$$

Evaluated on environments outside the training distribution (held-out maze layouts, real floorplans). A large $\Delta\text{PSNR}$ indicates the policy memorised training floorplans rather than learning a general exploration strategy.

**Reward Curves.** Monitored during training to verify the agent consistently improves rather than collapsing into degenerate behaviours (e.g., spinning in place, hugging walls). Assessed qualitatively by visual inspection of the smoothed reward curve.

## 🛠️ Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Change `name = "project"` in `pyproject.toml`.
3. Set the output dir in `configs/hydra/default.yaml` and the W&B `project` in `configs/logger/wandb/default.yaml`.
4. Run:
   ```sh
   uv sync
   uv run pre-commit install
   ```

## 🚀 How to use it

```sh
# Local run
WANDB_MODE=offline uv run python src/main.py

# SLURM
bash slurm/SLURM_script.sh --script slurm/run_experiment.sh --time 04:00:00 --mem 32GB --gpu 1
```

To build your experiment:
1. Subclass `BaseModel` → `src/model/`, add a matching YAML in `configs/model/`.
2. Subclass `BaseDataset` → `src/dataset/`, add a matching YAML in `configs/dataset/`.
3. Subclass `BaseMetric` → `src/metric/`, register in `configs/metric/default.yaml`.
4. Fill in the loop in `src/experiment/run.py`.

Log via the namespaced logger:
```python
logger.log({"metrics/loss": 0.42, "metadata/preds": tensor})
#           ^-- wandb scalar      ^-- saved as CSV in run dir
```

## 📚 Citation

> Citation instructions. How people should cite your work?

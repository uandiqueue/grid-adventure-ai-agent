# CS2109S Capstone Project: Grid Adventure V1 AI Agent

**Author:** Jayden

This repository contains the source code for the capstone project of **CS2109S (Introduction to AI and Machine Learning)** at the National University of Singapore (NUS), AY25/26 Semester 2.

The objective of this project is to build an intelligent, autonomous agent capable of solving **Grid Adventure: Variant 1 (V1)**, a 2D grid-based puzzle game. The agent must parse visual information from the environment (rendered top-down RGBA images), reason about game mechanics, and utilize search and planning algorithms to navigate mazes, interact with entities, and maximize its total reward.

## Overview

Grid Adventure V1 is a deterministic, fully observable, single-agent, sequential, static, and discrete puzzle environment. The game features a diverse range of entities including coins, gems, keys, locked doors, lava, pushable boxes, and power-ups (Boots/Speed, Ghost/Phasing, Shield). The agent must collect all gems (if present) before reaching the exit tile, while maximizing total reward.

### Tasks

The project is decomposed into three tasks:

| Task | Input Type | Description |
|------|-----------|-------------|
| **Task 1** | `GridState` (structured grid) | Planning and search on a fully structured grid representation. |
| **Task 2** | `ImageObservation` (RGBA image) | Perception + planning — the agent must parse a rendered image into a grid representation before planning. |
| **Task 3** | Either `GridState` or `ImageObservation` | Both input types with significantly increased level complexity (Boss Level). |

## Architecture

The solution features a two-stage pipeline combining **Machine Learning (Computer Vision)** and **Symbolic AI (Heuristic Search)**.

### 1. Vision Pipeline — CNN Tile Classifier

For tasks where the input is an `ImageObservation` (raw RGBA image), the agent employs a Convolutional Neural Network (CNN) built with PyTorch to classify each tile in the grid.

- **Architecture:** A lightweight 3-layer CNN (`Conv2d` → `ReLU` → `MaxPool2d` → `AdaptiveAvgPool2d`), followed by fully connected layers with Dropout (0.3) for regularization. Classifies tiles into 13 entity types.
- **Resolution Agnostic:** A custom `RandomPixelation` augmentation simulates varying tile resolutions (32×32 to 128×128) during training, ensuring the model generalizes across different grid rendering sizes.
- **Data Augmentation:** The training pipeline uses `torchvision.transforms.v2` for on-the-fly augmentations including `ColorJitter`, `RandomAffine`, and `RandomErasing`.
- **Handling Class Imbalance:** Weighted `CrossEntropyLoss` is used, with weights inversely proportional to class frequency.
- **Constraint-Based Post-Processing:** After CNN inference, game-rule constraints are enforced (e.g., exactly 1 agent, exactly 1 exit, equal keys and locked doors, at most 5 gems/boxes, at most 3 of each power-up).
- **Self-Contained Deployment:** The trained model is serialized into a base64-encoded, zlib-compressed blob embedded directly in the submission file — no external weight files needed. The model is lazily loaded on first `ImageObservation` encounter.

### 2. Planning Pipeline — Dynamic Weighted A* Search

Once the grid state is structured, the agent computes an optimal action sequence using **Dynamic Weighted A\* (DWA\*)**.

- **State Representation:** A frozen `SearchState` dataclass tracks dynamic elements (agent position, HP, key inventory, active power-ups, entity positions). A separate `StaticMap` holds immutable elements (walls, lava, exit, grid dimensions).
- **Dynamic Weighting:** The search weight `w(n) = 1 + ε × (1 - g(n)/(g(n)+h(n)))` naturally decays from greedy to optimal as the search progresses. The ε value is automatically tuned based on level complexity (number of interactive entities and grid size).
- **Complexity-Adaptive Search:** For extremely complex maps (complexity > 20 or large grids with many entities), the agent activates a `complex_map` mode that drops box positions from the state representation to reduce the state space, and tightens the heuristic by reducing coin benefit weight.
- **Full Lightweight State Expansion:** All actions — movement, pickup, key usage, box push, lava traversal, speed (double-step), phasing (ghost), and shield — are handled by dedicated lightweight expansion methods that directly compute successor `SearchState`s without calling the full `grid_step()` simulator. This dramatically reduces per-node overhead.
- **Heuristic Design:** The admissible heuristic uses:
  - SSSP (Single-Source Shortest Path via Dijkstra) from the exit to all reachable tiles (precomputed).
  - MST (Minimum Spanning Tree via Prim's) over gem positions for multi-gem collection estimation.
  - Turn cost modeling (+3 per step, winning step free) and coin reward offset.
  - Speed power-up lower-bound estimation for reduced movement cost.
- **Path Optimization:** The cost function encodes the game's full reward mechanics — coin pickups (+5), turn costs (-3), lava damage (-2 HP, blocked by shield) — ensuring the agent finds the highest-scoring path.

## Repository Structure

```
capstone-project/
├── capstone_project.py          # Main agent code (submission file)
├── capstone-project.ipynb       # Jupyter notebook version of the agent
├── training-lovely-model.py     # CNN training: dataset generation, augmentation, training loop
├── testing-agent-with-cases.py  # Testing script with custom test cases
├── utils.py                     # Utilities (model embedding, helpers)
├── environment.yml              # Conda environment specification
├── tutorial.ipynb               # Grid Adventure tutorial notebook (provided)
├── data/                        # Dataset folder (entity assets)
├── models/                      # Exported model weights (.pth files)
├── model_snippets/              # Base64-encoded model snippets for embedding
├── training_info/               # Training logs and per-class accuracy reports
└── agent_archive/               # Version history of the agent (iterative development)
    ├── version_1_optimality.py
    ├── version_2_normal_astar.py
    ├── version_3_normal_astar_admissible.py
    ├── version_4_pareto_dom_pruning.py
    ├── version_5_task1_workable.py
    ├── version_6_task2n3.py
    ├── version_7_lightweightbox.py
    ├── version_8_dwa_star_anytime.py
    ├── version_9_dwa_star_wo_anytime.py
    └── version_10_dwa_lightweight.py
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/uandiqueue/grid-adventure-ai-agent.git
   cd grid-adventure-ai-agent
   ```

2. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate cs2109s-ay2526s2-capstone-project
   ```

3. **Play the game manually** (to understand mechanics):
   ```bash
   grid-play --plugin grid_adventure.play.intro
   ```

4. **Create and test custom levels:**
   ```bash
   grid-play --plugin grid_adventure.play.editor
   ```

> For a more detailed breakdown of game mechanics, entity types, scoring rules, environment constraints, and how to run the agent, refer to the [`capstone-project.ipynb`](capstone-project.ipynb) notebook.

## References & AI Usage

AI tools were used during the development of this project in accordance with the course's academic integrity policies. A full log of AI-assisted interactions is available here:

📂 [AI Usage Log (Google Drive)](https://drive.google.com/drive/folders/1Nplx91pzYW7lyvZ2-9RHq637FkEHEOYM?usp=sharing)

## Acknowledgements

- **NUS CS2109S Course Team** for designing the Grid Adventure environment, API, and project framework.
- [Grid Adventure Player Guide](https://grid-universe.github.io/grid-adventure-v1/player-guide/introduction/) and [Agent Documentation](https://grid-universe.github.io/grid-adventure-v1/agent-doc/introduction/)

## Disclaimer

This repository is published for educational and portfolio purposes. The Grid Adventure environment, game assets, and project framework are the intellectual property of the NUS CS2109S course team and the Grid Universe project. If you believe any content in this repository infringes on your copyright or intellectual property rights, please contact me at **jayden8611@gmail.com** and I will take it down promptly.

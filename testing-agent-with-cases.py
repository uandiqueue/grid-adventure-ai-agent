## AI DECLARATION
# This file is used mainly to test the Agent class on various custom levels
# built upon the test cases and functions provided in the notebook provided
# Most of the code here is either pasted from the notebook, or modified by github copilot / chatgpt / claude / gemini 

## AGENT
from test_agent import Agent

## CONFIG
from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation
from grid_adventure.step import Action
from grid_adventure.grid import step as grid_step
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.rendering import DEFAULT_ASSET_ROOT
from grid_adventure.entities import (
    AgentEntity,
    FloorEntity,
    WallEntity,
    ExitEntity,
    CoinEntity,
    GemEntity,
    KeyEntity,
    LockedDoorEntity,
    UnlockedDoorEntity,
    LavaEntity,
    BoxEntity,
    SpeedPowerUpEntity,
    ShieldPowerUpEntity,
    PhasingPowerUpEntity,
)
from dataclasses import dataclass
from typing import Callable
from collections import deque
import numpy as np
import torch
from grid_adventure.entities import create_agent_entity
from PIL import Image as PILImage
import heapq
from grid_adventure.grid import GridState
from grid_adventure.grid import to_state
from grid_adventure.grid import step
from grid_adventure.actions import Action
# Movements and Objectives are gridstate parameters. For Grid Adventure V1, we will be using the default ones.
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
# Rendering and display
from grid_adventure.rendering import ImageRenderer
from IPython.display import display
# Asset root for rendering. You can specify your own asset folder path if you want to use custom game assets.
from grid_adventure.rendering import DEFAULT_ASSET_ROOT
from grid_adventure.entities import (
    create_agent_entity
)
renderer = ImageRenderer(resolution=240, asset_root=DEFAULT_ASSET_ROOT)



## TEST TASK 1
# GridState is the main representation of the game state
TURN_LIMIT = 150

def _floors(gridstate: GridState) -> None:
    for y in range(gridstate.height):
        for x in range(gridstate.width):
            gridstate.add((x, y), FloorEntity())

def _border(gridstate: GridState) -> None:
    for x in range(gridstate.width):
        gridstate.add((x, 0), WallEntity())
        gridstate.add((x, gridstate.height - 1), WallEntity())
    for y in range(gridstate.height):
        gridstate.add((0, y), WallEntity())
        gridstate.add((gridstate.width - 1, y), WallEntity())

def build_custom_edge_level_1(seed: int = 113) -> GridState:
    gridstate = GridState(
        width=15,
        height=15,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=TURN_LIMIT,
    )
    _floors(gridstate)

    # Agent and Exit
    gridstate.add((1, 1), create_agent_entity(health=5))
    gridstate.add((13, 1), ExitEntity())

    # Walls

    # Pushable box

    # Gems
    gridstate.add((5, 1), GemEntity())

    # Coins

    # Powerups
    gridstate.add((4, 1), SpeedPowerUpEntity())

    # Key-door pair

    # Hazards
    for x in range(15):
        gridstate.add((x, 0), LavaEntity())
        gridstate.add((x, 2), LavaEntity())
    for y in range(15):
        gridstate.add((0, y), LavaEntity())
        gridstate.add((14, y), LavaEntity())
    gridstate.add((2, 1), LavaEntity())
    gridstate.add((3, 1), LavaEntity())
    gridstate.add((14, 1), LavaEntity())

    # Render and display the grid
    # img = renderer.render(to_state(gridstate))
    # img.show()

    return gridstate

# min score -219
def build_custom_edge_level_2(seed: int = 123) -> GridState:
    gridstate = GridState(
        width=15,
        height=15,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=150,
    )
    _floors(gridstate)
    _border(gridstate)

    # Maze Walls
    for y in range(2, 14):
        gridstate.add((4, y), WallEntity())
        gridstate.add((5, y), WallEntity())
    gridstate.add((5, 1), LavaEntity())
    gridstate.add((1, 6), WallEntity())
    gridstate.add((2, 6), WallEntity())
    gridstate.add((3, 6), BoxEntity())
    gridstate.add((2, 8), WallEntity())
    gridstate.add((3, 8), WallEntity())

    for y in range(1, 14):
        if y != 6:
            gridstate.add((8, y), WallEntity())

    # Horizontal Wall separating Phasing from Exit Room
    for x in range(9, 14):
        gridstate.add((x, 9), WallEntity())
    
    # Vertical Wall enclosing Exit Room from West
    for y in range(10, 14):
        if y != 13:
            gridstate.add((10, y), WallEntity())

    # Agent
    gridstate.add((1, 1), create_agent_entity(health=5))
    
    # Compartment 1 (x=1..3) -> Key
    gridstate.add((2, 13), KeyEntity())

    # Compartment 2 (Locked Room for Gem)
    gridstate.add((8, 6), LockedDoorEntity())
    gridstate.add((11, 3), GemEntity())
    gridstate.add((12, 3), LavaEntity())
    gridstate.add((11, 4), LavaEntity())
    gridstate.add((10, 3), LavaEntity())

    # Phasing Room (North of Exit Room)
    gridstate.add((13, 8), PhasingPowerUpEntity())
    gridstate.add((7, 13), PhasingPowerUpEntity())

    # Exit Room (South)
    gridstate.add((13, 10), ExitEntity())
    
    # Lava everywhere else in Exit Room
    for x in range(11, 14):
        for y in range(10, 14):
            if (x, y) != (13, 10):
                gridstate.add((x, y), LavaEntity())

    # Render and display the grid
    # img = renderer.render(to_state(gridstate))
    # img.show()

    return gridstate

def build_custom_edge_level_3(seed: int = 113) -> GridState:
    gridstate = GridState(
        width=15,
        height=15,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=TURN_LIMIT,
    )
    _floors(gridstate)

    # Agent and Exit
    gridstate.add((1, 1), create_agent_entity(health=5))
    gridstate.add((13, 1), ExitEntity())

    # Walls

    # Pushable box

    # Gems
    gridstate.add((5, 1), GemEntity())

    # Coins
    for x in range(1, 6):
        for y in range(10, 15):
            gridstate.add((x, y), CoinEntity())

    # Powerups
    gridstate.add((4, 1), SpeedPowerUpEntity())

    # Key-door pair

    # Hazards
    for x in range(15):
        gridstate.add((x, 0), LavaEntity())
        gridstate.add((x, 2), LavaEntity())
    for y in range(15):
        gridstate.add((0, y), LavaEntity())
        gridstate.add((14, y), LavaEntity())
    gridstate.add((2, 1), LavaEntity())
    gridstate.add((3, 1), LavaEntity())
    gridstate.add((14, 1), LavaEntity())

    # Render and display the grid
    # img = renderer.render(to_state(gridstate))
    # img.show()

    return gridstate

def build_custom_edge_level_4(seed: int = 42) -> GridState:
    """
    Stress test: 15x15 grid, high complexity to push anytime search to its limits.

    Key insight: powerup duration counters (speed/shield/phasing each 0-5 turns)
    multiply the state space by ~216x compared to binary-only objects.
    Combined with gems/keys/doors/boxes on a large grid, this creates
    millions of reachable states — too many for optimal search within 8s.

    Layout:
    - Agent at (1,1), HP=3, Exit at (13,13)
    - 3 gems in different quadrants (all required)
    - 2 key-door pairs controlling access between sections
    - 2 boxes blocking corridors
    - 1 speed + 1 shield + 1 phasing powerup (duration counters = state explosion)
    - Lava hazards near exit and mid-map (shield timing matters)
    - Maze walls with multiple routes (phasing/speed timing matters)

    Complexity = 3 gems + 2 keys + 2 doors + 2 boxes + 1 speed + 1 shield + 1 phasing = 12
    Grid size = 225, so epsilon = 3.0 (hardest tier)
    State space: 225 pos x 2^3 gems x 2^2 keys x 2^2 doors x box_pos x 6^3 powerups x HP ~ millions
    """
    gridstate = GridState(
        width=15,
        height=15,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=150,
    )
    _floors(gridstate)
    _border(gridstate)

    # === MAZE WALLS ===
    # Central vertical wall at x=7 (divides left x=1-6 / right x=8-13)
    for y in range(1, 14):
        if y != 3 and y != 10:  # gaps: locked door at y=3, open at y=10
            gridstate.add((7, y), WallEntity())

    # Central horizontal wall at y=7 (divides top y=1-6 / bottom y=8-13)
    for x in range(1, 14):
        if x != 3 and x != 12:  # gaps at x=3 (left) and locked door at x=12
            gridstate.add((x, 7), WallEntity())

    # Top-right internal wall at x=10 (creates corridor in top-right)
    for y in range(1, 6):
        if y != 3:  # gap at y=3
            gridstate.add((10, y), WallEntity())

    # Bottom-left internal wall at y=10 (restricts bottom-left access)
    for x in range(1, 7):
        if x != 4:  # gap at x=4
            gridstate.add((x, 10), WallEntity())

    # Small wall clusters creating corridors
    gridstate.add((3, 3), WallEntity())
    gridstate.add((3, 4), WallEntity())
    gridstate.add((5, 5), WallEntity())
    gridstate.add((5, 6), WallEntity())
    gridstate.add((11, 10), WallEntity())
    gridstate.add((11, 11), WallEntity())

    # === AGENT ===
    gridstate.add((1, 1), create_agent_entity(health=3))

    # === EXIT ===
    gridstate.add((13, 13), ExitEntity())

    # === LAVA HAZARDS (shield timing critical) ===
    # Lava near exit — shield helps survive
    gridstate.add((12, 12), LavaEntity())
    gridstate.add((13, 12), LavaEntity())
    gridstate.add((12, 13), LavaEntity())
    # Mid-map lava patch — shortcut through lava vs long detour
    gridstate.add((5, 8), LavaEntity())
    gridstate.add((5, 9), LavaEntity())
    # Right-side lava near top-right gem area
    gridstate.add((8, 4), LavaEntity())
    gridstate.add((8, 5), LavaEntity())

    # === GEMS (3 gems in different quadrants, all required) ===
    gridstate.add((2, 2), GemEntity())       # top-left, easy reach
    gridstate.add((12, 2), GemEntity())      # top-right, behind wall+door
    gridstate.add((2, 12), GemEntity())      # bottom-left, behind wall at y=10

    # === KEYS (2 keys) ===
    gridstate.add((5, 2), KeyEntity())       # top-left area
    gridstate.add((9, 11), KeyEntity())      # bottom-right area

    # === LOCKED DOORS (2 doors controlling section access) ===
    gridstate.add((7, 3), LockedDoorEntity())   # central wall gap — guards right half from top
    gridstate.add((12, 7), LockedDoorEntity())  # horizontal wall gap — guards bottom-right

    # === BOXES (2 boxes blocking corridors) ===
    gridstate.add((4, 2), BoxEntity())       # near top-left gem area
    gridstate.add((9, 9), BoxEntity())       # bottom-right corridor

    # === POWERUPS (duration counters = state space explosion) ===
    gridstate.add((6, 1), SpeedPowerUpEntity())     # top center — speed for long corridors
    gridstate.add((1, 8), ShieldPowerUpEntity())    # left side — shield for lava crossing
    gridstate.add((10, 6), PhasingPowerUpEntity())  # right side — phasing for wall bypass

    # img = renderer.render(to_state(gridstate))
    # img.show()

    return gridstate

# min score -270
def build_custom_edge_level_5(seed: int = 301) -> GridState:
    gridstate = GridState(
        width=15,
        height=15,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=TURN_LIMIT,
    )
    _floors(gridstate)

    # Agent and Exit
    gridstate.add((3, 3), create_agent_entity(health=5))
    gridstate.add((14, 12), ExitEntity())

    # Walls (black boxes only, not the outer border)
    wall_pos = [
        # top-left
        (2, 0), (4, 0), (3, 1),
        (6, 0), (7, 0), (8, 0),
        (6, 2),
        (6, 3), 
        (6, 4),
        (6, 5),
        (6, 6),
        (6, 7),
        (6, 8),

        # top-right / mid-right cluster
        (13, 0), (13, 2),
        (10, 3), (12, 3), (13, 3),
        (11, 4), (14, 4),
        (10, 5), (12, 5), (13, 5),

        # bottom-left structure
        (0, 10), (2, 10), (4, 10), (6, 10),
        (0, 11), (2, 11), (4, 11), (6, 11),
        (0, 12), (2, 12), (4, 12), (6, 12),
        (0, 13),
        (0, 14), (1, 14), (2, 14),

        # bottom-right structure
        (9, 10), (10, 10), (11, 10), (13, 10), (14, 10),
        (9, 11), (9, 12), (9, 13), (9, 14),
    ]
    for p in wall_pos:
        gridstate.add(p, WallEntity())

    # Lava (red boxes)
    lava_pos = [
        # top-left
        (1, 0), (0, 1), (1, 1),

        # lower-left pool
        (0, 8), (1, 8), (2, 8),
        (0, 9), (1, 9), (2, 9),

        # lower-right pool
        (11, 12), (12, 12), (13, 12),
        (11, 13), (12, 13), (13, 13), (14, 13),
        (11, 14), (12, 14), (13, 14),
    ]
    for p in lava_pos:
        gridstate.add(p, LavaEntity())

    # Boxes (green outlined boxes)
    box_pos = [
        (6, 1), (8, 1),
        (8, 3),
        (13, 1),
        (6, 9),
    ]
    for p in box_pos:
        gridstate.add(p, BoxEntity())

    # Locked doors (purple D)
    for p in [(9, 4), (12, 10)]:
        gridstate.add(p, LockedDoorEntity())

    # Keys (yellow K)
    for p in [(3, 0), (1, 10)]:
        gridstate.add(p, KeyEntity())

    # Gems (green G)
    for p in [(14, 0), (7, 1), (13, 4), (14, 14)]:
        gridstate.add(p, GemEntity())

    # Coins (yellow circles)
    for p in [(12, 4), (3, 10), (3, 11), (3, 12)]:
        gridstate.add(p, CoinEntity())

    # Phasing powerups (blue ghost)
    for p in [(0, 0), (10, 4), (5, 10)]:
        gridstate.add(p, PhasingPowerUpEntity())

    # Speed powerups (blue boots / B)
    for p in [(12, 2), (12, 7), (3, 4)]:
        gridstate.add(p, SpeedPowerUpEntity())

    # Shield powerup (blue shield)
    gridstate.add((10, 14), ShieldPowerUpEntity())

    # img = renderer.render(to_state(gridstate))
    # img.show()

    return gridstate

# build_custom_edge_level_1()
# min cost -219
# build_custom_edge_level_2()
# build_custom_edge_level_3()
# build_custom_edge_level_4()
# min cost -270
# build_custom_edge_level_5()

if __name__ == '__main__':
    from utils import evaluate_level, get_result_string
    # Run the custom edge case level!
    print("Testing Custom Case...")
    result = evaluate_level(
        Agent, 
        build_custom_edge_level_5, 
        observation_type='gridstate', 
        max_total_reward=-230, 
        min_total_reward=-345, 
        turn_limit=150, 
        time_limit=10, 
        seed=42
    )
    print(get_result_string(result))

### OTHERS
# Creating a 5x4 grid
# gridstate_2_2 = GridState(width=6, height=4, movement=MOVEMENTS["cardinal"], objective=OBJECTIVES["collect_gems_and_exit"], seed=1)
# for x in range(gridstate_2_2.width):
#     for y in range(gridstate_2_2.height):
#         gridstate_2_2.add((x, y), FloorEntity())
# wall_pos = [(0,0),(0,3),(1,0),(1,3),(2,3),(3,2),(3,3),(4,2),(4,3),(5,2),(5,3),]
# for p in wall_pos:
#     gridstate_2_2.add(p, WallEntity())
# gridstate_2_2.add((0, 1), AgentEntity())
# gridstate_2_2.add((2, 1), BoxEntity())
# gridstate_2_2.add((3, 1), CoinEntity())
# gridstate_2_2.add((4, 1), LavaEntity())
# gridstate_2_2.add((5, 1), ExitEntity())

# # Render and display the grid
# img = renderer.render(to_state(gridstate_2_2))
# img.show()

# # TODO: Fill up the list of actions
# agent = Agent()
# actions = []
# while not gridstate_2_2.win and not gridstate_2_2.lose:
#     action = agent.step(gridstate_2_2)
#     actions.append(action)
#     gridstate_2_2 = step(gridstate_2_2, action)
# img1 = renderer.render(to_state(gridstate_2_2))
# print("Actions taken:", actions)
# img1.show()

# if gridstate_2_2.win:
#     print("Congratulations! You have solved the grid. Total Score:", gridstate_2_2.score)
#     if gridstate_2_2.score == -16:
#         print("You have solved the grid in the optimal number of steps!")
#     else:
#         print("Try to solve the grid in fewer steps to achieve the optimal score.")
# else:
#     print("The grid is not yet solved. Keep trying!")

from pprint import pprint

from grid_adventure.levels.intro import (
    build_level_maze_turns, build_level_optional_coin, build_level_required_multiple, 
    build_level_key_door, build_level_hazard_detour, build_level_pushable_box, build_level_power_shield, 
    build_level_power_ghost, build_level_power_boots, build_level_combined_mechanics, build_level_boss
)
from grid_adventure.grid import to_state
from utils import evaluate, create_env, evaluate_level, get_result_string

# Asset root for rendering. You can specify your own asset folder path if you want to use custom game assets.
from grid_adventure.rendering import DEFAULT_ASSET_ROOT

GAMEPLAY_LEVEL_BUILDERS = [
    build_level_maze_turns, build_level_optional_coin, build_level_required_multiple, 
    build_level_key_door, build_level_hazard_detour, build_level_pushable_box, build_level_power_shield, 
    build_level_power_ghost, build_level_power_boots, build_level_combined_mechanics, build_level_boss
]

LEVEL_MAX_REWARD: dict[str, int] = {
    build_level_maze_turns.__name__: -27, 
    build_level_optional_coin.__name__: -21,
    build_level_required_multiple.__name__: -63,
    build_level_key_door.__name__: -33, 
    build_level_hazard_detour.__name__: -39,
    build_level_pushable_box.__name__: -21,
    build_level_power_shield.__name__: -42, 
    build_level_power_ghost.__name__: -48, 
    build_level_power_boots.__name__: -27,
    build_level_combined_mechanics.__name__: -84,
    build_level_boss.__name__: -68,
}

# Min reward is set to 50% worse than the optimal reward
LEVEL_MIN_REWARD: dict[str, int] = {k: v * 1.5 for k, v in LEVEL_MAX_REWARD.items()}

LEVEL_TURN_LIMIT: int = 150 # Allowable turns per test instance
TIME_LIMIT = 10 # Seconds per test instance
BOSS_TIME_LIMIT = 35 # Seconds per test instance

def evaluate_all_gameplay_levels(agent_class: type[Agent], observation_type: str = 'level', seed: int | list[int] = 42):
    if isinstance(seed, list):
        assert len(seed) == len(GAMEPLAY_LEVEL_BUILDERS), "If seed is a list, its length must match the number of levels."
    for i, builder in enumerate(GAMEPLAY_LEVEL_BUILDERS):
        max_total_reward, min_total_reward = LEVEL_MAX_REWARD[builder.__name__], LEVEL_MIN_REWARD[builder.__name__]
        seed_i = seed if isinstance(seed, int) else seed[i]
        yield evaluate_level(agent_class, builder, observation_type=observation_type, max_total_reward=max_total_reward, min_total_reward=min_total_reward, turn_limit=LEVEL_TURN_LIMIT, time_limit=TIME_LIMIT, seed=seed_i)

for result in evaluate_all_gameplay_levels(Agent, observation_type='gridstate'):
    print(get_result_string(result))



## TEST TASK 2
from PIL import Image
from grid_adventure.grid import from_state

env = create_env(build_level_required_multiple, observation_type='image')
img_obs, _ = env.reset()
image = img_obs["image"]
print(image.shape)
image = Image.fromarray(img_obs["image"])
display(image)
pprint(img_obs["info"])

# You can see the underlying gridstate of the env via env.state, but your agent won't have access to this!
print("The underlying `GridState` object is:") 
pprint(from_state(env.state), depth=1)

env = create_env(build_level_required_multiple, observation_type='image', seed=4)
img_obs, _ = env.reset()
image = Image.fromarray(img_obs["image"])
# image.show()
display(image)

# Specify a different seed to test the agent on a different looking grid
# for result in evaluate_all_gameplay_levels(Agent, observation_type='image', seed=list(range(1, len(GAMEPLAY_LEVEL_BUILDERS)+1))):
#     print(get_result_string(result))

# References: https://gemini.google.com/share/dd8b258f9a58
import random
def evaluate_and_debug_vision(agent_class: type[Agent], seeds_per_level: int | list[int] = 5):
    """
    Evaluates levels and automatically displays the grid image if the agent scores < 1.
    """
    
    # Setup the seed list
    if isinstance(seeds_per_level, int):
        seed_list = [random.randint(0, 99999) for _ in range(seeds_per_level)]
    else:
        seed_list = seeds_per_level

    for builder in GAMEPLAY_LEVEL_BUILDERS:
        level_name = builder.__name__
        max_total_reward = LEVEL_MAX_REWARD[level_name]
        min_total_reward = LEVEL_MIN_REWARD[level_name]
        
        for seed_val in seed_list:
            # 1. Run the evaluation
            result = evaluate_level(
                agent_class, 
                builder, 
                observation_type='image', 
                max_total_reward=max_total_reward, 
                min_total_reward=min_total_reward, 
                turn_limit=LEVEL_TURN_LIMIT, 
                time_limit=TIME_LIMIT, 
                seed=seed_val
            )
            
            # Note: Adjust how you extract 'performance' if 'result' is a dictionary or custom object.
            # Assuming 'result' directly returns the normalized score (e.g., 0.0 to 1.0).
            performance = result['performance']
            
            # 2. Check for vision failure
            if performance < 1.0:
                print(f"⚠️ Vision Failure Detected | {result}")
                print(get_result_string(result)) # Print your custom result string
                
                # 3. Recreate the specific environment that failed
                env = create_env(builder, observation_type='image', seed=seed_val)
                img_obs, _ = env.reset()
                
                # 4. Extract and show the image
                image = Image.fromarray(img_obs["image"])
                
                # .show() opens it in your OS default image viewer
                image.show() 
                
                
                print("-" * 50)
            else:
                print(f"✅ Success | {result}")

# Run the debugging loop (e.g., testing 3 random seeds per level)
# evaluate_and_debug_vision(Agent, seeds_per_level=1)


# TEST TASK 3
result = evaluate_level(
    Agent, 
    build_level_boss, 
    observation_type='gridstate', 
    max_total_reward=-68,
    min_total_reward=LEVEL_MIN_REWARD[build_level_boss.__name__],
    turn_limit=LEVEL_TURN_LIMIT, 
    time_limit=BOSS_TIME_LIMIT, 
    seed=1
)
print(get_result_string(result))

# result = evaluate_level(
#     Agent, 
#     build_level_boss, 
#     observation_type='image', 
#     max_total_reward=-68,
#     min_total_reward=LEVEL_MIN_REWARD[build_level_boss.__name__],
#     turn_limit=LEVEL_TURN_LIMIT, 
#     time_limit=BOSS_TIME_LIMIT, 
#     seed=2
# )
# print(get_result_string(result))
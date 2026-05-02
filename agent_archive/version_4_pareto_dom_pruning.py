# Input states for Agent step function
from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation

# State steppers
from grid_adventure.step import Action
from grid_adventure.grid import step as grid_step

# Movements and Objectives are gridstate parameters. For Grid Adventure V1, we will be using the default ones.
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES

#Next, we import the methods to create the various entities in the game.
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

# Utility helpers
from dataclasses import dataclass
from typing import Callable
from collections import deque

# Custom data structures
# Reference: https://docs.python.org/3/library/dataclasses.html
@dataclass(frozen=True)
class SearchState:
    agent_pos: tuple[int, int] | None
    agent_hp: int
    key_inventory: int
    speed_time_left: int
    shield_usage_left: int
    phasing_time_left: int
    coins: tuple[tuple[int, int], ...]
    gems: tuple[tuple[int, int], ...]
    keys: tuple[tuple[int, int], ...]
    boxes: tuple[tuple[int, int], ...]
    locked_doors: tuple[tuple[int, int], ...]
    speed_pos: tuple[tuple[int, int], ...]
    shield_pos: tuple[tuple[int, int], ...]
    phasing_pos: tuple[tuple[int, int], ...]

# Reference: https://share.google/aimode/RXU3vaCbjRdnVjJ4p
# Reference: https://share.google/aimode/eRUKo9Mjmjp5fkuBc
@dataclass(frozen=True)
class StaticMap:
    width: int
    height: int
    exit: tuple[int, int] | None
    walls: tuple[tuple[int, int], ...]
    lava: tuple[tuple[int, int], ...]
    num_coins: int
    wall_set: frozenset[tuple[int, int]] = frozenset()

# Utility classes (adapted from PS1)
import heapq
class Node:
    r"""Node class for search tree
    Args:
        parent (Node): the parent node of this node in the tree
        act (Action): the action taken from parent to reach this node
        state (GridState): the state of this node
        cost (float): the path cost of reaching this state
    """
    
    def __init__(
            self, 
            parent: "Node", 
            act, 
            state: GridState, 
            search_state: SearchState | None = None,
            cost: float = 0.0):

        self.parent = parent # where am I from
        self.act = act # how to get here
        self.state = state # who am I
        self.search_state = search_state # the search state associated with this node
        self.cost = cost # what it costs to be here

    def __str__(self):
        return str(self.state)

    def __lt__(self, node):
        """Compare the path cost between states"""
        return self.cost < node.cost

    def __eq__(self, node):
        """Compare whether two nodes have the same state"""
        return isinstance(node, Node) and self.state == node.state

    def __hash__(self):
        """Node can be used as a KeyValue"""
        return hash(self.state)
    
    def has_search_state(self):
        return self.search_state is not None

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def __contains__(self, node):
        """Decide whether the node (state) is in the queue"""
        return any([item == node for _, item in self.heap])

    def __delitem__(self, node):
        """Delete the an existing node in the queue"""
        try: 
            del self.heap[[item == node for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(node) + " is not in the queue")
        heapq.heapify(self.heap) # O(n)

    def __getitem__(self, node):
        """Return the priority of the given node in the queue"""
        for value, item in self.heap:
            if item == node:
                return value
        raise KeyError(str(node) + " is not in the queue")

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        string = '['
        for priority, node in self.heap:
            string += f"({priority}, {node}), "
        string += ']'
        return string

    def push(self, priority, node):
        """Enqueue node with priority"""
        heapq.heappush(self.heap, (priority, node))

    def pop(self):
        """Dequeue node with highest priority (the minimum one)"""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Empty priority queue")

    def get_priority(self, node):
        return self.__getitem__(node)

MOVES = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0)
}

class Agent:
    """Grid Adventure: Variant 1 agent template.

    This class is the single public interface that Coursemology will import and
    interact with when evaluating your submission. You should extend the
    internals (add helper classes / functions in other files if you wish) but
    MUST preserve:

    1. The class name: Agent
    2. The public method: step(self, state: GridState | ImageObservation) -> Action

    High‑level lifecycle per environment tick:
        state  --->  step(...)  --->  Action

    The "state" object type depends on the task:
    - Task 1: A fully structured GridState instance.
    - Task 2: An ImageObservation dictionary whose primary observation is an RGBA image
      plus limited structured metadata in the 'info' sub‑dict. In this case you
      typically perform perception to build (or approximate) an internal
      structured representation before planning.
    - Task 3: Input state could be either a GridState instance 
      or an ImageObservation dictionary

    Constraints:
    - Keep per‑step latency small (single CPU, ~1GB RAM). Avoid O(W*H) scans of
      the full grid every step.
    - Determinism helps reproducibility; seed your own RNG if you add any
      random components.

    You may add __init__ parameters (with defaults) if needed for your own
    development, but the grader will instantiate Agent() with no arguments.
    """
    
    def __init__(self):
        """Initialize your agent.

        Put all one‑time setup here (e.g., hardcoded ML model weights,
        precomputing heuristic tables). Keep it fast and memory‑light 
        to respect platform limits.
        """
        self.cached_actions = None # list[Action]
        self.cached_actions_index = None
        self.prev_action = None # Action
        self.prev_state = None # GridState
        self.heuristic_gem_mst_cache = {} # dict[tuple[tuple[int, int], ...], int] list of gems_pos -> mst cost
        self.cached_static_map = None # StaticMap
        self.dist_to_exit = None # dict[(x,y) -> int] exit as source
        self.sssp_cache = {} # dict[(x,y) -> dict[(x,y) -> int]] different source pos

    def step(self, state: GridState | ImageObservation) -> Action:
        """Return the next action given the current environment state.

        Parameters
        ----------
        state : GridState | ImageObservation
            - If a GridState instance (Tasks 1 and 3): you have direct, structured
              access to grid, entities, objective message, score, etc.
            - If a ImageObservation dict (Tasks 2 and 3): contains 'image' (H×W×4 RGBA uint8)
              plus 'info' sub‑dictionary (agent stats, partial config, message).
              You likely need to parse the image into an internal representation.

        Returns
        -------
        Action
            A valid action from the Action enum. Must always return a member;
            never return None.
        """
        # Reference: https://share.google/aimode/IWyPP9kZB0CJoUa1l
        if isinstance(state, dict) and 'image' in state:
            state = self.parse_img(state)  # Convert image observation to GridState
        
        # Use cached actions if available
        prev_state = self.prev_state
        prev_action = self.prev_action
        if self.cached_actions is not None:
            self.cached_actions_index += 1
            if self.cached_actions_index < len(self.cached_actions):
                action = self.cached_actions[self.cached_actions_index]
                self.prev_state = state
                self.prev_action = action
                # Debug
                # print(f"Using cached action: {action}")
                return action
            self.cached_actions = None
            self.cached_actions_index = None

        # Cache static map and exit sssp
        if self.cached_static_map is None:
            static_map = self.parse_static_map(state)
            self.cached_static_map = static_map
            self.dist_to_exit = self.sssp_dist(static_map.exit, static_map)
            self.sssp_cache[static_map.exit] = self.dist_to_exit
        static_map = self.cached_static_map
        result = self.astar_search(state, self.heuristic_func, static_map)
        if result is False:
            return Action.WAIT

        actions, _ = result
        if not actions:
            return Action.WAIT
        
        self.cached_actions_index = 0
        self.cached_actions = actions
        self.prev_action = self.cached_actions[self.cached_actions_index]
        self.prev_state = state

        return actions[self.cached_actions_index]

    ## PARSE METHODS
    
    def parse_img(self, observation: ImageObservation) -> GridState:
        """Parse image observation into GridState representation.

        NOTE: This method is optional and intended for debugging in Grid Play only. 
        You do not need to implement it for Coursemology submission, but it can be 
        helpful for visualizing your agent's perception during development. 
        Implementing this method will not affect grading.

        Parameters
        ----------
        observation : ImageObservation
            The raw image observation and metadata from the environment.

        Returns
        -------
        GridState
            The reconstructed internal representation of the environment state.
        """
        # Placeholder: implement perception logic here
        pass

    def parse_static_map(self, state: GridState) -> StaticMap:
        width = state.width
        height = state.height
        exit = None
        walls = []
        lava = []
        num_coins = 0
        for x in range(state.width):
            for y in range(state.height):
                pos = (x, y)
                entities = state.objects_at(pos)
                for e in entities:
                    if isinstance(e, ExitEntity):
                        exit = pos
                    elif isinstance(e, WallEntity):
                        walls.append(pos)
                    elif isinstance(e, LavaEntity):
                        lava.append(pos)
                    elif isinstance(e, CoinEntity):
                        num_coins += 1
        walls_tuple = tuple(sorted(walls))
        return StaticMap(
            width=width,
            height=height,
            exit=exit,
            walls=walls_tuple,
            lava=tuple(sorted(lava)),
            num_coins=num_coins,
            wall_set=frozenset(walls),
        )
    
    def parse_search_state(self, state: GridState) -> SearchState:
        agent_pos = None
        agent_hp = 0
        key_inventory = 0
        speed_time_left = 0
        shield_usage_left = 0
        phasing_time_left = 0
        coins = []
        gems = []
        keys = []
        boxes = []
        locked_doors = []
        speed_pos = []
        shield_pos = []
        phasing_pos = []

        for x in range(state.width):
            for y in range(state.height):
                pos = (x, y)
                entities = state.objects_at(pos)
                for e in entities:
                    if isinstance(e, AgentEntity):
                        agent_pos = pos
                        agent_hp = e.health.current_health
                        key_inventory = self.count_inventory_keys(e)
                        speed_time_left, shield_usage_left, phasing_time_left = self.count_active_powerups(e)
                    elif isinstance(e, CoinEntity):
                        coins.append(pos)
                    elif isinstance(e, GemEntity):
                        gems.append(pos)
                    elif isinstance(e, KeyEntity):
                        keys.append(pos)
                    elif isinstance(e, BoxEntity):
                        boxes.append(pos)
                    elif isinstance(e, LockedDoorEntity):
                        locked_doors.append(pos)
                    elif isinstance(e, SpeedPowerUpEntity):
                        speed_pos.append(pos)
                    elif isinstance(e, ShieldPowerUpEntity):
                        shield_pos.append(pos)
                    elif isinstance(e, PhasingPowerUpEntity):
                        phasing_pos.append(pos)
        return SearchState(
            agent_pos=agent_pos,
            agent_hp=agent_hp,
            key_inventory=key_inventory,
            speed_time_left=speed_time_left,
            shield_usage_left=shield_usage_left,
            phasing_time_left=phasing_time_left,
            coins=tuple(sorted(coins)),
            gems=tuple(sorted(gems)),
            keys=tuple(sorted(keys)),
            boxes=tuple(sorted(boxes)),
            locked_doors=tuple(sorted(locked_doors)),
            speed_pos=tuple(sorted(speed_pos)),
            shield_pos=tuple(sorted(shield_pos)),
            phasing_pos=tuple(sorted(phasing_pos))
        )

    def derive_search_state(self, 
                            new_grid_state: GridState, 
                            parent_search_state: SearchState, 
                            action: Action, 
                            static_map: StaticMap
                            ) -> SearchState:
        # Initialise new search state variables
        s = parent_search_state
        new_agent_pos = s.agent_pos
        new_agent_hp = s.agent_hp
        new_key_inventory = s.key_inventory
        new_speed_time_left = s.speed_time_left
        new_shield_usage_left = s.shield_usage_left
        new_phasing_time_left = s.phasing_time_left
        new_coins = s.coins
        new_gems = s.gems
        new_keys = s.keys
        new_boxes = s.boxes
        new_locked_doors = s.locked_doors
        new_speed_pos = s.speed_pos
        new_shield_pos = s.shield_pos
        new_phasing_pos = s.phasing_pos

        # Get from new_grid_state
        (pos_x, pos_y) = s.agent_pos
        if s.speed_time_left > 0:
            range_bound = range(-2, 3)
        else:
            range_bound = range(-1, 2)
        found = False
        for diff_x in range_bound:
            if found:
                break
            for diff_y in range_bound:
                check_pos = (pos_x + diff_x, pos_y + diff_y)
                if not (0 <= check_pos[0] < static_map.width and 0 <= check_pos[1] < static_map.height):
                    continue
                entities = new_grid_state.objects_at(check_pos)
                for e in entities:
                    if isinstance(e, AgentEntity):
                        new_agent_pos = check_pos
                        new_agent_hp = e.health.current_health
                        new_key_inventory = self.count_inventory_keys(e)
                        new_speed_time_left, new_shield_usage_left, new_phasing_time_left = self.count_active_powerups(e)
                        found = True
                        break

        # Self-derive
        if action == Action.WAIT:
            pass

        elif action == Action.USE_KEY:
            new_locked_doors = list(s.locked_doors)
            # Unlocks min(keys, surrounding_doors) simultaneously
            # Priority: CURRENT, LEFT, RIGHT, UP, DOWN
            for diff_x, diff_y in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                check_pos = (pos_x + diff_x, pos_y + diff_y)
                if not (0 <= check_pos[0] < static_map.width and 0 <= check_pos[1] < static_map.height):
                    continue
                if check_pos in new_locked_doors:
                    for e in new_grid_state.objects_at(check_pos):
                        if isinstance(e, UnlockedDoorEntity):
                            new_locked_doors.remove(check_pos)
                            break
            new_locked_doors = tuple(new_locked_doors)
        
        # Update position of pickables
        elif action == Action.PICK_UP:
            pos = new_agent_pos
            if pos in s.coins:
                new_coins = tuple(p for p in s.coins if p != pos)
            elif pos in s.gems:
                new_gems = tuple(p for p in s.gems if p != pos)
            elif pos in s.keys:
                new_keys = tuple(p for p in s.keys if p != pos)
            elif pos in s.speed_pos:
                new_speed_pos = tuple(p for p in s.speed_pos if p != pos)
            elif pos in s.shield_pos:
                new_shield_pos = tuple(p for p in s.shield_pos if p != pos)
            elif pos in s.phasing_pos:
                new_phasing_pos = tuple(p for p in s.phasing_pos if p != pos)

        # Check for boxes
        elif action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            if s.boxes and new_agent_pos != s.agent_pos:
                move_x, move_y = MOVES[action]
                new_boxes_list = []
                for box_pos in s.boxes:
                    # Check if still at old position
                    still_there = any(isinstance(e, BoxEntity) for e in new_grid_state.objects_at(box_pos))
                    if still_there:
                        new_boxes_list.append(box_pos)
                    else:
                        # Box moved, search 1 or 2 tiles in movement direction
                        for dist in range(1, 3):
                            new_pos = (box_pos[0] + move_x * dist, box_pos[1] + move_y * dist)
                            if 0 <= new_pos[0] < static_map.width and 0 <= new_pos[1] < static_map.height:
                                if any(isinstance(e, BoxEntity) for e in new_grid_state.objects_at(new_pos)):
                                    new_boxes_list.append(new_pos)
                                    break
                new_boxes = tuple(sorted(new_boxes_list))
        
        return SearchState(
            agent_pos = new_agent_pos,
            agent_hp = new_agent_hp,
            key_inventory = new_key_inventory,
            speed_time_left = new_speed_time_left,
            shield_usage_left = new_shield_usage_left,
            phasing_time_left = new_phasing_time_left,
            coins = new_coins,
            gems = new_gems,
            keys = new_keys,
            boxes = new_boxes,
            locked_doors = new_locked_doors,
            speed_pos = new_speed_pos,
            shield_pos = new_shield_pos,
            phasing_pos = new_phasing_pos
        )
    
    # If need to reduce for dict lookup (Replace by dominate)
    def reduce_search_state(self, search_state: SearchState):
        return (
            search_state.agent_pos,
            search_state.agent_hp,
            search_state.key_inventory,
            search_state.speed_time_left,
            search_state.shield_usage_left,
            search_state.phasing_time_left,
            # Prune coins as heuristic handles it, and detour is never profitable
            search_state.gems,
            # Convert keys to len because any key can unlock any door
            # Astar will choose the key collected with lower cost
            # Prune key as number is stored in key_inventory
            search_state.boxes,
            search_state.locked_doors,
            # Prune speed as its position only leads to small wastage in score if missed
            search_state.shield_pos,
            search_state.phasing_pos
        )

    # For Pareto-Dominance Pruning
    # Reference: https://gemini.google.com/share/1d9a62163c80
    def dominates(self, state_a: SearchState, cost_a: int, state_b: SearchState, cost_b: int) -> bool:
        """Returns True if state_a strictly dominates state_b at the same agent_pos"""
        if cost_a > cost_b: return False
        if state_a.agent_hp < state_b.agent_hp: return False
        if state_a.key_inventory < state_b.key_inventory: return False
        # For keys, it does not matter where or which keys are collected
        if state_a.speed_time_left < state_b.speed_time_left: return False
        if state_a.shield_usage_left < state_b.shield_usage_left: return False
        if state_a.phasing_time_left < state_b.phasing_time_left: return False
        # Ensure boxes at same pos
        if state_a.boxes != state_b.boxes: return False
        # Remaining gems of A <= remaining gems of B
        if len(state_a.gems) > len(state_b.gems): return False
        # Non-collected keys in A must not be collected in B also
        if any(g not in state_b.gems for g in state_a.gems): return False
        # Remaining locked doors of A <= remaining locked doors of B
        if len(state_a.locked_doors) > len(state_b.locked_doors): return False
        if any(d not in state_b.locked_doors for d in state_a.locked_doors): return False
        # Remaining powerups of A >= remaining powerups of B for same score
        if len(state_a.speed_pos) < len(state_b.speed_pos): return False
        if any(p not in state_a.speed_pos for p in state_b.speed_pos): return False
        if len(state_a.shield_pos) < len(state_b.shield_pos): return False
        if any(p not in state_a.shield_pos for p in state_b.shield_pos): return False
        if len(state_a.phasing_pos) < len(state_b.phasing_pos): return False
        if any(p not in state_a.phasing_pos for p in state_b.phasing_pos): return False

        return True


    ## SEARCH METHODS

    # Pareto-Dominance Pruning A* Search
    # Reference: https://www.emergentmind.com/topics/dynamic-programming-planner-with-pareto-pruning
    def astar_search(self, state: GridState, heuristic_func: Callable, static_map: StaticMap) -> list[Action] | bool:
        r"""
        A* Search finds the solution to reach the goal from the initial.
        If no solution is found, return False.
        Uses pareto frontier to prune dominated states.
        
        Args:
            state (GridState): GridState instance
            heuristic_func (Callable): heuristic function for the A* search

        Returns:
            solution (List[Action]): the action sequence
            num_nodes_expand (int): the number of nodes expanded during the search
            OR
            False: if no solution is found
        """
        fail = True
        solution = []

        # Reached tracks the Pareto frontier of pos
        reached = {} # dict[tuple[int, int], list[tuple[SearchState, int]]]
        frontier = PriorityQueue()
        initial = state
        initial_search_state = self.parse_search_state(initial)
        agent_pos = initial_search_state.agent_pos
        reached[agent_pos] = [(initial_search_state, 0)]
        curr = Node(parent=None, 
                    act=None, 
                    state=initial,
                    search_state=initial_search_state,
                    cost=0)
        num_nodes_expand = 0
        # Adaptive weight that changes depending on the complexity of the search state (number of entities).
        complexity = (len(initial_search_state.gems)
                      + len(initial_search_state.keys)
                      + len(initial_search_state.locked_doors) 
                      + len(initial_search_state.boxes)
                      + len(initial_search_state.speed_pos)
                      + len(initial_search_state.shield_pos)
                      + len(initial_search_state.phasing_pos))
        ASTAR_WEIGHT = 1.0
        if complexity > 4:
            ASTAR_WEIGHT = 1.5
        if complexity > 8:
            ASTAR_WEIGHT = 2.0
            
        frontier.push((curr.cost + ASTAR_WEIGHT * heuristic_func(static_map, initial_search_state)), curr)

        while frontier.__len__() > 0:
            next_node = frontier.pop() # next_node is Node instance
            if not next_node.has_search_state():
                next_search_state = self.parse_search_state(next_node.state) # state is GridState instance
            else:
                next_search_state = next_node.search_state
            
            agent_pos = next_search_state.agent_pos

            # Check if node was dominated
            is_dominated = False
            for r_state, r_cost in reached.get(agent_pos, []):
                if r_state is not next_search_state and self.dominates(r_state, r_cost, next_search_state, next_node.cost):
                    is_dominated = True
                    break
            if is_dominated:
                continue
            if next_node.state.win:
                curr = next_node
                fail = False
                break
            elif next_node.state.lose:
                continue
            else:
                for child_node in self.expand(next_node, next_search_state, static_map):
                    num_nodes_expand += 1
                    if child_node is None:
                        continue
                    child_search_state = child_node.search_state
                    child_pos = child_search_state.agent_pos
                    if child_pos not in reached:
                        reached[child_pos] = []
                    
                    # Check if child node was dominated
                    is_child_dominated = False
                    for r_state, r_cost in reached[child_pos]:
                        if self.dominates(r_state, r_cost, child_search_state, child_node.cost):
                            is_child_dominated = True
                            break
                    if is_child_dominated:
                        continue
                    # A new pareto front
                    reached[child_pos] = [
                        (s, c) for (s, c) in reached[child_pos] 
                        if not self.dominates(child_search_state, child_node.cost, s, c)
                    ]
                    reached[child_pos].append((child_search_state, child_node.cost))
                    frontier.push((child_node.cost + ASTAR_WEIGHT * heuristic_func(static_map, child_search_state)), child_node)

        if not fail:
            while curr.parent:
                solution.append(curr.act)
                curr = curr.parent
        solution.reverse()

        if fail:
            return False
        return solution, num_nodes_expand

    def heuristic_func(self, static_map: StaticMap, search_state: SearchState) -> float:
        agent = search_state.agent_pos
        dist_exit = self.dist_to_exit  # precomputed SSSP from exit
        speed_time = search_state.speed_time_left
        speed_pos_count = len(search_state.speed_pos)
        has_phasing = len(search_state.phasing_pos) > 0 or search_state.phasing_time_left > 0

        # No gems: move to exit
        if len(search_state.gems) == 0:
            # If phasing on, all walls are gone
            if has_phasing:
                agent_to_exit = self.manhattan(agent, static_map.exit)
            else:
                agent_to_exit = dist_exit.get(agent, self.manhattan(agent, static_map.exit))
            movement_turns = self.lower_bound(agent_to_exit, speed_time, speed_pos_count)
            # Use cost for heuristic
            turn_cost = max(0, movement_turns - 1) * 3 # winning step free
            # A coin +5, 1 turn to PICK_UP -3 = nett +2
            coin_benefit = len(search_state.coins) * 2
            return max(0, turn_cost - coin_benefit)
        
        # Have gems
        # Min distance from gem to exit (SSSP from exit)
        if has_phasing:
            min_gem_to_exit = min(self.manhattan(g, static_map.exit) for g in search_state.gems)
            min_agent_to_gem = min(self.manhattan(agent, g) for g in search_state.gems)
        else:
            min_gem_to_exit = min(dist_exit.get(g, self.manhattan(g, static_map.exit)) for g in search_state.gems)
            dist_agent = self.get_sssp(agent, static_map)
            min_agent_to_gem = min(dist_agent.get(g, self.manhattan(agent, g)) for g in search_state.gems)

        # Min distance between gems (MST with SSSP)
        gem_mst_distance = 0
        if len(search_state.gems) > 1:
            cache_key = tuple(search_state.gems) + (has_phasing,)
            if cache_key in self.heuristic_gem_mst_cache:
                gem_mst_distance = self.heuristic_gem_mst_cache[cache_key]
            else:
                gem_mst_distance = self.mst_cost(search_state.gems, static_map, has_phasing)
                self.heuristic_gem_mst_cache[cache_key] = gem_mst_distance
        total_dist = min_agent_to_gem + gem_mst_distance + min_gem_to_exit
        movement_turns = self.lower_bound(total_dist, speed_time, speed_pos_count)
        gem_pickup_turns = len(search_state.gems)
        total_min_turns = movement_turns + gem_pickup_turns

        # Use cost for heuristic
        turn_cost = max(0, total_min_turns - 1) * 3 # winning step free
        coin_benefit = len(search_state.coins) * 2
        return max(0, turn_cost - coin_benefit)
    

    ## DISTANCE ESTIMATION METHODS

    def manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Either from cache or compute
    def get_sssp(self, pos: tuple[int, int], static_map: StaticMap) -> dict[tuple[int, int], int]:
        if pos not in self.sssp_cache:
            self.sssp_cache[pos] = self.sssp_dist(pos, static_map)
        return self.sssp_cache[pos]

    # Adapted from CS2040
    def sssp_dist(self, source: tuple[int, int], static_map: StaticMap) -> dict[tuple[int, int], int]:
        """Dijkstra SSSP from source. 
        Returns dict mapping positions to shortest distance.
        Walls block movement."""
        wall_set = static_map.wall_set
        dist = {source: 0}
        visited = set()
        pq = PriorityQueue()
        pq.push(0, source)
        movements = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        while pq.__len__() > 0:
            pos = pq.pop()
            if pos in visited:
                continue
            visited.add(pos)
            d = dist[pos]
            for move_x, move_y in movements:
                new_pos = (pos[0] + move_x, pos[1] + move_y)
                if not (0 <= new_pos[0] < static_map.width and 0 <= new_pos[1] < static_map.height):
                    continue
                if new_pos in wall_set:
                    continue
                new_d = d + 1  # uniform cost
                if new_d < dist.get(new_pos, float('inf')):
                    dist[new_pos] = new_d
                    pq.push(new_d, new_pos)
        return dist
    
    # Minimum spanning tree cost for gems (using SSSP), adapted from CS2040
    def mst_cost(self, gems_pos: tuple[tuple[int, int], ...], static_map: StaticMap, has_phasing: bool = False) -> int:
        if len(gems_pos) <= 1:
            return 0
        # Precompute SSSP from each gem for pairwise distances
        gem_dists = {}
        for gem in gems_pos:
            if has_phasing:
                gem_dists[gem] = {g: self.manhattan(gem, g) for g in gems_pos}
            else:
                gem_dists[gem] = self.get_sssp(gem, static_map)
        # Prim's algorithm  
        gems_pos = list(gems_pos)
        pos_in_mst = set()
        best_distance = {}
        pq = PriorityQueue()
        start = gems_pos[0]
        best_distance[start] = 0
        pq.push(0, start)
        total_cost = 0

        while pq.__len__() > 0:
            current_pos = pq.pop()
            if current_pos in pos_in_mst:
                continue
            pos_in_mst.add(current_pos)
            total_cost += best_distance[current_pos]

            cur_dists = gem_dists[current_pos]
            for pos in gems_pos:
                if pos in pos_in_mst:
                    continue
                distance = cur_dists.get(pos, self.manhattan(current_pos, pos))
                if pos not in best_distance or distance < best_distance[pos]:
                    best_distance[pos] = distance
                    if pos in pq:
                        del pq[pos]
                    pq.push(distance, pos)

        return total_cost

    ## UTILITY METHODS 
    
    def lower_bound(self, distance: int, speed_time_left: int = 0, speed_pos_count: int = 0) -> int:
        # Each speed double 5 turns (max 3)
        total_speed_turns = speed_time_left + (speed_pos_count * 5)
        # Speed distance /2 cost
        if total_speed_turns > 0:
            speed_dist = min(distance, total_speed_turns * 2)
            normal_dist = distance - speed_dist
            return (speed_dist + 1) // 2 + normal_dist
        return distance
    
    def expand(self, node: Node, search_state: SearchState, static_map: StaticMap) -> list[Node]:
        child_nodes = []
        agent_pos = search_state.agent_pos
        for action in Action:
            if action == Action.WAIT:
                continue
            elif action == Action.USE_KEY:
                if search_state.key_inventory <= 0:
                    continue
                # Check distance between agent and locked doors <= 1
                can_unlock = False
                for locked_door_pos in search_state.locked_doors:
                    if self.manhattan(agent_pos, locked_door_pos) <= 1:
                        can_unlock = True
                        break
                if not can_unlock:
                    continue
            elif action == Action.PICK_UP:
                # Skip if not on pickable
                if agent_pos not in (
                    search_state.coins 
                    + search_state.gems 
                    + search_state.keys
                    + search_state.speed_pos
                    + search_state.shield_pos
                    + search_state.phasing_pos
                ):
                    continue
            elif action in MOVES:
                # Skip if target cell is out of grid
                move_x, move_y = MOVES[action]
                target = (agent_pos[0] + move_x, agent_pos[1] + move_y)
                if not (0 <= target[0] < static_map.width and 0 <= target[1] < static_map.height):
                    continue

            new_state = grid_step(node.state, action)
            if new_state is not None:
                child_search_state = self.derive_search_state(new_state, search_state, action, static_map)
                # Score-based cost: 3 per turn, winning turn free, coin pickup gives +5
                step_cost = 0 if new_state.win else 3
                if action == Action.PICK_UP and search_state.agent_pos in search_state.coins:
                    step_cost -= 5
                child_node = Node(parent=node, 
                                  act=action, 
                                  state=new_state, 
                                  search_state=child_search_state,
                                  cost=node.cost + step_cost)
                child_nodes.append(child_node)
        return child_nodes

    def count_inventory_keys(self, agent_entity: AgentEntity) -> int:
        count = 0
        for item in agent_entity.inventory_list:
            if isinstance(item, KeyEntity):
                count += 1
        return count
    
    def count_coins(self, search_state: SearchState, static_map: StaticMap) -> int:
        initial_num_coins = static_map.num_coins
        current_num_coins = len(search_state.coins)
        return initial_num_coins - current_num_coins

    def count_active_powerups(self, agent_entity: AgentEntity) -> tuple[int, int, int]:
        speed = 0
        shield = 0
        phasing = 0
        for status in agent_entity.status_list:
            if isinstance(status, SpeedPowerUpEntity):
                speed = max(speed, status.time_limit.amount)
            elif isinstance(status, ShieldPowerUpEntity):
                shield = max(shield, status.usage_limit.amount)
            elif isinstance(status, PhasingPowerUpEntity):
                phasing = max(phasing, status.time_limit.amount)
        return speed, shield, phasing
    
    def info(self) -> dict[str, str]:
        """Return info about the agent.

        NOTE: This method is optional and intended for debugging in Grid Play only. 
        You do not need to implement it for Coursemology submission, but it can be 
        helpful for visualizing your agent's internal state during development. 
        Implementing this method will not affect grading.
        """
        # Optional: return info about the agent
        return {"name": "Random AI Agent"}
    




### TEST CASES
# GridState is the main representation of the game state
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
# Default renderer used throughout the notebook unless overridden in a cell
renderer = ImageRenderer(resolution=240, asset_root=DEFAULT_ASSET_ROOT)
TURN_LIMIT = 50

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
    img = renderer.render(to_state(gridstate))
    img.show()

    return gridstate

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
    img = renderer.render(to_state(gridstate))
    img.show()

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
    img = renderer.render(to_state(gridstate))
    img.show()

    return gridstate

# build_custom_edge_level_2()

# if __name__ == '__main__':
#     from utils import evaluate_level, get_result_string
#     # Run the custom edge case level!
#     print("Testing Custom Case...")
#     result = evaluate_level(
#         Agent, 
#         build_custom_edge_level_2, 
#         observation_type='gridstate', 
#         max_total_reward=-225, 
#         min_total_reward=-337, 
#         turn_limit=150, 
#         time_limit=10, 
#         seed=42
#     )
#     print(get_result_string(result))

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
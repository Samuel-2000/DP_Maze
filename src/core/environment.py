# src/core/environment.py
"""
Optimized Maze Environment using Gymnasium
"""

import numpy as np
import cv2
import gymnasium as gym  # Changed from gym to gymnasium
from gymnasium import spaces
from numba import jit
import numba
from typing import Tuple, Dict, Any, Optional
import enum


class TileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    FOOD_SOURCE = 2
    FOOD = 3
    AGENT = 4


class Actions(enum.Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4
    START = 5


@jit(nopython=True)
def add_obstacles_connectivity(grid: np.ndarray, n_obstacles: int) -> np.ndarray:
    """Add obstacles while maintaining connectivity (Numba optimized)"""
    h, w = grid.shape
    total_cells = h * w
    
    # Get empty cells
    empty_ids = np.empty(total_cells, dtype=np.int32)
    count = 0
    for idx in range(total_cells):
        if grid[idx // w, idx % w] == 0:
            empty_ids[count] = idx
            count += 1
    
    if n_obstacles > count - 1:
        n_obstacles = count - 1
    
    # BFS arrays
    visited = np.zeros(total_cells, dtype=np.uint8)
    queue = np.empty(total_cells, dtype=np.int32)
    
    added = 0
    for _ in range(n_obstacles):
        for _ in range(count):
            pick = np.random.randint(0, count)
            cell = empty_ids[pick]
            r, c = cell // w, cell % w
            
            # Try placing obstacle
            grid[r, c] = 1
            
            # Find start for BFS
            start = -1
            for j in range(count):
                if j == pick:
                    continue
                nid = empty_ids[j]
                rr, cc = nid // w, nid % w
                if grid[rr, cc] == 0:
                    start = nid
                    break
            
            if start < 0:
                grid[r, c] = 0
                continue
            
            # BFS
            visited[:] = 0
            head = tail = 0
            visited[start] = 1
            queue[tail] = start
            tail += 1
            reach = 1
            
            while head < tail:
                cur = queue[head]
                head += 1
                cr, cc = cur // w, cur % w
                
                # Check neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    nid = nr * w + nc
                    if (0 <= nr < h and 0 <= nc < w and 
                        grid[nr, nc] == 0 and visited[nid] == 0):
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
            
            if reach == count - 1:
                # Success
                empty_ids[pick] = empty_ids[count - 1]
                count -= 1
                added += 1
                break
            else:
                grid[r, c] = 0
    
    return grid


@jit(nopython=True)
def food_step_optimized(agent_y: int, agent_x: int, 
                       food_sources: np.ndarray, 
                       food_energy: float) -> float:
    """Optimized food step processing"""
    energy_gained = 0.0
    for i in range(food_sources.shape[0]):
        y, x, time_left, has_food = food_sources[i]
        
        if agent_y == y and agent_x == x and has_food:
            energy_gained += food_energy
            food_sources[i, 2] = np.random.randint(5, 15)  # Regeneration time
            food_sources[i, 3] = 0
        elif time_left == 0:
            food_sources[i, 3] = 1
        elif time_left > 0:
            food_sources[i, 2] = time_left - 1
    
    return energy_gained


class GridMazeWorld(gym.Env):
    """Optimized Grid Maze Environment"""
    
    def __init__(self, 
                 grid_size: int = 11,
                 max_steps: int = 100,
                 obstacle_fraction: float = 0.25,
                 n_food_sources: int = 4,
                 food_energy: float = 10.0,
                 initial_energy: float = 30.0,
                 energy_decay: float = 0.98,
                 energy_per_step: float = 0.1,
                 render_size: int = 512):
        
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_food_sources = n_food_sources
        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.energy_per_step = energy_per_step
        self.render_size = render_size
        
        # Calculate obstacle count
        self.n_obstacles = int((grid_size - 2) ** 2 * obstacle_fraction)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        
        # Initialize state
        self.grid = None
        self.food_sources = None
        self.agent_pos = None
        self.energy = None
        self.steps = None
        self.done = None
        self.last_action = None
        
        # Colors for rendering
        self.colors = {
            TileType.EMPTY: (40, 40, 40),
            TileType.OBSTACLE: (100, 100, 100),
            TileType.FOOD_SOURCE: (200, 50, 50),
            TileType.FOOD: (50, 200, 50),
            TileType.AGENT: (50, 50, 200)
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create grid with borders
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[0, :] = TileType.OBSTACLE.value
        self.grid[-1, :] = TileType.OBSTACLE.value
        self.grid[:, 0] = TileType.OBSTACLE.value
        self.grid[:, -1] = TileType.OBSTACLE.value
        
        # Add obstacles
        self.grid = add_obstacles_connectivity(self.grid, self.n_obstacles)
        
        # Initialize food sources
        self._init_food_sources()
        
        # Place agent
        empty_cells = np.argwhere(self.grid == TileType.EMPTY.value)
        self.agent_pos = empty_cells[np.random.choice(len(empty_cells))]
        
        # Reset state
        self.energy = self.initial_energy
        self.steps = 0
        self.done = False
        self.last_action = Actions.START.value
        
        info = {
            'energy': self.energy,
            'steps': self.steps,
            'position': self.agent_pos.copy()
        }
        
        return self._get_observation(), info
    
    def _init_food_sources(self):
        """Initialize food sources"""
        empty_cells = np.argwhere(self.grid == TileType.EMPTY.value)
        indices = np.random.choice(len(empty_cells), self.n_food_sources, replace=False)
        
        self.food_sources = np.zeros((self.n_food_sources, 4), dtype=np.int32)
        for i, idx in enumerate(indices):
            y, x = empty_cells[idx]
            regen_time = np.random.randint(5, 15)
            self.food_sources[i] = [y, x, regen_time, 1]  # Start with food
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        if self.done:
            obs = self._get_observation()
            return obs, 0.0, True, True, {}
        
        # Update agent position
        new_pos = self._move_agent(action)
        
        # Process food
        energy_gained = food_step_optimized(
            self.agent_pos[0], self.agent_pos[1],
            self.food_sources, self.food_energy
        )
        
        # Update energy
        self.energy = (self.energy * self.energy_decay + 
                      energy_gained - self.energy_per_step)
        self.energy = max(0.0, min(self.energy, 100.0))
        
        # Update state
        self.steps += 1
        self.last_action = action
        
        # Check termination
        terminated = (self.steps >= self.max_steps or self.energy <= 0)
        truncated = False  # Time limit handled by max_steps
        self.done = terminated or truncated
        
        # Calculate reward
        reward = self._calculate_reward(energy_gained)
        
        info = {
            'energy': self.energy,
            'steps': self.steps,
            'position': self.agent_pos.copy(),
            'food_collected': energy_gained > 0
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _move_agent(self, action: int) -> np.ndarray:
        """Move agent based on action"""
        y, x = self.agent_pos
        
        if action == Actions.LEFT.value and x > 0 and self.grid[y, x-1] == 0:
            x -= 1
        elif action == Actions.RIGHT.value and x < self.grid_size-1 and self.grid[y, x+1] == 0:
            x += 1
        elif action == Actions.UP.value and y > 0 and self.grid[y-1, x] == 0:
            y -= 1
        elif action == Actions.DOWN.value and y < self.grid_size-1 and self.grid[y+1, x] == 0:
            y += 1
        
        self.agent_pos = np.array([y, x])
        return self.agent_pos
    
    def _calculate_reward(self, energy_gained: float) -> float:
        """Calculate reward for current step"""
        # Base reward for surviving
        reward = 0.01
        
        # Reward for collecting food
        if energy_gained > 0:
            reward += 1.0
        
        # Penalty for low energy
        if self.energy < 10:
            reward -= 0.1
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        y, x = self.agent_pos
        
        # Get 3x3 neighborhood (excluding center)
        obs = np.zeros(8, dtype=np.int32)  # Change from float32 to int32
        idx = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    cell_type = self.grid[ny, nx]
                    
                    # Check if food at this position
                    for fy, fx, _, has_food in self.food_sources:
                        if ny == fy and nx == fx and has_food:
                            cell_type = TileType.FOOD.value
                            break
                    
                    obs[idx] = cell_type  # Don't divide by 4.0 - keep as integer
                else:
                    obs[idx] = TileType.OBSTACLE.value  # Don't divide by 4.0
                idx += 1
        
        # Add last action and normalized energy - convert to integer range
        # Assuming vocab_size=20, scale to 0-19
        energy_scaled = int(self.energy / 100.0 * 19)  # Scale 0-100 energy to 0-19
        energy_scaled = max(0, min(19, energy_scaled))  # Clamp to range
        
        # Actions are already 0-5, scale if needed
        action_scaled = self.last_action  # Assuming actions are 0-5
        
        # If your vocab_size expects different ranges, adjust accordingly
        # For example, if you need 0-19 for all observations:
        action_scaled = int(self.last_action / 5.0 * 19)  # Scale 0-5 to 0-19
        
        obs = np.concatenate([
            obs,
            [action_scaled, energy_scaled]
        ])
        
        return obs.astype(np.int32)  # The network expects integer tokens
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state"""
        cell_size = self.render_size // self.grid_size
        
        # Create RGB image
        img = np.zeros((self.grid_size * cell_size, 
                       self.grid_size * cell_size, 3), dtype=np.uint8)
        
        # Draw grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.colors[TileType(self.grid[y, x])]
                cv2.rectangle(img,
                            (x * cell_size, y * cell_size),
                            ((x + 1) * cell_size, (y + 1) * cell_size),
                            color, -1)
        
        # Draw food
        for y, x, _, has_food in self.food_sources:
            if has_food:
                center = ((x + 0.5) * cell_size, (y + 0.5) * cell_size)
                radius = cell_size // 3
                cv2.circle(img, (int(center[0]), int(center[1])), 
                         radius, (0, 255, 0), -1)
        
        # Draw agent
        ay, ax = self.agent_pos
        center = ((ax + 0.5) * cell_size, (ay + 0.5) * cell_size)
        radius = cell_size // 2
        cv2.circle(img, (int(center[0]), int(center[1])), 
                  radius, (255, 255, 255), -1)
        
        # Add info overlay
        info = f"Energy: {self.energy:.1f} | Step: {self.steps}/{self.max_steps}"
        cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return img
    
    def close(self):
        """Close environment"""
        cv2.destroyAllWindows()
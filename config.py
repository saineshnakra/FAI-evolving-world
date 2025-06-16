"""
Configuration module for the Artificial World Simulation.

Contains all configurable parameters and constants used throughout the simulation.
"""

from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """
    Enumeration defining the two different genetic algorithm strategies.
    Each type uses different fitness functions and behavioral priorities.
    """
    COOPERATIVE = "GA1_Cooperative"    # GA1: Focuses on cooperation and efficiency
    AGGRESSIVE = "GA2_Aggressive"      # GA2: Focuses on competition and resource acquisition


@dataclass
class Config:
    """
    Central configuration class for all simulation parameters.
    Modify these values to experiment with different scenarios.
    """
    # World/Environment settings
    WORLD_WIDTH: int = 800          # Simulation world width in pixels
    WORLD_HEIGHT: int = 600         # Simulation world height in pixels
    GRID_SIZE: int = 20             # Size of world grid cells (for collision detection)
    
    # Population genetics settings
    POPULATION_SIZE_GA1: int = 100      # Cooperative agents (larger population)
    POPULATION_SIZE_GA2: int = 20       # Aggressive agents (smaller population)
    POPULATION_SIZE: int = 50           # Keep for backwards compatibility
    MAX_GENERATIONS: int = 501          # Maximum generations to simulate
    
    # Environment dynamics - IMPROVED FOR BETTER SURVIVAL
    FOOD_COUNT: int = 120           # Even more initial food availability
    HAZARD_COUNT: int = 8           # More hazards to make them noticeable
    FOOD_SPAWN_RATE: float = 0.35   # Very high food spawn rate
    
    # Agent energy and survival mechanics - IMPROVED FOR SURVIVAL
    AGENT_ENERGY: int = 150         # Increased starting energy
    MOVEMENT_COST: int = 0.5        # Reduced energy cost for movement
    EATING_REWARD: int = 30         # Increased energy from food
    SURVIVAL_REWARD: int = 1        # Fitness bonus per frame survived
    
    # Genetic algorithm parameters
    MUTATION_RATE: float = 0.15      # Probability of mutation during reproduction (0.0-1.0)
    CROSSOVER_RATE: float = 0.8     # Probability of crossover vs. cloning (0.0-1.0)
    TOURNAMENT_SIZE: int = 3        # Number of agents competing in tournament selection
    
    # Visualization and performance
    FPS: int = 1000                   # Frames per second for real-time display
    COLORS = {                      # Color scheme for different elements
        'BACKGROUND': (20, 20, 30),     # Dark background
        'FOOD': (0, 255, 0),            # Green food items
        'HAZARD': (255, 0, 0),          # Red hazardous areas
        'AGENT_GA1': (0, 150, 255),     # Blue for cooperative agents (GA1)
        'AGENT_GA2': (255, 150, 0),     # Orange for aggressive agents (GA2)
        'TEXT': (255, 255, 255)         # White text
    }


# Global configuration instance used throughout the simulation
config = Config() 
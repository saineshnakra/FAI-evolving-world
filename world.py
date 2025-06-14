"""
World module for the Artificial World Simulation.

Contains the environment management, world objects, and physics simulation
for the artificial world where agents live and evolve.
"""

import random
import math
from typing import List, Dict, Any, Tuple

from config import config


class WorldObject:
    """
    Simple class representing objects in the world (food, hazards).
    
    This is a basic container for position and type information.
    More complex objects could inherit from this class.
    """
    def __init__(self, x: int, y: int, obj_type: str):
        self.x = x              # X coordinate in world
        self.y = y              # Y coordinate in world  
        self.type = obj_type    # Object type ('food' or 'hazard')


class World:
    """
    Manages the physical environment where agents live and evolve.
    
    The world contains:
    - Food items that agents can eat for energy
    - Hazards that damage agents
    - Boundary enforcement
    - Inter-agent collision detection
    - Generation timing logic
    """
    
    def __init__(self):
        """Initialize the world with food and hazards."""
        # World dimensions
        self.width = config.WORLD_WIDTH
        self.height = config.WORLD_HEIGHT
        
        # World objects
        self.food: List[WorldObject] = []       # Food items agents can eat
        self.hazards: List[WorldObject] = []    # Dangerous areas that harm agents
        
        # Generation timing
        self.generation_time = 0                # Frames elapsed in current generation
        self.max_generation_time = 1000         # Maximum frames per generation
        
        # Initialize world with objects
        self._spawn_initial_objects()
    
    def _spawn_initial_objects(self):
        """
        Place initial food and hazards randomly in the world.
        
        This creates the starting environment that agents must navigate.
        Food placement affects foraging strategies, while hazard placement
        creates risk/reward tradeoffs.
        """
        # Spawn food items at random locations
        for _ in range(config.FOOD_COUNT):
            x = random.randint(0, self.width - 20)   # Leave space for object size
            y = random.randint(0, self.height - 20)
            self.food.append(WorldObject(x, y, 'food'))
        
        # Spawn hazardous areas at random locations
        for _ in range(config.HAZARD_COUNT):
            x = random.randint(0, self.width - 20)
            y = random.randint(0, self.height - 20)
            self.hazards.append(WorldObject(x, y, 'hazard'))
    
    def update(self, agents: List):
        """
        Update world state and handle all agent-environment interactions.
        
        This function is called each frame and handles:
        1. Time progression
        2. Dynamic food spawning
        3. Agent-food interactions (eating)
        4. Agent-hazard interactions (damage)
        5. Boundary enforcement
        
        Args:
            agents: List of all agents in the world
        """
        # Advance world time
        self.generation_time += 1
        
        # DYNAMIC ENVIRONMENT: Occasionally spawn new food
        # This prevents resource depletion and creates ongoing foraging opportunities
        if random.random() < config.FOOD_SPAWN_RATE:
            x = random.randint(0, self.width - 20)
            y = random.randint(0, self.height - 20)
            self.food.append(WorldObject(x, y, 'food'))
        
        # AGENT-ENVIRONMENT INTERACTIONS
        for agent in agents:
            if not agent.alive:
                continue  # Skip dead agents
                
            # FOOD CONSUMPTION: Check if agent is close enough to eat food
            # GA2 agents are carnivorous - they don't eat regular food, only hunt GA1 agents
            from config import AgentType
            if agent.agent_type == AgentType.COOPERATIVE:  # Only GA1 agents eat food
                for food in self.food[:]:  # Use slice copy to allow safe removal during iteration
                    # Simple collision detection: if agent and food overlap
                    if abs(agent.x - food.x) < 20 and abs(agent.y - food.y) < 20:
                        # Agent successfully eats food
                        agent.energy += config.EATING_REWARD
                        agent.food_collected += 1
                        self.food.remove(food)  # Remove eaten food from world
                        break  # Agent can only eat one food per frame
            
            # AGENT-TO-AGENT COMBAT: Aggressive agents attack cooperative ones
            if (agent.agent_type == AgentType.AGGRESSIVE and agent.energy > 50):
                
                for other_agent in agents:
                    if (other_agent != agent and other_agent.alive and 
                        other_agent.agent_type == AgentType.COOPERATIVE):
                        
                        # Check if agents are close enough for combat (based on size)
                        agent_size = 10 + (agent.genome.size * 15)  # 10-25 pixel radius
                        other_size = 10 + (other_agent.genome.size * 15)
                        collision_distance = agent_size + other_size
                        
                        distance = math.sqrt((agent.x - other_agent.x)**2 + (agent.y - other_agent.y)**2)
                        if distance < collision_distance and random.random() < 0.3:  # 30% chance of attack
                            # Combat! Larger agents do more damage
                            base_damage = 15
                            size_damage_bonus = agent.genome.size * 10  # Larger = more damage
                            total_damage = base_damage + size_damage_bonus
                            
                            other_agent.energy = max(0, other_agent.energy - total_damage)
                            agent.attacks_made += 1
                            
                            # Attacker gains energy from successful attack
                            energy_gain = 8
                            agent.energy = min(config.AGENT_ENERGY, agent.energy + energy_gain)
                            
                            if other_agent.energy <= 0:
                                other_agent.alive = False
                            
                            break  # One attack per frame
            
            # HAZARD DAMAGE: Check if agent is in dangerous area
            for hazard in self.hazards:
                if abs(agent.x - hazard.x) < 30 and abs(agent.y - hazard.y) < 30:
                    # Agent takes continuous damage from hazard
                    hazard_damage = 15  # Increased damage
                    agent.energy -= hazard_damage
                    
                    # Visual feedback - mark agent as in hazard
                    if not hasattr(agent, 'in_hazard'):
                        agent.in_hazard = 0
                    agent.in_hazard = 5  # Mark for 5 frames
                    
                    # Kill agent if energy drops too low from hazard
                    if agent.energy <= 0:
                        agent.alive = False
    
    def get_world_state_for_agent(self, agent, all_agents: List) -> Dict[str, Any]:
        """
        Provide environmental information visible to an agent for decision-making.
        
        This implements the agent's "sensory system" - what information they can
        perceive about their environment. Vision range is now determined by the
        agent's genetic sense trait.
        
        Args:
            agent: The agent requesting world state information
            all_agents: All agents in the world (for detecting neighbors)
            
        Returns:
            Dictionary containing nearby objects within agent's vision range
        """
        # Vision range based on agent's sense trait (50-150 pixel range)
        base_vision = 50
        max_vision_bonus = 100
        vision_range = base_vision + (agent.genome.sense * max_vision_bonus)
        
        # FOOD DETECTION: Find all food within vision range
        nearby_food = []
        for food in self.food:
            distance = math.sqrt((food.x - agent.x)**2 + (food.y - agent.y)**2)
            if distance < vision_range:
                nearby_food.append((food.x, food.y))
        
        # AGENT DETECTION: Find other agents within vision range
        nearby_agents = []
        for other_agent in all_agents:
            if other_agent == agent or not other_agent.alive:
                continue  # Skip self and dead agents
            
            distance = math.sqrt((other_agent.x - agent.x)**2 + (other_agent.y - agent.y)**2)
            if distance < vision_range:
                nearby_agents.append((other_agent.x, other_agent.y, other_agent.agent_type))
        
        # HAZARD DETECTION: Find hazards within vision range
        nearby_hazards = []
        for hazard in self.hazards:
            distance = math.sqrt((hazard.x - agent.x)**2 + (hazard.y - agent.y)**2)
            if distance < vision_range:
                nearby_hazards.append((hazard.x, hazard.y))
        
        return {
            'nearby_food': nearby_food,
            'nearby_agents': nearby_agents,
            'nearby_hazards': nearby_hazards
        }
    
    def is_generation_complete(self, agents: List) -> bool:
        """
        Determine if the current generation should end and evolution should occur.
        
        Generation end conditions:
        1. Very few agents remain alive (population bottleneck)
        2. Maximum time limit reached (prevents infinite generations)
        
        Args:
            agents: All agents in current generation
            
        Returns:
            bool: True if generation should end
        """
        alive_agents = [a for a in agents if a.alive]
        
        return (
            len(alive_agents) <= 2 or                      # Population bottleneck
            self.generation_time > self.max_generation_time  # Time limit reached
        )
    
    def reset_for_new_generation(self):
        """
        Reset world state for the beginning of a new generation.
        
        This provides a fresh environment for each generation, ensuring
        that evolution is based on genetic fitness rather than
        accumulated environmental advantages.
        """
        self.generation_time = 0    # Reset generation timer
        
        # Clear all objects and respawn fresh ones
        self.food.clear()
        self.hazards.clear()
        self._spawn_initial_objects() 
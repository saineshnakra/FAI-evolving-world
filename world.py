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
        4. Agent-agent interactions (combat)
        5. Agent-hazard interactions (damage)
        6. Boundary enforcement
        
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
                
            # FIX 1: FOOD CONSUMPTION - Use the agent's eat_food() method
            # This properly handles both GA1 and GA2 food consumption
            for food in self.food[:]:  # Use slice copy to allow safe removal during iteration
                # Simple collision detection: if agent and food overlap
                if abs(agent.x - food.x) < 20 and abs(agent.y - food.y) < 20:
                    # Use agent's eat_food method which handles GA1/GA2 differences
                    agent.eat_food(config.EATING_REWARD)
                    self.food.remove(food)  # Remove eaten food from world
                    break  # Agent can only eat one food per frame
            
            # HAZARD DAMAGE: Check if agent is in dangerous area
            for hazard in self.hazards:
                if abs(agent.x - hazard.x) < 30 and abs(agent.y - hazard.y) < 30:
                    # Agent takes continuous damage from hazard
                    hazard_damage = 15
                    agent.energy -= hazard_damage
                    
                    # Visual feedback - mark agent as in hazard
                    if not hasattr(agent, 'in_hazard'):
                        agent.in_hazard = 0
                    agent.in_hazard = 5  # Mark for 5 frames
                    
                    # Kill agent if energy drops too low from hazard
                    if agent.energy <= 0:
                        agent.alive = False
        
        # FIX 2: AGENT-TO-AGENT COMBAT - Separate loop to use proper attack mechanics
        # Handle combat interactions using the agent's attack_agent() method
        from config import AgentType
        
        alive_agents = [a for a in agents if a.alive]
        
        for agent in alive_agents:
            if agent.agent_type == AgentType.AGGRESSIVE and agent.energy > 30:  # Lower threshold for attacking
                
                for target in alive_agents:
                    if (target != agent and target.alive and 
                        target.agent_type == AgentType.COOPERATIVE):
                        
                        # Check if agents are close enough for combat (based on size)
                        agent_size = 10 + (agent.genome.size * 15)  # 10-25 pixel radius
                        target_size = 10 + (target.genome.size * 15)
                        collision_distance = agent_size + target_size
                        
                        distance = math.sqrt((agent.x - target.x)**2 + (agent.y - target.y)**2)
                        
                        # FIX 3: Use the agent's attack_agent() method for proper combat mechanics
                        if distance < collision_distance and random.random() < 0.2:  # 20% chance of attack per frame
                            success = agent.attack_agent(target)
                            if success:
                                break  # One successful attack per frame per agent
    
    def get_world_state_for_agent(self, agent, all_agents: List) -> Dict[str, Any]:
        """
        Provide environmental information visible to an agent for decision-making.
        
        This implements the agent's "sensory system" - what information they can
        perceive about their environment. Vision range is determined by the
        agent's genetic sense trait using the agent's get_vision_range() method.
        
        Args:
            agent: The agent requesting world state information
            all_agents: All agents in the world (for detecting neighbors)
            
        Returns:
            Dictionary containing nearby objects within agent's vision range
        """
        # FIX 4: Use agent's genetic vision range method
        vision_range = agent.get_vision_range()
        
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
    
    def apply_agent_movement(self, agent, dx: int, dy: int):
        """
        FIX 5: Apply agent movement with proper speed scaling and boundary enforcement.
        
        This method handles:
        - Speed trait scaling
        - Boundary collision
        - Movement validation
        
        Args:
            agent: The agent to move
            dx, dy: Desired movement direction (-1, 0, or 1)
        """
        if not agent.alive or (dx == 0 and dy == 0):
            return
        
        # Apply genetic speed scaling
        speed_multiplier = agent.get_movement_speed()
        
        # Calculate actual movement distance based on genetics
        # Base movement is 2-6 pixels, scaled by speed trait
        base_movement = 4  # Increased from 2 to 4
        actual_movement = max(2, int(base_movement * speed_multiplier))  # Min 2 pixels
        
        # Apply movement with boundary checking
        new_x = agent.x + (dx * actual_movement)
        new_y = agent.y + (dy * actual_movement)
        
        # Boundary enforcement with proper margins for agent size
        agent_radius = 5 + int(agent.genome.size * 10)  # Agent visual size
        margin = agent_radius + 5  # Extra safety margin
        
        if new_x < margin:
            new_x = margin
        elif new_x >= self.width - margin:
            new_x = self.width - margin - 1
            
        if new_y < margin:
            new_y = margin
        elif new_y >= self.height - margin:
            new_y = self.height - margin - 1
        
        # Update agent position
        agent.x = new_x
        agent.y = new_y
    
    def is_generation_complete(self, agents: List) -> bool:
        """
        Determine if the current generation should end and evolution should occur.
        
        Generation end conditions:
        1. Very few agents remain alive (population bottleneck)
        2. Maximum time limit reached (prevents infinite generations)
        3. Population stagnation (optional - could add if needed)
        
        Args:
            agents: All agents in current generation
            
        Returns:
            bool: True if generation should end
        """
        alive_agents = [a for a in agents if a.alive]
        
        # FIX 6: More nuanced generation end conditions
        total_population = len(agents)
        alive_count = len(alive_agents)
        
        # End generation if population drops below 20% of original
        population_threshold = max(2, total_population * 0.2)
        
        return (
            alive_count <= population_threshold or              # Population bottleneck
            self.generation_time > self.max_generation_time     # Time limit reached
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
    
    def get_spawn_positions(self, count: int) -> List[Tuple[int, int]]:
        """
        FIX 7: Generate safe spawn positions for new agents.
        
        This ensures agents don't spawn inside hazards or on top of each other.
        
        Args:
            count: Number of spawn positions needed
            
        Returns:
            List of (x, y) coordinate tuples
        """
        positions = []
        max_attempts = count * 10  # Prevent infinite loops
        attempts = 0
        
        while len(positions) < count and attempts < max_attempts:
            x = random.randint(50, self.width - 50)   # Stay away from edges
            y = random.randint(50, self.height - 50)
            
            # Check if position is safe (not in hazard, not too close to other spawn points)
            safe = True
            
            # Check distance from hazards
            for hazard in self.hazards:
                if math.sqrt((x - hazard.x)**2 + (y - hazard.y)**2) < 50:
                    safe = False
                    break
            
            # Check distance from other spawn positions
            if safe:
                for px, py in positions:
                    if math.sqrt((x - px)**2 + (y - py)**2) < 30:
                        safe = False
                        break
            
            if safe:
                positions.append((x, y))
            
            attempts += 1
        
        # If we couldn't find enough safe positions, fill with random ones
        while len(positions) < count:
            x = random.randint(20, self.width - 20)
            y = random.randint(20, self.height - 20)
            positions.append((x, y))
        
        return positions
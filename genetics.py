"""
Genetics module for the Artificial World Simulation.

Contains the genetic representation, agent behavior, and genetic algorithm
implementation for evolving populations.
"""

import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from config import config, AgentType


@dataclass
class Genome:
    """
    Genetic representation of an agent's behavioral traits.
    
    This class encodes all heritable characteristics that determine how an agent
    behaves in the world. Each trait is a floating-point value typically between 0-1.
    
    Attributes:
        move_weights: Preferences for movement directions [up, down, left, right]
        aggression: Tendency to attack other agents (0=peaceful, 1=highly aggressive)
        food_priority: How much to prioritize seeking food (0=low, 1=high)
        risk_tolerance: Willingness to approach dangerous areas (0=cautious, 1=reckless)
        energy_conservation: Tendency to save energy when low (0=always move, 1=conservative)
        attack_threshold: Energy level required before attacking others (0.0-1.0)
        flee_threshold: Energy level that triggers fleeing behavior (0.0-1.0)
    """
    # Movement behavior - influences random walk patterns
    move_weights: List[float]      # [up, down, left, right] movement preferences
    
    # Core behavioral traits that define agent personality
    aggression: float              # Likelihood to attack other agents (0.0-1.0)
    food_priority: float           # How much to prioritize food seeking (0.0-1.0)
    risk_tolerance: float          # Willingness to approach hazards (0.0-1.0)
    energy_conservation: float     # When to stop moving to save energy (0.0-1.0)
    
    # Decision-making thresholds that trigger specific behaviors
    attack_threshold: float        # Energy level needed to attack (0.0-1.0)
    flee_threshold: float          # Energy level that triggers fleeing (0.0-1.0)
    
    @classmethod
    def random(cls) -> 'Genome':
        """
        Generate a random genome for initial population or mutation recovery.
        
        Returns:
            Genome: A new genome with all traits randomized
        """
        return cls(
            move_weights=[random.random() for _ in range(4)],  # Random movement preferences
            aggression=random.random(),
            food_priority=random.random(),
            risk_tolerance=random.random(),
            energy_conservation=random.random(),
            attack_threshold=random.uniform(0.3, 0.8),  # Reasonable range for attacking
            flee_threshold=random.uniform(0.2, 0.6)     # Reasonable range for fleeing
        )


class Agent:
    """
    Individual agent that exists in the world and evolves over time.
    
    Each agent has a genome that determines its behavior, and tracks its
    performance metrics for fitness evaluation. Agents can move, eat food,
    interact with hazards, and potentially attack other agents.
    """
    
    def __init__(self, x: int, y: int, agent_type: AgentType, genome: Genome = None):
        """
        Initialize a new agent.
        
        Args:
            x, y: Starting position in the world
            agent_type: Whether this is a cooperative (GA1) or aggressive (GA2) agent
            genome: Genetic traits (if None, generates random genome)
        """
        # Position and identity
        self.x = x                          # Current x coordinate
        self.y = y                          # Current y coordinate
        self.agent_type = agent_type        # GA1 (cooperative) or GA2 (aggressive)
        self.genome = genome or Genome.random()  # Genetic traits
        
        # Survival and energy management
        self.energy = config.AGENT_ENERGY   # Current energy level
        self.age = 0                        # How many frames this agent has survived
        self.alive = True                   # Whether agent is still active
        
        # Performance tracking for fitness calculation
        self.fitness = 0                    # Overall fitness score
        self.food_collected = 0             # Number of food items eaten
        self.attacks_made = 0               # Number of attacks on other agents
        self.distance_traveled = 0          # Total distance moved (for efficiency metrics)
        
    def update(self, world_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Update agent for one simulation step and return desired movement.
        
        This is the main agent behavior function called each frame. It:
        1. Checks if agent is still alive
        2. Ages the agent and consumes energy
        3. Calculates current fitness
        4. Decides on action based on genome and world state
        
        Args:
            world_state: Dictionary containing nearby objects and agents
            
        Returns:
            Tuple of (dx, dy) representing desired movement direction
        """
        # Check if agent died from energy depletion
        if not self.alive or self.energy <= 0:
            self.alive = False
            return 0, 0  # No movement if dead
            
        # Age the agent and consume energy for basic metabolism
        self.age += 1
        self.energy -= config.MOVEMENT_COST
        
        # Update fitness based on current performance
        self._calculate_fitness()
        
        # Make behavioral decision based on genetics and environment
        return self._decide_action(world_state)
    
    def _calculate_fitness(self):
        """
        Calculate fitness score based on agent type and performance metrics.
        
        GA1 (Cooperative): Rewards food collection, survival time, and energy efficiency
        GA2 (Aggressive): Rewards resource acquisition, successful attacks, and dominance
        
        This is where the two different evolutionary strategies are implemented.
        """
        # Base fitness components shared by both strategies
        base_fitness = (
            self.food_collected * 10 +      # Reward successful foraging
            self.age * 0.1                  # Reward longevity
        )
        
        if self.agent_type == AgentType.COOPERATIVE:
            # GA1 Strategy: Cooperative and efficient
            # Rewards energy conservation, peaceful coexistence, and sustainability
            energy_bonus = self.energy * 0.1        # Reward energy conservation
            self.fitness = base_fitness + energy_bonus
            
        else:
            # GA2 Strategy: Aggressive and competitive  
            # Rewards resource acquisition through competition and dominance
            aggression_bonus = self.attacks_made * 5    # Reward successful attacks
            resource_bonus = self.food_collected * 2    # Extra reward for food acquisition
            self.fitness = base_fitness + aggression_bonus + resource_bonus
    
    def _decide_action(self, world_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Make behavioral decision based on genome and current world state.
        
        This function implements the agent's decision-making process:
        1. Extract relevant information from world state
        2. Apply genetic preferences to prioritize actions
        3. Return movement decision
        
        Args:
            world_state: Contains nearby food, agents, and hazards
            
        Returns:
            Movement direction as (dx, dy) tuple (-1, 0, or 1 for each axis)
        """
        # Extract environmental information
        nearby_food = world_state.get('nearby_food', [])
        nearby_agents = world_state.get('nearby_agents', [])
        nearby_hazards = world_state.get('nearby_hazards', [])
        
        dx, dy = 0, 0  # Default: no movement
        
        # BEHAVIOR 1: Food-seeking behavior
        # If food is visible and agent prioritizes food, move toward closest food
        if nearby_food and random.random() < self.genome.food_priority:
            closest_food = min(nearby_food, key=lambda f: 
                             math.sqrt((f[0] - self.x)**2 + (f[1] - self.y)**2))
            
            # Calculate direction to food
            if closest_food[0] > self.x:
                dx = 1      # Move right toward food
            elif closest_food[0] < self.x:
                dx = -1     # Move left toward food
                
            if closest_food[1] > self.y:
                dy = 1      # Move down toward food
            elif closest_food[1] < self.y:
                dy = -1     # Move up toward food
        
        # BEHAVIOR 2: Random exploration based on movement preferences
        # If not pursuing food, move based on genetic movement weights
        elif random.random() < 0.3:  # 30% chance of random movement
            # Normalize movement weights to create probability distribution
            total_weight = sum(self.genome.move_weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in self.genome.move_weights]
                direction = np.random.choice(4, p=probabilities)
                
                # Apply chosen direction
                if direction == 0:
                    dy = -1    # Up
                elif direction == 1:
                    dy = 1     # Down
                elif direction == 2:
                    dx = -1    # Left
                elif direction == 3:
                    dx = 1     # Right
        
        # BEHAVIOR 3: Hazard avoidance (future enhancement)
        # TODO: Implement hazard avoidance based on risk_tolerance
        
        # BEHAVIOR 4: Agent interaction (future enhancement)  
        # TODO: Implement attack/flee behavior based on aggression and thresholds
        
        return dx, dy


class GeneticAlgorithm:
    """
    Manages evolution of a population using genetic algorithm principles.
    
    This class handles:
    - Population initialization and management
    - Selection of parents for reproduction
    - Crossover and mutation operations
    - Generation tracking and statistics
    
    Each GA instance manages one population (either cooperative or aggressive agents).
    """
    
    def __init__(self, agent_type: AgentType):
        """
        Initialize genetic algorithm for a specific agent type.
        
        Args:
            agent_type: Whether this GA evolves cooperative or aggressive agents
        """
        self.agent_type = agent_type        # Type of agents this GA manages
        self.generation = 0                 # Current generation number
        self.population: List[Agent] = []   # Current population of agents
        
        # Performance tracking across generations
        self.best_fitness_history = []      # Best fitness each generation
        self.avg_fitness_history = []       # Average fitness each generation
        
    def initialize_population(self, spawn_positions: List[Tuple[int, int]]):
        """
        Create the first generation with random genomes.
        
        Args:
            spawn_positions: List of (x, y) coordinates where agents can spawn
        """
        self.population = []
        for i in range(config.POPULATION_SIZE):
            # Cycle through spawn positions if there are fewer positions than agents
            pos = spawn_positions[i % len(spawn_positions)]
            agent = Agent(pos[0], pos[1], self.agent_type)
            self.population.append(agent)
    
    def evolve_generation(self, spawn_positions: List[Tuple[int, int]]):
        """
        Evolve from current generation to the next using GA operations.
        
        Evolution process:
        1. Record statistics from current generation
        2. Handle population extinction scenarios
        3. Select elite agents (best performers)
        4. Generate new population through crossover and mutation
        5. Increment generation counter
        
        Args:
            spawn_positions: Where new agents can be placed
        """
        if not self.population:
            return
            
        # Step 1: Collect statistics and handle extinction
        alive_agents = [agent for agent in self.population if agent.alive]
        
        if not alive_agents:
            # EXTINCTION RECOVERY: If all agents died, restart with random population
            print(f"Population extinction in {self.agent_type.value}! Restarting...")
            self.initialize_population(spawn_positions)
            return
            
        # Record generation statistics for analysis
        fitnesses = [agent.fitness for agent in alive_agents]
        if fitnesses:
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        
        # Step 2: Create new population through genetic operations
        new_population = []
        
        # ELITISM: Keep the best performers from current generation
        # This ensures that good solutions are not lost during evolution
        elite_count = max(1, min(len(alive_agents), config.POPULATION_SIZE // 10))
        elite = sorted(alive_agents, key=lambda x: x.fitness, reverse=True)[:elite_count]
        
        # Add elite agents to new population (with new positions)
        for agent in elite:
            pos = random.choice(spawn_positions)
            new_agent = Agent(pos[0], pos[1], self.agent_type, agent.genome)
            new_population.append(new_agent)
        
        # REPRODUCTION: Generate remaining population through crossover and mutation
        while len(new_population) < config.POPULATION_SIZE:
            if len(alive_agents) >= 2:
                # Normal case: select two parents for reproduction
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # CROSSOVER: Combine genetic material from two parents
                if random.random() < config.CROSSOVER_RATE:
                    child_genome = self._crossover(parent1.genome, parent2.genome)
                else:
                    # No crossover: clone one parent
                    child_genome = parent1.genome
            else:
                # Edge case: very few survivors, use best available or random
                if alive_agents:
                    child_genome = alive_agents[0].genome
                else:
                    child_genome = Genome.random()
            
            # MUTATION: Randomly modify genetic traits
            if random.random() < config.MUTATION_RATE:
                child_genome = self._mutate(child_genome)
            
            # Create new agent with evolved genome
            pos = random.choice(spawn_positions)
            child = Agent(pos[0], pos[1], self.agent_type, child_genome)
            new_population.append(child)
        
        # Step 3: Replace old population and advance generation
        self.population = new_population
        self.generation += 1
    
    def _tournament_selection(self) -> Agent:
        """
        Select parent agent using tournament selection.
        
        Tournament selection works by:
        1. Randomly choosing a small group of agents (tournament)
        2. Selecting the best agent from that group
        3. This gives better agents higher chance of reproduction while
           maintaining some randomness
        
        Returns:
            Agent: Selected parent for reproduction
        """
        alive_agents = [a for a in self.population if a.alive]
        
        # Handle edge cases where few or no agents are alive
        if not alive_agents:
            return random.choice(self.population)  # Fallback to any agent
        
        # Ensure tournament size doesn't exceed available agents
        tournament_size = min(config.TOURNAMENT_SIZE, len(alive_agents))
        if tournament_size == 0:
            return random.choice(alive_agents)
        
        # Conduct tournament: select random group and return best
        tournament = random.sample(alive_agents, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, genome1: Genome, genome2: Genome) -> Genome:
        """
        Create offspring genome by combining traits from two parents.
        
        Uses uniform crossover: for each trait, randomly choose which parent
        to inherit from. This maintains genetic diversity while combining
        successful traits from both parents.
        
        Args:
            genome1, genome2: Parent genomes
            
        Returns:
            Genome: New child genome with mixed traits
        """
        new_genome = Genome(
            # For each movement weight, randomly choose from parent1 or parent2
            move_weights=[g1 if random.random() < 0.5 else g2 
                         for g1, g2 in zip(genome1.move_weights, genome2.move_weights)],
            
            # For each behavioral trait, randomly inherit from one parent
            aggression=genome1.aggression if random.random() < 0.5 else genome2.aggression,
            food_priority=genome1.food_priority if random.random() < 0.5 else genome2.food_priority,
            risk_tolerance=genome1.risk_tolerance if random.random() < 0.5 else genome2.risk_tolerance,
            energy_conservation=genome1.energy_conservation if random.random() < 0.5 else genome2.energy_conservation,
            attack_threshold=genome1.attack_threshold if random.random() < 0.5 else genome2.attack_threshold,
            flee_threshold=genome1.flee_threshold if random.random() < 0.5 else genome2.flee_threshold
        )
        return new_genome
    
    def _mutate(self, genome: Genome) -> Genome:
        """
        Apply random mutations to a genome.
        
        Uses Gaussian (normal) mutation: adds small random values to each trait.
        This allows for fine-tuning of successful strategies while occasionally
        making larger changes that could lead to breakthrough behaviors.
        
        Args:
            genome: Original genome to mutate
            
        Returns:
            Genome: Mutated version with modified traits
        """
        mutated_genome = Genome(
            # Mutate movement weights: add small random changes, keep in [0,1] range
            move_weights=[max(0, min(1, w + random.gauss(0, 0.1))) for w in genome.move_weights],
            
            # Mutate behavioral traits: small Gaussian perturbations
            aggression=max(0, min(1, genome.aggression + random.gauss(0, 0.1))),
            food_priority=max(0, min(1, genome.food_priority + random.gauss(0, 0.1))),
            risk_tolerance=max(0, min(1, genome.risk_tolerance + random.gauss(0, 0.1))),
            energy_conservation=max(0, min(1, genome.energy_conservation + random.gauss(0, 0.1))),
            attack_threshold=max(0, min(1, genome.attack_threshold + random.gauss(0, 0.1))),
            flee_threshold=max(0, min(1, genome.flee_threshold + random.gauss(0, 0.1)))
        )
        return mutated_genome 
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
    Simplified genetic representation with only three key traits.
    
    This streamlined approach focuses on the most impactful characteristics:
    - speed: Movement speed multiplier (0.0-1.0) - affects how fast agent moves
    - sense: Vision/sight range multiplier (0.0-1.0) - determines how far agent can see food, enemies, and hazards (50-150 pixels)
    - size: Agent size multiplier (0.0-1.0) - affects collision detection, combat damage, and movement penalty
    
    All traits are normalized between 0.0 and 1.0 for consistency.
    Vision range formula: 50 + (sense * 100) pixels = 50-150 pixel sight range
    """
    speed: float        # Movement speed multiplier (0.0-1.0)
    sense: float        # Vision/sight range multiplier (0.0-1.0) -> 50-150 pixels sight range
    size: float         # Size multiplier (0.0-1.0, larger = slower but more damage)
    
    @classmethod
    def random(cls) -> 'Genome':
        """
        Generate a random genome with all three traits randomized.
        
        Returns:
            Genome: A new genome with random trait values
        """
        return cls(
            speed=random.random(),
            sense=random.random(), 
            size=random.random()
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
        self.successful_attacks = 0         # Number of successful attacks (energy gained)
        self.distance_traveled = 0          # Total distance moved (for efficiency metrics)
        
        # FIX 1: Track previous position for distance calculation
        self.prev_x = x
        self.prev_y = y
        
    def get_vision_range(self) -> float:
        """
        Calculate actual vision range based on sense trait.
        
        Returns:
            float: Vision range in pixels (50-150 based on sense trait)
        """
        return 50 + (self.genome.sense * 100)
    
    def get_movement_speed(self) -> float:
        """
        Calculate actual movement speed based on speed trait and size penalty.
        
        Returns:
            float: Movement speed multiplier
        """
        # FIX 2: Larger agents move slower (size penalty)
        size_penalty = 1.0 - (self.genome.size * 0.3)  # Up to 30% speed reduction
        return self.genome.speed * size_penalty
    
    def get_energy_cost(self) -> float:
        """
        Calculate energy cost per move based on size and speed.
        
        Returns:
            float: Energy cost for this agent's movement
        """
        # FIX 3: Energy cost scales with size and speed
        base_cost = config.MOVEMENT_COST
        size_cost = 1.0 + (self.genome.size * 0.5)  # Larger agents cost more energy
        speed_cost = 1.0 + (self.genome.speed * 0.3)  # Faster movement costs more
        return base_cost * size_cost * speed_cost

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
            
        # Age the agent and consume energy based on genetics
        self.age += 1
        energy_cost = self.get_energy_cost()
        self.energy -= energy_cost
        
        # Update fitness based on current performance
        self._calculate_fitness()
        
        # Make behavioral decision based on genetics and environment
        dx, dy = self._decide_action(world_state)
        
        # FIX 4: Track distance traveled for efficiency metrics
        if dx != 0 or dy != 0:
            self.distance_traveled += math.sqrt(dx*dx + dy*dy)
            self.prev_x, self.prev_y = self.x, self.y
        
        return dx, dy
    
    def attack_agent(self, target_agent: 'Agent') -> bool:
        """
        FIX 5: Implement actual attack mechanics for GA2 agents.
        
        Args:
            target_agent: The agent being attacked
            
        Returns:
            bool: True if attack was successful
        """
        if self.agent_type != AgentType.AGGRESSIVE:
            return False
            
        if not target_agent.alive:
            return False
            
        # Attack success based on size difference and random factor
        attack_power = self.genome.size * 50 + random.uniform(0, 20)  # 0-70 damage
        defense_power = target_agent.genome.size * 30 + random.uniform(0, 30)  # 0-60 defense
        
        self.attacks_made += 1
        
        if attack_power > defense_power:
            # Successful attack
            damage = attack_power - defense_power
            target_agent.energy -= damage
            
            # FIX 6: GA2 agents gain energy from successful attacks
            energy_gained = min(damage * 0.8, target_agent.energy * 0.3)  # Gain 80% of damage dealt or 30% of target's energy
            self.energy += energy_gained
            self.successful_attacks += 1
            
            if target_agent.energy <= 0:
                target_agent.alive = False
                # Bonus energy for killing
                self.energy += 20
            
            return True
        
        return False
    
    def eat_food(self, food_energy: float):
        """
        Food consumption with different efficiency for different agent types.
        
        Args:
            food_energy: Amount of energy to gain from food
        """
        if self.agent_type == AgentType.COOPERATIVE:
            # GA1: Full energy gain from food (natural herbivores)
            self.energy += food_energy
            self.food_collected += 1
        else:
            # GA2: Reduced energy gain from food (carnivores eating plants as backup)
            # They can digest it, but not efficiently
            backup_energy = food_energy * 0.5  # Only 50% efficiency
            self.energy += backup_energy
            self.food_collected += 1  # Still counts as food consumed
    
    def _calculate_fitness(self):
        """
        Energy-informed fitness function that balances survival with strategic effectiveness.
        
        Fitness = Survival Success + Strategic Performance + Energy Management + Genetic Optimization
        
        This approach uses energy as a survival constraint while rewarding strategies
        that lead to sustainable, effective long-term survival.
        """
        # COMPONENT 1: Survival Success (primary objective)
        # Reward staying alive longer, but with diminishing returns to avoid pure camping
        survival_fitness = math.log(self.age + 1) * 5  # Logarithmic scaling
        
        # COMPONENT 2: Energy Management (sustainability measure)
        # Reward agents that maintain energy effectively over time
        if self.age > 0:
            avg_energy = self.energy / self.age if self.age > 0 else self.energy
            energy_sustainability = min(avg_energy * 0.2, 20)  # Cap to prevent energy hoarding
        else:
            energy_sustainability = 0
        
        # COMPONENT 3: Strategic Performance (population-specific objectives)
        strategic_fitness = 0
        
        if self.agent_type == AgentType.COOPERATIVE:
            # GA1 Strategy: Efficient resource acquisition and survival
            
            # Resource efficiency: food per unit time and energy spent
            if self.age > 0:
                food_efficiency = (self.food_collected / (self.age + 1)) * 15
                strategic_fitness += food_efficiency
            
            # Energy efficiency: minimize energy waste while staying active
            if self.distance_traveled > 0:
                movement_efficiency = self.food_collected / (self.distance_traveled + 1)
                strategic_fitness += movement_efficiency * 10
            
            # Genetic optimization bonuses
            sense_effectiveness = self.genome.sense * self.food_collected * 1.5
            size_efficiency = (1.0 - self.genome.size) * energy_sustainability * 0.5
            strategic_fitness += sense_effectiveness + size_efficiency
            
        else:
            # GA2 Strategy: Effective hunting and dominance
            
            # Hunting effectiveness: successful attacks per unit time
            if self.age > 0:
                hunt_effectiveness = (self.successful_attacks / (self.age + 1)) * 25
                strategic_fitness += hunt_effectiveness
            
            # Combat efficiency: successful attack rate
            if self.attacks_made > 0:
                attack_success_rate = self.successful_attacks / self.attacks_made
                strategic_fitness += attack_success_rate * 15
            
            # Genetic optimization bonuses
            combat_size_bonus = self.genome.size * self.successful_attacks * 2
            hunting_speed_bonus = self.genome.speed * self.attacks_made * 1.5
            hunting_sense_bonus = self.genome.sense * self.successful_attacks * 2
            strategic_fitness += combat_size_bonus + hunting_speed_bonus + hunting_sense_bonus
        
        # COMPONENT 4: Energy-Survival Integration
        # Bonus for agents that survive long while maintaining active energy levels
        if self.age > 50 and self.energy > 30:  # Long-term survivors with good energy
            longevity_bonus = math.sqrt(self.age) * (self.energy / 100) * 5
        else:
            longevity_bonus = 0
        
        # FINAL FITNESS CALCULATION
        self.fitness = survival_fitness + energy_sustainability + strategic_fitness + longevity_bonus
        
        # Ensure fitness is always positive
        self.fitness = max(self.fitness, 1.0)
    
    def _decide_action(self, world_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        FIX 11: Make behavioral decision that uses genetic traits effectively.
        
        Args:
            world_state: Contains nearby food, agents, and hazards
            
        Returns:
            Movement direction as (dx, dy) tuple (-1, 0, or 1 for each axis)
        """
        # Extract environmental information
        nearby_food = world_state.get('nearby_food', [])
        nearby_agents = world_state.get('nearby_agents', [])
        
        dx, dy = 0, 0  # Default: no movement
        
        # FIX 12: Use vision range based on genetics
        vision_range = self.get_vision_range()
        
        # Filter objects by actual vision range
        visible_food = [(fx, fy) for fx, fy in nearby_food 
                       if math.sqrt((fx - self.x)**2 + (fy - self.y)**2) <= vision_range]
        
        visible_agents = [(ax, ay, agent_type) for ax, ay, agent_type in nearby_agents 
                         if math.sqrt((ax - self.x)**2 + (ay - self.y)**2) <= vision_range]
        
        # BEHAVIOR 1: Agent-to-agent interactions
        if self.agent_type == AgentType.AGGRESSIVE and visible_agents:
            # Aggressive agents pursue cooperative agents
            coop_agents = [(ax, ay, agent_type) for ax, ay, agent_type in visible_agents 
                          if agent_type == AgentType.COOPERATIVE]
            if coop_agents:
                # Move towards nearest cooperative agent
                target_x, target_y, _ = min(coop_agents, 
                    key=lambda agent_info: math.sqrt((agent_info[0] - self.x)**2 + (agent_info[1] - self.y)**2))
                
                dx, dy = self._calculate_movement_direction(target_x, target_y)
                return dx, dy
        
        elif self.agent_type == AgentType.COOPERATIVE and visible_agents:
            # FIX 13: GA1 agents flee when they detect threats within their vision
            aggr_agents = [(ax, ay, agent_type) for ax, ay, agent_type in visible_agents 
                          if agent_type == AgentType.AGGRESSIVE]
            if aggr_agents and (self.energy < 50 or self.genome.size < 0.5):  # Flee if low energy or small
                # Move away from nearest aggressive agent
                threat_x, threat_y, _ = min(aggr_agents, 
                    key=lambda agent_info: math.sqrt((agent_info[0] - self.x)**2 + (agent_info[1] - self.y)**2))
                
                dx, dy = self._calculate_movement_direction(threat_x, threat_y, flee=True)
                return dx, dy
        
        # BEHAVIOR 2: Food-seeking (GA1 only, but GA2 can eat food as emergency backup)
        if visible_food:
            if self.agent_type == AgentType.COOPERATIVE or self.energy < 30:  # GA2 eats food only when desperate
                closest_food = min(visible_food, key=lambda f: 
                                 math.sqrt((f[0] - self.x)**2 + (f[1] - self.y)**2))
                
                dx, dy = self._calculate_movement_direction(closest_food[0], closest_food[1])
                return dx, dy
        
        # BEHAVIOR 3: Random exploration based on speed trait
        exploration_chance = 0.2 + (self.genome.speed * 0.3)  # Faster agents explore more
        if random.random() < exploration_chance:
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
        
        return dx, dy
    
    def _calculate_movement_direction(self, target_x: int, target_y: int, flee: bool = False) -> Tuple[int, int]:
        """
        FIX 14: Helper method to calculate movement direction towards or away from target.
        
        Args:
            target_x, target_y: Target coordinates
            flee: If True, move away from target instead of towards
            
        Returns:
            Tuple of movement direction (-1, 0, or 1 for each axis)
        """
        if flee:
            # Calculate direction away from target
            dx = -1 if target_x > self.x else (1 if target_x < self.x else 0)
            dy = -1 if target_y > self.y else (1 if target_y < self.y else 0)
        else:
            # Calculate direction towards target
            dx = 1 if target_x > self.x else (-1 if target_x < self.x else 0)
            dy = 1 if target_y > self.y else (-1 if target_y < self.y else 0)
        
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
        Create the first generation using genetic algorithm operations on random seed population.
        
        Process:
        1. Create random seed population with diverse traits
        2. Assign random fitness values to simulate initial selection pressure
        3. Apply genetic algorithm operations (selection, crossover, mutation)
        4. Result: Starting population that follows GA principles but has trait diversity
        
        Args:
            spawn_positions: List of (x, y) coordinates where agents can spawn
        """
        population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
        # STEP 1: Create random seed population (larger than target for selection)
        seed_population = []
        seed_size = population_size * 2  # Create 2x population for better selection diversity
        
        for i in range(seed_size):
            pos = spawn_positions[i % len(spawn_positions)]
            
            # Create completely random genome - maximum diversity
            genome = Genome(
                speed=random.random(),  # Random speed (0-100%)
                sense=random.random(),  # Random sense/sight range (0-100%)
                size=random.random()    # Random size (0-100%)
            )
            
            agent = Agent(pos[0], pos[1], self.agent_type, genome)
            
            # Assign random fitness to simulate initial performance variation
            # This creates selection pressure even before agents have "lived"
            agent.fitness = random.uniform(10, 100)  # Random fitness 10-100
            
            seed_population.append(agent)
        
        # STEP 2: Apply genetic algorithm operations to create actual starting population
        self.population = []
        
        # ELITISM: Keep some of the "best" random agents (top 20%)
        elite_count = max(1, seed_size // 5)
        elite_agents = sorted(seed_population, key=lambda x: x.fitness, reverse=True)[:elite_count]
        
        # Add elite agents to starting population (with new positions)
        for agent in elite_agents[:population_size//4]:  # Use 25% elites
            pos = random.choice(spawn_positions)
            new_agent = Agent(pos[0], pos[1], self.agent_type, agent.genome)
            self.population.append(new_agent)
        
        # REPRODUCTION: Generate remaining population through GA operations
        while len(self.population) < population_size:
            # Select two parents using tournament selection from seed population
            parent1 = self._tournament_selection_from_population(seed_population)
            parent2 = self._tournament_selection_from_population(seed_population)
            
            # CROSSOVER: Combine genetic material from two parents
            if random.random() < config.CROSSOVER_RATE:
                child_genome = self._crossover(parent1.genome, parent2.genome)
            else:
                # No crossover: clone better parent
                child_genome = parent1.genome if parent1.fitness > parent2.fitness else parent2.genome
            
            # MUTATION: Apply random changes to traits
            if random.random() < config.MUTATION_RATE:
                child_genome = self._mutate(child_genome)
            
            # Create new agent with evolved genome
            pos = random.choice(spawn_positions)
            child = Agent(pos[0], pos[1], self.agent_type, child_genome)
            self.population.append(child)
        
        print(f"Initialized {self.agent_type.value} with GA-processed population:")
        print(f"  Population size: {len(self.population)}")
        
        # Display trait diversity in starting population
        speeds = [agent.genome.speed for agent in self.population]
        senses = [agent.genome.sense for agent in self.population]
        sizes = [agent.genome.size for agent in self.population]
        
        print(f"  Speed range: {min(speeds):.2f} - {max(speeds):.2f}")
        print(f"  Sense range: {min(senses):.2f} - {max(senses):.2f}")
        print(f"  Size range: {min(sizes):.2f} - {max(sizes):.2f}")
    
    def _tournament_selection_from_population(self, population: List[Agent]) -> Agent:
        """
        Tournament selection from a specific population (used for initialization).
        
        Args:
            population: Population to select from
            
        Returns:
            Agent: Selected agent
        """
        tournament_size = min(config.TOURNAMENT_SIZE, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
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
            # EXTINCTION RECOVERY: Restart with random population
            print(f"Population extinction in {self.agent_type.value}! Restarting with random genomes...")
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
        population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
        while len(new_population) < population_size:
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
            speed=genome1.speed if random.random() < 0.5 else genome2.speed,
            sense=genome1.sense if random.random() < 0.5 else genome2.sense,
            size=genome1.size if random.random() < 0.5 else genome2.size
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
            speed=max(0, min(1, genome.speed + random.gauss(0, 0.1))),
            sense=max(0, min(1, genome.sense + random.gauss(0, 0.1))),
            size=max(0, min(1, genome.size + random.gauss(0, 0.1)))
        )
        return mutated_genome
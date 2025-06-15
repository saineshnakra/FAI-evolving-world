"""
Genetics module for the Artificial World Simulation - FIXED VERSION

Key fixes applied:
1. Improved mutation system with adaptive rates and minimum diversity protection
2. Better trait boundary enforcement
3. Enhanced crossover with intermediate recombination
4. Population size consistency fixes
5. Improved fitness function balance
6. Better selection pressure management
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
    """
    speed: float        # Movement speed multiplier (0.0-1.0)
    sense: float        # Vision/sight range multiplier (0.0-1.0) -> 50-150 pixels sight range
    size: float         # Size multiplier (0.0-1.0, larger = slower but more damage)
    
    @classmethod
    def random(cls) -> 'Genome':
        """Generate a random genome with all three traits randomized."""
        return cls(
            speed=random.uniform(0.1, 0.9),  # FIX: Avoid extreme values (0.0, 1.0)
            sense=random.uniform(0.1, 0.9),
            size=random.uniform(0.1, 0.9)
        )


class Agent:
    """Individual agent that exists in the world and evolves over time."""
    
    def __init__(self, x: int, y: int, agent_type: AgentType, genome: Genome = None):
        # Position and identity
        self.x = x
        self.y = y
        self.agent_type = agent_type
        self.genome = genome or Genome.random()
        
        # FIX: Enforce trait boundaries at creation
        self._enforce_trait_boundaries()
        
        # Survival and energy management
        self.energy = config.AGENT_ENERGY
        self.age = 0
        self.alive = True
        
        # Performance tracking for fitness calculation
        self.fitness = 0
        self.food_collected = 0
        self.attacks_made = 0
        self.successful_attacks = 0
        self.distance_traveled = 0
        
        # Track previous position for distance calculation
        self.prev_x = x
        self.prev_y = y
    
    def _enforce_trait_boundaries(self):
        """FIX: Ensure all traits stay within reasonable bounds."""
        self.genome.speed = max(0.01, min(0.99, self.genome.speed))    # Prevent 0.0 speed
        self.genome.sense = max(0.01, min(0.99, self.genome.sense))    # Prevent 0.0 sense
        self.genome.size = max(0.01, min(0.99, self.genome.size))      # Prevent 0.0 size
        
    def get_vision_range(self) -> float:
        """Calculate actual vision range based on sense trait."""
        return 50 + (self.genome.sense * 100)
    
    def get_movement_speed(self) -> float:
        """Calculate actual movement speed based on speed trait and size penalty."""
        size_penalty = 1.0 - (self.genome.size * 0.2)  # FIX: Reduced size penalty (was 0.3)
        return self.genome.speed * size_penalty
    
    def get_energy_cost(self) -> float:
        """Calculate energy cost per move based on size and speed."""
        base_cost = config.MOVEMENT_COST
        size_cost = 1.0 + (self.genome.size * 0.3)    # FIX: Reduced size cost (was 0.5)
        speed_cost = 1.0 + (self.genome.speed * 0.2)  # FIX: Reduced speed cost (was 0.3)
        return base_cost * size_cost * speed_cost

    def update(self, world_state: Dict[str, Any]) -> Tuple[int, int]:
        """Update agent for one simulation step and return desired movement."""
        if not self.alive or self.energy <= 0:
            self.alive = False
            return 0, 0
            
        # Age the agent and consume energy based on genetics
        self.age += 1
        energy_cost = self.get_energy_cost()
        self.energy -= energy_cost
        
        # Update fitness based on current performance
        self._calculate_fitness()
        
        # Make behavioral decision based on genetics and environment
        dx, dy = self._decide_action(world_state)
        
        # Track distance traveled for efficiency metrics
        if dx != 0 or dy != 0:
            self.distance_traveled += math.sqrt(dx*dx + dy*dy)
            self.prev_x, self.prev_y = self.x, self.y
        
        return dx, dy
    
    def attack_agent(self, target_agent: 'Agent') -> bool:
        """Implement attack mechanics for GA2 agents."""
        if self.agent_type != AgentType.AGGRESSIVE:
            return False
            
        if not target_agent.alive:
            return False
            
        # FIX: More balanced attack mechanics
        attack_power = self.genome.size * 40 + random.uniform(5, 25)    # 5-65 damage
        defense_power = target_agent.genome.size * 25 + random.uniform(5, 20)  # 5-45 defense
        
        self.attacks_made += 1
        
        if attack_power > defense_power:
            damage = min(attack_power - defense_power, 30)  # FIX: Cap damage to prevent instakills
            target_agent.energy -= damage
            
            # GA2 agents gain energy from successful attacks
            energy_gained = min(damage * 0.6, 25)  # FIX: Reduced energy gain to balance ecosystem
            self.energy += energy_gained
            self.successful_attacks += 1
            
            if target_agent.energy <= 0:
                target_agent.alive = False
                self.energy += 15  # FIX: Reduced kill bonus
            
            return True
        
        return False
    
    def eat_food(self, food_energy: float):
        """Food consumption with different efficiency for different agent types."""
        if self.agent_type == AgentType.COOPERATIVE:
            self.energy += food_energy
            self.food_collected += 1
        else:
            # GA2: Reduced but not terrible efficiency from food
            backup_energy = food_energy * 0.7  # FIX: Improved from 0.5 to 0.7
            self.energy += backup_energy
            self.food_collected += 1
    
    def _calculate_fitness(self):
        """FIX: Improved fitness function with better balance."""
        # Component 1: Survival Success
        survival_fitness = math.log(self.age + 1) * 3  # FIX: Reduced from 5 to 3
        
        # Component 2: Energy Management
        if self.age > 0:
            energy_ratio = self.energy / config.AGENT_ENERGY
            energy_sustainability = min(energy_ratio * 15, 15)  # FIX: Better energy scaling
        else:
            energy_sustainability = 0
        
        # Component 3: Strategic Performance
        strategic_fitness = 0
        
        if self.agent_type == AgentType.COOPERATIVE:
            # GA1 Strategy: Efficient resource acquisition
            if self.age > 0:
                food_efficiency = (self.food_collected / (self.age + 1)) * 20  # FIX: Increased reward
                strategic_fitness += food_efficiency
            
            # FIX: Reward balanced traits, not just extreme ones
            trait_balance = 1.0 - abs(0.5 - self.genome.speed) - abs(0.5 - self.genome.sense)
            strategic_fitness += trait_balance * 5
            
        else:
            # GA2 Strategy: Effective hunting
            if self.age > 0:
                hunt_effectiveness = (self.successful_attacks / (self.age + 1)) * 30
                strategic_fitness += hunt_effectiveness
            
            if self.attacks_made > 0:
                attack_success_rate = self.successful_attacks / self.attacks_made
                strategic_fitness += attack_success_rate * 10
            
            # FIX: Reward trait diversity in GA2 as well
            size_bonus = self.genome.size * self.successful_attacks * 1.5
            strategic_fitness += size_bonus
        
        # Component 4: Longevity Bonus
        if self.age > 100 and self.energy > 20:
            longevity_bonus = math.sqrt(self.age) * 2  # FIX: Reduced bonus
        else:
            longevity_bonus = 0
        
        # Final fitness calculation
        self.fitness = survival_fitness + energy_sustainability + strategic_fitness + longevity_bonus
        self.fitness = max(self.fitness, 1.0)
    
    def _decide_action(self, world_state: Dict[str, Any]) -> Tuple[int, int]:
        """Make behavioral decision using genetic traits effectively."""
        nearby_food = world_state.get('nearby_food', [])
        nearby_agents = world_state.get('nearby_agents', [])
        
        dx, dy = 0, 0
        vision_range = self.get_vision_range()
        
        # Filter objects by actual vision range
        visible_food = [(fx, fy) for fx, fy in nearby_food 
                       if math.sqrt((fx - self.x)**2 + (fy - self.y)**2) <= vision_range]
        
        visible_agents = [(ax, ay, agent_type) for ax, ay, agent_type in nearby_agents 
                         if math.sqrt((ax - self.x)**2 + (ay - self.y)**2) <= vision_range]
        
        # Agent-to-agent interactions
        if self.agent_type == AgentType.AGGRESSIVE and visible_agents:
            coop_agents = [(ax, ay, agent_type) for ax, ay, agent_type in visible_agents 
                          if agent_type == AgentType.COOPERATIVE]
            if coop_agents:
                target_x, target_y, _ = min(coop_agents, 
                    key=lambda agent_info: math.sqrt((agent_info[0] - self.x)**2 + (agent_info[1] - self.y)**2))
                
                dx, dy = self._calculate_movement_direction(target_x, target_y)
                return dx, dy
        
        elif self.agent_type == AgentType.COOPERATIVE and visible_agents:
            aggr_agents = [(ax, ay, agent_type) for ax, ay, agent_type in visible_agents 
                          if agent_type == AgentType.AGGRESSIVE]
            if aggr_agents:
                # FIX: More nuanced fleeing behavior based on traits
                flee_threshold = 60 + (self.genome.size * 40)  # Larger agents flee less readily
                if self.energy < flee_threshold:
                    threat_x, threat_y, _ = min(aggr_agents, 
                        key=lambda agent_info: math.sqrt((agent_info[0] - self.x)**2 + (agent_info[1] - self.y)**2))
                    
                    dx, dy = self._calculate_movement_direction(threat_x, threat_y, flee=True)
                    return dx, dy
        
        # Food-seeking behavior
        if visible_food:
            if self.agent_type == AgentType.COOPERATIVE or self.energy < 40:  # FIX: Adjusted threshold
                closest_food = min(visible_food, key=lambda f: 
                                 math.sqrt((f[0] - self.x)**2 + (f[1] - self.y)**2))
                
                dx, dy = self._calculate_movement_direction(closest_food[0], closest_food[1])
                return dx, dy
        
        # Random exploration based on speed trait
        exploration_chance = 0.15 + (self.genome.speed * 0.25)  # FIX: Reduced exploration
        if random.random() < exploration_chance:
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
        
        return dx, dy
    
    def _calculate_movement_direction(self, target_x: int, target_y: int, flee: bool = False) -> Tuple[int, int]:
        """Helper method to calculate movement direction towards or away from target."""
        if flee:
            dx = -1 if target_x > self.x else (1 if target_x < self.x else 0)
            dy = -1 if target_y > self.y else (1 if target_y < self.y else 0)
        else:
            dx = 1 if target_x > self.x else (-1 if target_x < self.x else 0)
            dy = 1 if target_y > self.y else (-1 if target_y < self.y else 0)
        
        return dx, dy


class GeneticAlgorithm:
    """FIX: Enhanced GA with better diversity management and population control."""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.generation = 0
        self.population: List[Agent] = []
        
        # Performance tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # FIX: Add diversity tracking
        self.diversity_history = []
        self.last_diversity = 0.0
    
    def initialize_population(self, spawn_positions: List[Tuple[int, int]]):
        """FIX: Improved population initialization with better diversity."""
        population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
        # Create initial population with enforced diversity
        self.population = []
        
        # Generate diverse starting population
        for i in range(population_size):
            pos = spawn_positions[i % len(spawn_positions)]
            
            # FIX: Create genomes with guaranteed diversity
            genome = Genome(
                speed=0.1 + (i / population_size) * 0.8,      # Spread across range
                sense=random.uniform(0.2, 0.8),               # More moderate range
                size=random.uniform(0.2, 0.8)                 # More moderate range
            )
            
            # Add some randomization to avoid too rigid patterns
            genome.speed += random.uniform(-0.1, 0.1)
            genome.sense += random.uniform(-0.1, 0.1)
            genome.size += random.uniform(-0.1, 0.1)
            
            agent = Agent(pos[0], pos[1], self.agent_type, genome)
            self.population.append(agent)
        
        print(f"Initialized {self.agent_type.value} population:")
        print(f"  Population size: {len(self.population)}")
        
        # Display trait diversity
        self._print_population_stats()
    
    def evolve_generation(self, spawn_positions: List[Tuple[int, int]]):
        """FIX: Enhanced evolution with diversity protection."""
        if not self.population:
            return
            
        alive_agents = [agent for agent in self.population if agent.alive]
        
        # FIX: Better extinction recovery
        if len(alive_agents) < 3:  # Changed from 0 to 3
            print(f"Population critically low in {self.agent_type.value}! Boosting diversity...")
            self._boost_population_diversity(spawn_positions, alive_agents)
            return
        
        # Record generation statistics
        fitnesses = [agent.fitness for agent in alive_agents]
        if fitnesses:
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        
        # FIX: Calculate and track genetic diversity
        diversity = self._calculate_genetic_diversity(alive_agents)
        self.diversity_history.append(diversity)
        self.last_diversity = diversity
        
        # Create new population
        new_population = []
        population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
        # FIX: Adaptive elitism based on diversity
        if diversity > 0.3:
            elite_count = max(1, population_size // 8)  # Normal elitism
        else:
            elite_count = max(1, population_size // 12)  # Reduced elitism when diversity is low
        
        elite = sorted(alive_agents, key=lambda x: x.fitness, reverse=True)[:elite_count]
        
        # Add elite agents
        for agent in elite:
            pos = random.choice(spawn_positions)
            new_agent = Agent(pos[0], pos[1], self.agent_type, agent.genome)
            new_population.append(new_agent)
        
        # Generate remaining population
        while len(new_population) < population_size:
            if len(alive_agents) >= 2:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # FIX: Adaptive crossover rate based on diversity
                crossover_rate = config.CROSSOVER_RATE
                if diversity < 0.2:
                    crossover_rate *= 1.5  # Increase crossover when diversity is low
                
                if random.random() < crossover_rate:
                    child_genome = self._enhanced_crossover(parent1.genome, parent2.genome)
                else:
                    child_genome = parent1.genome
            else:
                child_genome = alive_agents[0].genome if alive_agents else Genome.random()
            
            # FIX: Adaptive mutation rate based on diversity
            mutation_rate = self._adaptive_mutation_rate(diversity)
            if random.random() < mutation_rate:
                child_genome = self._enhanced_mutate(child_genome, diversity)
            
            pos = random.choice(spawn_positions)
            child = Agent(pos[0], pos[1], self.agent_type, child_genome)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # FIX: Print diversity warning if needed
        if diversity < 0.1:
            print(f"Warning: Low genetic diversity ({diversity:.3f}) in {self.agent_type.value}")
    
    def _boost_population_diversity(self, spawn_positions: List[Tuple[int, int]], survivors: List[Agent]):
        """FIX: Emergency diversity boost when population is critically low."""
        population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
        self.population = []
        
        # Keep survivors
        for agent in survivors:
            pos = random.choice(spawn_positions)
            new_agent = Agent(pos[0], pos[1], self.agent_type, agent.genome)
            self.population.append(new_agent)
        
        # Add highly diverse new agents
        while len(self.population) < population_size:
            pos = random.choice(spawn_positions)
            
            # Create diverse genome
            genome = Genome(
                speed=random.uniform(0.1, 0.9),
                sense=random.uniform(0.1, 0.9),
                size=random.uniform(0.1, 0.9)
            )
            
            new_agent = Agent(pos[0], pos[1], self.agent_type, genome)
            self.population.append(new_agent)
    
    def _calculate_genetic_diversity(self, agents: List[Agent]) -> float:
        """FIX: Calculate actual genetic diversity of population."""
        if len(agents) < 2:
            return 0.0
        
        # Calculate variance for each trait
        speeds = [agent.genome.speed for agent in agents]
        senses = [agent.genome.sense for agent in agents]
        sizes = [agent.genome.size for agent in agents]
        
        speed_var = np.var(speeds) if len(speeds) > 1 else 0.0
        sense_var = np.var(senses) if len(senses) > 1 else 0.0
        size_var = np.var(sizes) if len(sizes) > 1 else 0.0
        
        # Average variance across all traits
        return (speed_var + sense_var + size_var) / 3.0
    
    def _adaptive_mutation_rate(self, diversity: float) -> float:
        """FIX: Adaptive mutation rate based on population diversity."""
        base_rate = config.MUTATION_RATE
        
        if diversity < 0.1:
            return base_rate * 3.0    # Triple mutation when diversity is very low
        elif diversity < 0.2:
            return base_rate * 2.0    # Double mutation when diversity is low
        elif diversity > 0.5:
            return base_rate * 0.7    # Reduce mutation when diversity is high
        else:
            return base_rate
    
    def _enhanced_crossover(self, genome1: Genome, genome2: Genome) -> Genome:
        """FIX: Enhanced crossover with intermediate recombination."""
        # 50% chance of uniform crossover, 50% chance of intermediate recombination
        if random.random() < 0.5:
            # Uniform crossover (original method)
            return Genome(
                speed=genome1.speed if random.random() < 0.5 else genome2.speed,
                sense=genome1.sense if random.random() < 0.5 else genome2.sense,
                size=genome1.size if random.random() < 0.5 else genome2.size
            )
        else:
            # Intermediate recombination (blend traits)
            alpha = random.uniform(0.3, 0.7)  # Blending factor
            return Genome(
                speed=alpha * genome1.speed + (1 - alpha) * genome2.speed,
                sense=alpha * genome1.sense + (1 - alpha) * genome2.sense,
                size=alpha * genome1.size + (1 - alpha) * genome2.size
            )
    
    def _enhanced_mutate(self, genome: Genome, diversity: float) -> Genome:
        """FIX: Enhanced mutation with diversity-based strength."""
        # Stronger mutations when diversity is low
        if diversity < 0.1:
            mutation_strength = 0.2    # Large mutations
        elif diversity < 0.2:
            mutation_strength = 0.15   # Medium mutations
        else:
            mutation_strength = 0.1    # Normal mutations
        
        mutated_genome = Genome(
            speed=max(0.01, min(0.99, genome.speed + random.gauss(0, mutation_strength))),
            sense=max(0.01, min(0.99, genome.sense + random.gauss(0, mutation_strength))),
            size=max(0.01, min(0.99, genome.size + random.gauss(0, mutation_strength)))
        )
        return mutated_genome
    
    def _tournament_selection(self) -> Agent:
        """Tournament selection with fallback handling."""
        alive_agents = [a for a in self.population if a.alive]
        
        if not alive_agents:
            return random.choice(self.population)
        
        tournament_size = min(config.TOURNAMENT_SIZE, len(alive_agents))
        if tournament_size == 0:
            return random.choice(alive_agents)
        
        tournament = random.sample(alive_agents, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _print_population_stats(self):
        """Helper method to print population statistics."""
        speeds = [agent.genome.speed for agent in self.population]
        senses = [agent.genome.sense for agent in self.population]
        sizes = [agent.genome.size for agent in self.population]
        
        print(f"  Speed range: {min(speeds):.3f} - {max(speeds):.3f} (var: {np.var(speeds):.3f})")
        print(f"  Sense range: {min(senses):.3f} - {max(senses):.3f} (var: {np.var(senses):.3f})")
        print(f"  Size range: {min(sizes):.3f} - {max(sizes):.3f} (var: {np.var(sizes):.3f})")
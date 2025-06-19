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
    
# Additional improvements for the genetics.py file

    def attack_agent(self, target_agent: 'Agent') -> bool:
        """Implement attack mechanics for GA2 agents - FURTHER IMPROVED."""
        if self.agent_type != AgentType.AGGRESSIVE:
            return False
            
        if not target_agent.alive:
            return False
        
        # IMPROVEMENT 1: Make attack success depend more on traits
        # Attacker advantages
        attack_speed_bonus = self.genome.speed * 20  # Speed helps with accuracy
        attack_size_bonus = self.genome.size * 30    # Size provides power
        
        # Defender advantages  
        defense_speed_bonus = target_agent.genome.speed * 25  # Speed helps dodge
        defense_sense_bonus = target_agent.genome.sense * 15  # Sense helps detect attacks
        
        attack_power = attack_size_bonus + attack_speed_bonus + random.uniform(10, 30)
        defense_power = defense_speed_bonus + defense_sense_bonus + random.uniform(10, 25)
        
        self.attacks_made += 1
        
        # IMPROVEMENT 2: Graduated success levels instead of binary
        success_margin = attack_power - defense_power
        
        if success_margin > 5:  # Clear success
            damage = min(success_margin * 0.8, 25)
            energy_gained = min(damage * 0.5, 20)
            self.energy += energy_gained
            self.successful_attacks += 1
            target_agent.energy -= damage
            
            if target_agent.energy <= 0:
                target_agent.alive = False
                self.energy += 10  # Kill bonus
            
            return True
        elif success_margin > -5:  # Partial success
            damage = max(5, success_margin * 0.3)
            target_agent.energy -= damage
            self.successful_attacks += 0.5  # Partial credit
            return True
        
        # IMPROVEMENT 3: Attack cost for failed attempts
        self.energy -= 2  # Small energy cost for failed attacks
        return False
    def eat_food(self, food_energy: float):
        """Food consumption with different efficiency for different agent types."""
        if self.agent_type == AgentType.COOPERATIVE:
            self.energy += food_energy
            self.food_collected += 1
        elif self.agent_type == AgentType.AGGRESSIVE and self.energy < 0.2 * config.AGENT_ENERGY:
            # GA2: Reduced but not terrible efficiency from food
            backup_energy = food_energy * 0.7  # FIX: Improved from 0.5 to 0.7
            self.energy += backup_energy
            self.food_collected += 1

    def _calculate_fitness(self):
        """IMPROVED: More sophisticated fitness calculation."""
        # Base survival
        survival_fitness = math.log(self.age + 1) * 2.5
        
        # Energy management with sustainability focus
        energy_ratio = self.energy / config.AGENT_ENERGY
        energy_sustainability = min(energy_ratio * 12, 12)
        
        strategic_fitness = 0
        
        if self.agent_type == AgentType.COOPERATIVE:
            # IMPROVEMENT 4: Reward survival efficiency over pure resource collection
            if self.age > 50:  # Only after surviving reasonable time
                survival_efficiency = (self.energy + self.food_collected * 10) / math.sqrt(self.age)
                strategic_fitness += survival_efficiency * 1.5
            
            # IMPROVEMENT 5: Reward balanced traits that work well together
            # Speed-sense synergy (fast agents need good senses)
            if self.genome.speed > 0.6 and self.genome.sense > 0.4:
                strategic_fitness += 8
            # Size-sense balance (bigger agents can afford lower speed but need senses)
            if self.genome.size > 0.6 and self.genome.sense > 0.5:
                strategic_fitness += 6
            
            # Penalize extreme trait combinations that don't make sense
            if self.genome.speed > 0.8 and self.genome.size > 0.8:  # Fast AND big is unrealistic
                strategic_fitness -= 5
                
        else:  # GA2 (Predators)
            # IMPROVEMENT 6: More nuanced hunting fitness
            if self.age > 30:
                hunt_efficiency = (self.successful_attacks * 15) / math.sqrt(self.age)
                strategic_fitness += hunt_efficiency
            
            # IMPROVEMENT 7: Reward diverse hunting strategies
            if self.attacks_made > 0:
                success_rate = self.successful_attacks / self.attacks_made
                if 0.3 <= success_rate <= 0.7:  # Reward moderate success rates
                    strategic_fitness += success_rate * 15
                else:  # Penalize too high or too low success rates
                    strategic_fitness += success_rate * 10
            
            # Trait synergy rewards for predators
            if self.genome.size > 0.5 and self.genome.speed > 0.4:  # Balanced hunter
                strategic_fitness += 8
        
        # IMPROVEMENT 8: Age-based fitness plateau to prevent runaway selection
        if self.age > 200:
            age_penalty = (self.age - 200) * 0.02  # Small penalty for extreme age
            survival_fitness -= age_penalty
        
        self.fitness = max(survival_fitness + energy_sustainability + strategic_fitness, 1.0)

    def _decide_action(self, world_state: Dict[str, Any]) -> Tuple[int, int]:
        """IMPROVED: More sophisticated decision making."""
        nearby_food = world_state.get('nearby_food', [])
        nearby_agents = world_state.get('nearby_agents', [])
        
        vision_range = self.get_vision_range()
        
        # Filter by vision
        visible_food = [(fx, fy) for fx, fy in nearby_food 
                    if math.sqrt((fx - self.x)**2 + (fy - self.y)**2) <= vision_range]
        
        visible_agents = [(ax, ay, agent_type) for ax, ay, agent_type in nearby_agents 
                        if math.sqrt((ax - self.x)**2 + (ay - self.y)**2) <= vision_range]
        
        # IMPROVEMENT 9: Energy-based decision thresholds
        energy_ratio = self.energy / config.AGENT_ENERGY
        
        if self.agent_type == AgentType.AGGRESSIVE:
            # Predators become more food-focused when energy is low
            if energy_ratio < 0.3 and visible_food:
                closest_food = min(visible_food, key=lambda f: 
                                math.sqrt((f[0] - self.x)**2 + (f[1] - self.y)**2))
                return self._calculate_movement_direction(closest_food[0], closest_food[1])
            
            # Normal hunting behavior
            if visible_agents and energy_ratio > 0.2:
                prey = [(ax, ay, at) for ax, ay, at in visible_agents 
                    if at == AgentType.COOPERATIVE]
                if prey:
                    # IMPROVEMENT 10: Smart target selection based on traits
                    def target_priority(prey_info):
                        px, py, _ = prey_info
                        distance = math.sqrt((px - self.x)**2 + (py - self.y)**2)
                        # Prefer closer, easier targets (this would need agent reference)
                        return distance
                    
                    target_x, target_y, _ = min(prey, key=target_priority)
                    return self._calculate_movement_direction(target_x, target_y)
        
        else:  # Cooperative agents
            # IMPROVEMENT 11: Dynamic threat assessment
            if visible_agents:
                predators = [(ax, ay, at) for ax, ay, at in visible_agents 
                            if at == AgentType.AGGRESSIVE]
                if predators:
                    closest_pred_dist = min(math.sqrt((px - self.x)**2 + (py - self.y)**2) 
                                        for px, py, _ in predators)
                    
                    # Flee threshold based on traits and energy
                    flee_threshold = 40 + (self.genome.size * 30) + (energy_ratio * 20)
                    threat_distance = 30 + (self.genome.sense * 40)  # Better senses = detect threats earlier
                    
                    if closest_pred_dist < threat_distance and self.energy < flee_threshold:
                        threat_x, threat_y, _ = min(predators, key=lambda p: 
                            math.sqrt((p[0] - self.x)**2 + (p[1] - self.y)**2))
                        return self._calculate_movement_direction(threat_x, threat_y, flee=True)
            
            # Food seeking with energy consideration
            if visible_food and (energy_ratio < 0.6 or not visible_agents):
                closest_food = min(visible_food, key=lambda f: 
                                math.sqrt((f[0] - self.x)**2 + (f[1] - self.y)**2))
                return self._calculate_movement_direction(closest_food[0], closest_food[1])
        
        # IMPROVEMENT 12: Smarter exploration
        exploration_chance = 0.1 + (self.genome.speed * 0.2) + (energy_ratio * 0.1)
        if random.random() < exploration_chance:
            # Bias exploration away from world edges
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            return dx, dy
        
        return 0, 0
        
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
    
    # def evolve_generation(self, spawn_positions: List[Tuple[int, int]]):
    #     """FIX: Enhanced evolution with diversity protection."""
    #     if not self.population:
    #         return
            
    #     alive_agents = [agent for agent in self.population if agent.alive]
        
    #     # FIX: Better extinction recovery
    #     if len(alive_agents) < 3:  # Changed from 0 to 3
    #         print(f"Population critically low in {self.agent_type.value}! Boosting diversity...")
    #         self._boost_population_diversity(spawn_positions, alive_agents)
    #         return
        
    #     # Record generation statistics
    #     fitnesses = [agent.fitness for agent in alive_agents]
    #     if fitnesses:
    #         self.best_fitness_history.append(max(fitnesses))
    #         self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        
    #     # FIX: Calculate and track genetic diversity
    #     diversity = self._calculate_genetic_diversity(alive_agents)
    #     self.diversity_history.append(diversity)
    #     self.last_diversity = diversity
        
    #     # Create new population
    #     new_population = []
    #     population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
    #     # FIX: Adaptive elitism based on diversity
    #     if diversity > 0.3:
    #         elite_count = max(1, population_size // 8)  # Normal elitism
    #     else:
    #         elite_count = max(1, population_size // 12)  # Reduced elitism when diversity is low
        
    #     elite = sorted(alive_agents, key=lambda x: x.fitness, reverse=True)[:elite_count]
        
    #     # Add elite agents
    #     for agent in elite:
    #         pos = random.choice(spawn_positions)
    #         new_agent = Agent(pos[0], pos[1], self.agent_type, agent.genome)
    #         new_population.append(new_agent)
        
    #     # Generate remaining population
    #     while len(new_population) < population_size:
    #         if len(alive_agents) >= 2:
    #             parent1 = self._tournament_selection()
    #             parent2 = self._tournament_selection()
                
    #             # FIX: Adaptive crossover rate based on diversity
    #             crossover_rate = config.CROSSOVER_RATE
    #             if diversity < 0.2:
    #                 crossover_rate *= 1.5  # Increase crossover when diversity is low
                
    #             if random.random() < crossover_rate:
    #                 child_genome = self._enhanced_crossover(parent1.genome, parent2.genome)
    #             else:
    #                 child_genome = parent1.genome
    #         else:
    #             child_genome = alive_agents[0].genome if alive_agents else Genome.random()
            
    #         # FIX: Adaptive mutation rate based on diversity
    #         mutation_rate = self._adaptive_mutation_rate(diversity)
    #         if random.random() < mutation_rate:
    #             child_genome = self._enhanced_mutate(child_genome, diversity)
            
    #         pos = random.choice(spawn_positions)
    #         child = Agent(pos[0], pos[1], self.agent_type, child_genome)
    #         new_population.append(child)
        
    #     self.population = self.population + new_population
    #     self.generation += 1
        
    #     # FIX: Print diversity warning if needed
    #     if diversity < 0.1:
    #         print(f"Warning: Low genetic diversity ({diversity:.3f}) in {self.agent_type.value}")
    
    # def _boost_population_diversity(self, spawn_positions: List[Tuple[int, int]], survivors: List[Agent]):
    #     """FIX: Emergency diversity boost when population is critically low."""
    #     population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
    #     self.population = []
        
    #     # Keep survivors
    #     for agent in survivors:
    #         pos = random.choice(spawn_positions)
    #         new_agent = Agent(pos[0], pos[1], self.agent_type, agent.genome)
    #         self.population.append(new_agent)
        
    #     # Add highly diverse new agents
    #     while len(self.population) < population_size:
    #         pos = random.choice(spawn_positions)
            
    #         # Create diverse genome
    #         genome = Genome(
    #             speed=random.uniform(0.1, 0.9),
    #             sense=random.uniform(0.1, 0.9),
    #             size=random.uniform(0.1, 0.9)
    #         )
            
    #         new_agent = Agent(pos[0], pos[1], self.agent_type, genome)
    #         self.population.append(new_agent)
    
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
    
    # CHANGE 1: Replace the evolve_generation method (starting around line 427)
    def evolve_generation(self, spawn_positions: List[Tuple[int, int]]):
        """Evolution with controlled overlapping generations."""
        if not self.population:
            return
            
        # Filter alive agents ONCE
        alive_agents = [agent for agent in self.population if agent.alive]
        
        # Handle extinction/low population
        if len(alive_agents) < 3:
            print(f"Population critically low in {self.agent_type.value}! Boosting diversity...")
            self._boost_population_diversity(spawn_positions, alive_agents)
            return
        
        # Record generation statistics
        fitnesses = [agent.fitness for agent in alive_agents]
        if fitnesses:
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        
        # Calculate and track genetic diversity
        diversity = self._calculate_genetic_diversity(alive_agents)
        self.diversity_history.append(diversity)
        self.last_diversity = diversity
        
        # Population parameters
        base_pop_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        max_population = int(base_pop_size * 1.5)  # Allow 50% overpopulation
        offspring_count = base_pop_size  # Always produce full generation of offspring
        
        # Population control - remove weakest if population would exceed limit
        survivors = alive_agents
        if len(alive_agents) + offspring_count > max_population:
            # Sort by fitness (ascending) so we can remove weakest
            alive_agents.sort(key=lambda x: x.fitness)
            
            # Calculate how many to keep
            agents_to_keep = max_population - offspring_count
            survivors = alive_agents[-agents_to_keep:]  # Keep the best
            
            print(f"  Population control: Keeping {agents_to_keep} best survivors out of {len(alive_agents)}")
        
        # Create offspring population
        offspring = []
        
        # Adaptive elitism based on diversity
        if diversity > 0.3:
            elite_count = max(1, offspring_count // 8)
        else:
            elite_count = max(1, offspring_count // 12)
        
        elite = sorted(survivors, key=lambda x: x.fitness, reverse=True)[:elite_count]
        
        # Add elite offspring
        for agent in elite:
            if len(offspring) >= offspring_count:
                break
            pos = random.choice(spawn_positions)
            new_agent = Agent(pos[0], pos[1], self.agent_type, agent.genome)
            offspring.append(new_agent)
        
        # Generate remaining offspring
        while len(offspring) < offspring_count:
            if len(survivors) >= 2:
                # IMPORTANT: Pass survivors to selection, not self.population
                parent1 = self._tournament_selection_from_pool(survivors)
                parent2 = self._tournament_selection_from_pool(survivors)
                
                # Adaptive crossover rate based on diversity
                crossover_rate = config.CROSSOVER_RATE
                if diversity < 0.2:
                    crossover_rate *= 1.5
                
                if random.random() < crossover_rate:
                    child_genome = self._enhanced_crossover(parent1.genome, parent2.genome)
                else:
                    child_genome = parent1.genome if random.random() < 0.5 else parent2.genome
            else:
                child_genome = survivors[0].genome if survivors else Genome.random()
            
            # Adaptive mutation
            mutation_rate = self._adaptive_mutation_rate(diversity)
            if random.random() < mutation_rate:
                child_genome = self._enhanced_mutate(child_genome, diversity)
            
            pos = random.choice(spawn_positions)
            child = Agent(pos[0], pos[1], self.agent_type, child_genome)
            offspring.append(child)
        
        # CRITICAL CHANGE: Combine survivors + offspring (not old population + offspring)
        self.population = survivors + offspring
        self.generation += 1
        
        # Print generation summary
        print(f"\nGeneration {self.generation} - {self.agent_type.value}:")
        print(f"  Survivors: {len(survivors)}, Offspring: {len(offspring)}, Total: {len(self.population)}")
        print(f"  Diversity: {diversity:.3f}")
        
        # Diversity warning if needed
        if diversity < 0.1:
            print(f"Warning: Low genetic diversity ({diversity:.3f}) in {self.agent_type.value}")


    # CHANGE 2: Replace _tournament_selection method (around line 701)
    def _tournament_selection(self) -> Agent:
        """Tournament selection from alive agents only."""
        alive_agents = [a for a in self.population if a.alive]
        return self._tournament_selection_from_pool(alive_agents)


    # CHANGE 3: Add this new method after _tournament_selection
    def _tournament_selection_from_pool(self, candidate_pool: List[Agent]) -> Agent:
        """Tournament selection from a specific pool of agents."""
        if not candidate_pool:
            # Emergency fallback
            return Agent(0, 0, self.agent_type, Genome.random())
        
        tournament_size = min(config.TOURNAMENT_SIZE, len(candidate_pool))
        if tournament_size == 0:
            return random.choice(candidate_pool)
        
        tournament = random.sample(candidate_pool, tournament_size)
        return max(tournament, key=lambda x: x.fitness)


    # CHANGE 4: Update _boost_population_diversity method (around line 543)
    def _boost_population_diversity(self, spawn_positions: List[Tuple[int, int]], survivors: List[Agent]):
        """Emergency diversity boost when population is critically low."""
        population_size = config.POPULATION_SIZE_GA1 if self.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
        # Start with empty population
        self.population = []
        
        # Keep survivors (they stay alive)
        self.population.extend(survivors)
        
        # Add highly diverse new agents to reach population size
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
    
    def _print_population_stats(self):
        """Helper method to print population statistics."""
        speeds = [agent.genome.speed for agent in self.population]
        senses = [agent.genome.sense for agent in self.population]
        sizes = [agent.genome.size for agent in self.population]
        
        print(f"  Speed range: {min(speeds):.3f} - {max(speeds):.3f} (var: {np.var(speeds):.3f})")
        print(f"  Sense range: {min(senses):.3f} - {max(senses):.3f} (var: {np.var(senses):.3f})")
        print(f"  Size range: {min(sizes):.3f} - {max(sizes):.3f} (var: {np.var(sizes):.3f})")
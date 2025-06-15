"""
Analytics module for the Artificial World Simulation.

Contains comprehensive data collection and analysis systems for tracking
evolutionary progress and comparing genetic algorithm strategies.
"""

import numpy as np
import json
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional
from collections import defaultdict

from config import config


class GenomeAnalyzer:
    """
    Analyzes genome evolution patterns and trait distributions.
    
    This class tracks how genetic traits change over generations,
    identifying evolutionary trends and successful adaptations.
    """
    
    def __init__(self):
        """Initialize genome tracking systems."""
        self.trait_history = {
            'GA1': defaultdict(list),  # Track each trait over generations
            'GA2': defaultdict(list)
        }
        self.diversity_history = {
            'GA1': [],  # Genetic diversity over time
            'GA2': []
        }
        
        # NEW: Track trait correlations with fitness
        self.trait_fitness_correlations = {
            'GA1': defaultdict(list),
            'GA2': defaultdict(list)
        }
    
    def analyze_population_genetics(self, agents: List, population_name: str) -> Dict[str, Any]:
        """
        Analyze the genetic composition of a population.
        
        Args:
            agents: List of agents to analyze
            population_name: 'GA1' or 'GA2'
            
        Returns:
            Dictionary containing genetic analysis results
        """
        if not agents:
            return {
                'avg_traits': {},
                'trait_diversity': {},
                'genetic_diversity': 0.0,
                'trait_fitness_correlations': {}
            }
        
        # Extract all genomes and fitness values
        genomes = [agent.genome for agent in agents]
        fitness_values = [agent.fitness for agent in agents]
        
        # Calculate average trait values
        avg_traits = {
            'speed': np.mean([g.speed for g in genomes]),
            'sense': np.mean([g.sense for g in genomes]),
            'size': np.mean([g.size for g in genomes])
        }
        
        # Calculate trait diversity (standard deviation)
        trait_diversity = {
            'speed': np.std([g.speed for g in genomes]),
            'sense': np.std([g.sense for g in genomes]),
            'size': np.std([g.size for g in genomes])
        }
        
        # NEW: Calculate trait-fitness correlations
        trait_fitness_correlations = {}
        if len(agents) > 2:  # Need at least 3 agents for meaningful correlation
            speed_values = [g.speed for g in genomes]
            sense_values = [g.sense for g in genomes]
            size_values = [g.size for g in genomes]
            
            trait_fitness_correlations = {
                'speed': np.corrcoef(speed_values, fitness_values)[0, 1] if np.std(speed_values) > 0 else 0,
                'sense': np.corrcoef(sense_values, fitness_values)[0, 1] if np.std(sense_values) > 0 else 0,
                'size': np.corrcoef(size_values, fitness_values)[0, 1] if np.std(size_values) > 0 else 0
            }
            
            # Store correlation history
            for trait, correlation in trait_fitness_correlations.items():
                if not np.isnan(correlation):
                    self.trait_fitness_correlations[population_name][trait].append(correlation)
        
        # Calculate overall genetic diversity
        all_traits = []
        for genome in genomes:
            traits = [genome.speed, genome.sense, genome.size]
            all_traits.append(traits)
        
        if len(all_traits) > 1:
            genetic_diversity = np.mean([np.std(trait_values) for trait_values in zip(*all_traits)])
        else:
            genetic_diversity = 0.0
        
        # Store historical data
        for trait, value in avg_traits.items():
            self.trait_history[population_name][trait].append(value)
        self.diversity_history[population_name].append(genetic_diversity)
        
        return {
            'avg_traits': avg_traits,
            'trait_diversity': trait_diversity,
            'genetic_diversity': genetic_diversity,
            'trait_fitness_correlations': trait_fitness_correlations
        }


class Analytics:
    """
    Comprehensive data collection and analysis system for the simulation.
    
    This class tracks:
    - Generation-by-generation performance statistics  
    - Population dynamics and survival rates
    - Behavioral evolution trends
    - Comparative performance between GA strategies
    - Genome evolution patterns
    - Energy flow and resource competition
    - Combat effectiveness and predator-prey dynamics
    
    The analytics system is essential for understanding which evolutionary
    strategies are most effective and how behaviors emerge over time.
    """
    
    def __init__(self):
        """Initialize analytics tracking systems."""
        # Raw data storage
        self.generation_data = []           # Detailed stats for each generation
        self.frame_data = []               # Frame-by-frame data for real-time analysis
        
        # Comparative analysis between the two GA strategies
        self.performance_comparison = {
            'GA1': [],  # Cooperative strategy performance over time
            'GA2': []   # Aggressive strategy performance over time
        }
        
        # Survival and extinction tracking
        self.survival_history = {
            'GA1': [],  # Number of survivors each generation
            'GA2': []
        }
        
        self.extinction_events = {
            'GA1': [],  # Generations where GA1 went extinct
            'GA2': []   # Generations where GA2 went extinct
        }
        
        # Genome analysis system
        self.genome_analyzer = GenomeAnalyzer()
        
        # NEW: Enhanced behavioral tracking
        self.behavior_patterns = {
            'GA1': {
                'exploration_rate': [], 
                'food_efficiency': [], 
                'energy_management': [],
                'collision_avoidance': [],
                'foraging_success_rate': []
            },
            'GA2': {
                'exploration_rate': [], 
                'attack_efficiency': [], 
                'hunt_success_rate': [],
                'energy_from_combat': [],
                'territorial_behavior': []
            }
        }
        
        # NEW: Energy flow analysis
        self.energy_flow = {
            'total_food_energy_consumed': [],
            'total_combat_energy_gained': [],
            'energy_loss_rate': [],
            'resource_competition_index': []
        }
        
        # NEW: Population dynamics
        self.population_dynamics = {
            'predator_prey_ratio': [],
            'population_pressure': [],
            'resource_availability': [],
            'generation_duration': []
        }
        
        # NEW: Real-time frame tracking
        self.current_frame_data = {
            'frame': 0,
            'generation': 0,
            'interactions': 0,
            'attacks': 0,
            'food_consumed': 0
        }
    
    def record_frame_data(self, world, ga1, ga2):
        """
        Record frame-by-frame data for real-time analysis.
        
        Args:
            world: Current world state
            ga1: GA1 population
            ga2: GA2 population
        """
        self.current_frame_data['frame'] += 1
        self.current_frame_data['generation'] = max(ga1.generation, ga2.generation)
        
        # Count current frame interactions
        ga1_alive = [a for a in ga1.population if a.alive]
        ga2_alive = [a for a in ga2.population if a.alive]
        
        frame_record = {
            'frame': self.current_frame_data['frame'],
            'generation': self.current_frame_data['generation'],
            'ga1_alive': len(ga1_alive),
            'ga2_alive': len(ga2_alive),
            'total_energy_ga1': sum(a.energy for a in ga1_alive),
            'total_energy_ga2': sum(a.energy for a in ga2_alive),
            'food_count': len(world.food),
            'world_time': world.generation_time
        }
        
        # Keep only last 1000 frames to prevent memory issues
        self.frame_data.append(frame_record)
        if len(self.frame_data) > 1000:
            self.frame_data.pop(0)
    
    def record_generation(self, generation: int, ga1, ga2, world=None):
        """
        Record comprehensive statistics for the current generation.
        
        This function captures a snapshot of both populations' performance,
        allowing for detailed analysis of evolutionary trends and strategy
        effectiveness over time.
        
        Args:
            generation: Current generation number
            ga1: Cooperative genetic algorithm population
            ga2: Aggressive genetic algorithm population
            world: World state for environmental analysis
        """
        # Extract living agents from each population
        ga1_agents = [a for a in ga1.population if a.alive]
        ga2_agents = [a for a in ga2.population if a.alive]
        
        # Track survival numbers
        self.survival_history['GA1'].append(len(ga1_agents))
        self.survival_history['GA2'].append(len(ga2_agents))
        
        # Track extinction events
        if len(ga1_agents) == 0:
            self.extinction_events['GA1'].append(generation)
        if len(ga2_agents) == 0:
            self.extinction_events['GA2'].append(generation)
        
        # Calculate detailed statistics for each population
        ga1_stats = self._calculate_population_stats(ga1_agents, "GA1")
        ga2_stats = self._calculate_population_stats(ga2_agents, "GA2")
        
        # Perform genome analysis
        ga1_genetics = self.genome_analyzer.analyze_population_genetics(ga1_agents, 'GA1')
        ga2_genetics = self.genome_analyzer.analyze_population_genetics(ga2_agents, 'GA2')
        
        # Analyze emergent behaviors
        ga1_behaviors = self._analyze_emergent_behaviors(ga1_agents, 'GA1')
        ga2_behaviors = self._analyze_emergent_behaviors(ga2_agents, 'GA2')
        
        # NEW: Analyze population dynamics
        population_dynamics = self._analyze_population_dynamics(ga1_agents, ga2_agents, world)
        
        # NEW: Analyze energy flow
        energy_flow = self._analyze_energy_flow(ga1_agents, ga2_agents)
        
        # Create comprehensive generation record
        generation_record = {
            'generation': generation,
            'ga1': {**ga1_stats, 'genetics': ga1_genetics, 'behaviors': ga1_behaviors},
            'ga2': {**ga2_stats, 'genetics': ga2_genetics, 'behaviors': ga2_behaviors},
            'population_dynamics': population_dynamics,
            'energy_flow': energy_flow,
            'total_agents': len(ga1_agents) + len(ga2_agents),
            'timestamp': self._get_timestamp()
        }
        
        # Store data for analysis
        self.generation_data.append(generation_record)
        self.performance_comparison['GA1'].append(ga1_stats['avg_fitness'])
        self.performance_comparison['GA2'].append(ga2_stats['avg_fitness'])
        
        # Store population dynamics
        self.population_dynamics['predator_prey_ratio'].append(
            len(ga2_agents) / max(1, len(ga1_agents))
        )
        
        # Reset frame counter for new generation
        self.current_frame_data['frame'] = 0
    
    def _analyze_population_dynamics(self, ga1_agents: List, ga2_agents: List, world) -> Dict[str, float]:
        """
        NEW: Analyze population dynamics and ecosystem health.
        
        Args:
            ga1_agents: GA1 population
            ga2_agents: GA2 population
            world: World state
            
        Returns:
            Population dynamics metrics
        """
        total_pop = len(ga1_agents) + len(ga2_agents)
        
        dynamics = {
            'total_population': total_pop,
            'ga1_proportion': len(ga1_agents) / max(1, total_pop),
            'ga2_proportion': len(ga2_agents) / max(1, total_pop),
            'predator_prey_ratio': len(ga2_agents) / max(1, len(ga1_agents)),
            'population_density': total_pop / (config.WORLD_WIDTH * config.WORLD_HEIGHT / 10000),
            'resource_availability': len(world.food) / max(1, total_pop) if world else 0,
            'resource_pressure': max(0, total_pop - len(world.food)) if world else 0
        }
        
        return dynamics
    
    def _analyze_energy_flow(self, ga1_agents: List, ga2_agents: List) -> Dict[str, float]:
        """
        NEW: Analyze energy flow through the ecosystem.
        
        Args:
            ga1_agents: GA1 population
            ga2_agents: GA2 population
            
        Returns:
            Energy flow metrics
        """
        total_food_consumed = sum(a.food_collected for a in ga1_agents + ga2_agents)
        total_attacks = sum(a.attacks_made for a in ga2_agents)
        total_successful_attacks = sum(getattr(a, 'successful_attacks', 0) for a in ga2_agents)
        
        energy_flow = {
            'total_food_consumed': total_food_consumed,
            'total_attacks_made': total_attacks,
            'total_successful_attacks': total_successful_attacks,
            'attack_success_rate': total_successful_attacks / max(1, total_attacks),
            'energy_from_hunting': total_successful_attacks * 20,  # Approximate energy gain
            'energy_from_foraging': total_food_consumed * config.EATING_REWARD,
            'total_energy_in_system': sum(a.energy for a in ga1_agents + ga2_agents)
        }
        
        return energy_flow
    
    def _analyze_emergent_behaviors(self, agents: List, population_name: str) -> Dict[str, float]:
        """
        Analyze emergent behavioral patterns in the population.
        
        Args:
            agents: List of agents to analyze
            population_name: 'GA1' or 'GA2'
            
        Returns:
            Dictionary containing behavioral metrics
        """
        if not agents:
            return {
                'exploration_rate': 0.0,
                'efficiency_metric': 0.0,
                'energy_management': 0.0,
                'avg_lifespan': 0.0
            }
        
        # Calculate exploration rate (distance traveled per frame)
        exploration_rates = []
        for agent in agents:
            if agent.age > 0:
                exploration_rates.append(agent.distance_traveled / agent.age)
            else:
                exploration_rates.append(0.0)
        
        # Calculate efficiency based on population type
        efficiency_values = []
        for agent in agents:
            if population_name == 'GA1':
                # GA1: Food efficiency (food per distance traveled)
                if agent.distance_traveled > 0:
                    efficiency_values.append(agent.food_collected / agent.distance_traveled)
                else:
                    efficiency_values.append(0.0)
            else:
                # GA2: Attack efficiency (successful attacks per attempt)
                if agent.attacks_made > 0:
                    successful_attacks = getattr(agent, 'successful_attacks', 0)
                    efficiency_values.append(successful_attacks / agent.attacks_made)
                else:
                    efficiency_values.append(0.0)
        
        # Calculate energy management (energy retention rate)
        energy_management = []
        for agent in agents:
            initial_energy = config.AGENT_ENERGY
            retention_rate = agent.energy / initial_energy
            energy_management.append(retention_rate)
        
        behaviors = {
            'exploration_rate': np.mean(exploration_rates) if exploration_rates else 0.0,
            'efficiency_metric': np.mean(efficiency_values) if efficiency_values else 0.0,
            'energy_management': np.mean(energy_management) if energy_management else 0.0,
            'avg_lifespan': np.mean([a.age for a in agents]) if agents else 0.0
        }
        
        # Store behavioral history
        for behavior, value in behaviors.items():
            if behavior in self.behavior_patterns[population_name]:
                self.behavior_patterns[population_name][behavior].append(value)
        
        return behaviors
    
    def _calculate_population_stats(self, agents: List, pop_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a population.
        
        This function computes various metrics that help understand:
        - Population health and survival
        - Resource acquisition efficiency  
        - Fitness distribution and trends
        - Behavioral characteristics
        
        Args:
            agents: List of agents in the population
            pop_name: Name identifier for the population
            
        Returns:
            Dictionary containing calculated statistics
        """
        # Get correct population size for this population type
        if pop_name == "GA1":
            max_population = config.POPULATION_SIZE_GA1
        else:
            max_population = config.POPULATION_SIZE_GA2
            
        if not agents:
            # Return zero statistics for extinct populations
            return {
                'population': pop_name,
                'count': 0,
                'avg_fitness': 0,
                'max_fitness': 0,
                'min_fitness': 0,
                'fitness_std': 0,
                'avg_energy': 0,
                'avg_food_collected': 0,
                'avg_age': 0,
                'survival_rate': 0,
                'total_attacks': 0,
                'total_successful_attacks': 0
            }
        
        # Extract performance metrics
        fitnesses = [a.fitness for a in agents]
        energies = [a.energy for a in agents]
        food_counts = [a.food_collected for a in agents]
        ages = [a.age for a in agents]
        attacks = [a.attacks_made for a in agents]
        successful_attacks = [getattr(a, 'successful_attacks', 0) for a in agents]
        
        return {
            'population': pop_name,
            'count': len(agents),
            
            # Fitness metrics
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'fitness_std': np.std(fitnesses),
            
            # Resource and survival metrics
            'avg_energy': sum(energies) / len(energies),
            'avg_food_collected': sum(food_counts) / len(food_counts),
            'avg_age': sum(ages) / len(ages),
            'survival_rate': len(agents) / max_population,
            
            # NEW: Combat metrics
            'total_attacks': sum(attacks),
            'total_successful_attacks': sum(successful_attacks),
            'attack_success_rate': sum(successful_attacks) / max(1, sum(attacks))
        }
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """
        NEW: Get current real-time statistics for live display.
        
        Returns:
            Dictionary with current frame statistics
        """
        if not self.frame_data:
            return {}
        
        recent_frames = self.frame_data[-10:]  # Last 10 frames
        
        return {
            'current_frame': self.current_frame_data['frame'],
            'current_generation': self.current_frame_data['generation'],
            'recent_ga1_population': [f['ga1_alive'] for f in recent_frames],
            'recent_ga2_population': [f['ga2_alive'] for f in recent_frames],
            'recent_food_count': [f['food_count'] for f in recent_frames],
            'population_trend_ga1': self._calculate_trend([f['ga1_alive'] for f in recent_frames]),
            'population_trend_ga2': self._calculate_trend([f['ga2_alive'] for f in recent_frames])
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        
        start = np.mean(values[:len(values)//2])
        end = np.mean(values[len(values)//2:])
        diff = end - start
        
        if diff > 1:
            return "increasing"
        elif diff < -1:
            return "decreasing"
        else:
            return "stable"
    
    def generate_evolution_report(self) -> str:
        """
        Generate a comprehensive evolutionary analysis report.
        
        Returns:
            Detailed report analyzing evolutionary trends and patterns
        """
        if not self.generation_data:
            return "No data collected yet. Run simulation to gather data."
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE EVOLUTIONARY ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Overall simulation summary
        total_generations = len(self.generation_data)
        report.append(f"\nSIMULATION OVERVIEW:")
        report.append(f"Total Generations: {total_generations}")
        report.append(f"GA1 Extinctions: {len(self.extinction_events['GA1'])}")
        report.append(f"GA2 Extinctions: {len(self.extinction_events['GA2'])}")
        
        # Survival analysis
        if self.survival_history['GA1']:
            ga1_avg_survival = np.mean(self.survival_history['GA1'])
            ga1_max_survival = max(self.survival_history['GA1'])
        else:
            ga1_avg_survival = ga1_max_survival = 0
            
        if self.survival_history['GA2']:
            ga2_avg_survival = np.mean(self.survival_history['GA2'])
            ga2_max_survival = max(self.survival_history['GA2'])
        else:
            ga2_avg_survival = ga2_max_survival = 0
        
        report.append(f"\nSURVIVAL ANALYSIS:")
        report.append(f"GA1 Average Survivors: {ga1_avg_survival:.1f}/{config.POPULATION_SIZE_GA1} ({ga1_avg_survival/config.POPULATION_SIZE_GA1*100:.1f}%)")
        report.append(f"GA1 Best Generation: {ga1_max_survival}/{config.POPULATION_SIZE_GA1}")
        report.append(f"GA2 Average Survivors: {ga2_avg_survival:.1f}/{config.POPULATION_SIZE_GA2} ({ga2_avg_survival/config.POPULATION_SIZE_GA2*100:.1f}%)")
        report.append(f"GA2 Best Generation: {ga2_max_survival}/{config.POPULATION_SIZE_GA2}")
        
        # NEW: Enhanced trait evolution analysis
        report.append(f"\nTRAIT EVOLUTION & SELECTION PRESSURE:")
        for pop_name in ['GA1', 'GA2']:
            if pop_name in self.genome_analyzer.trait_fitness_correlations:
                report.append(f"\n{pop_name} TRAIT-FITNESS CORRELATIONS:")
                correlations = self.genome_analyzer.trait_fitness_correlations[pop_name]
                
                for trait, corr_history in correlations.items():
                    if corr_history:
                        avg_corr = np.mean(corr_history)
                        recent_corr = np.mean(corr_history[-3:]) if len(corr_history) >= 3 else avg_corr
                        
                        # Determine selection pressure
                        if abs(recent_corr) > 0.5:
                            pressure = "STRONG"
                        elif abs(recent_corr) > 0.2:
                            pressure = "MODERATE"
                        else:
                            pressure = "WEAK"
                        
                        direction = "POSITIVE" if recent_corr > 0 else "NEGATIVE"
                        report.append(f"  {trait:12}: {recent_corr:+.3f} ({pressure} {direction} selection)")
        
        # NEW: Energy flow analysis
        if self.generation_data:
            latest = self.generation_data[-1]
            energy_flow = latest.get('energy_flow', {})
            if energy_flow:
                report.append(f"\nENERGY FLOW ANALYSIS:")
                report.append(f"Total Food Consumed: {energy_flow.get('total_food_consumed', 0)}")
                report.append(f"Attack Success Rate: {energy_flow.get('attack_success_rate', 0):.1%}")
                report.append(f"Energy from Hunting: {energy_flow.get('energy_from_hunting', 0)}")
                report.append(f"Energy from Foraging: {energy_flow.get('energy_from_foraging', 0)}")
        
        # Continue with existing report sections...
        # [Previous report content remains the same]
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def create_evolution_graphs(self, save_path: str = "evolution_analysis.png"):
        """
        Create comprehensive visualization graphs showing evolution patterns.
        
        Args:
            save_path: Path to save the graph image
        """
        if not self.generation_data:
            print("No data to visualize. Run simulation first.")
            return
        
        # Create figure with 3x2 subplots for comprehensive analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Evolution Analysis', fontsize=16, fontweight='bold')
        
        # Graph 1: Population Over Time
        ax1 = axes[0, 0]
        generations = range(len(self.survival_history['GA1']))
        ga1_populations = self.survival_history['GA1']
        ga2_populations = self.survival_history['GA2']
        
        ax1.plot(generations, ga1_populations, 'b-', label='GA1 (Cooperative)', linewidth=3, marker='o', markersize=4)
        ax1.plot(generations, ga2_populations, 'r-', label='GA2 (Aggressive)', linewidth=3, marker='s', markersize=4)
        ax1.set_title('Population Survival Over Generations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Survivors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graph 2: Fitness Evolution
        ax2 = axes[0, 1]
        if self.performance_comparison['GA1']:
            ax2.plot(range(len(self.performance_comparison['GA1'])), self.performance_comparison['GA1'], 
                    'b-', label='GA1 Fitness', linewidth=3, marker='o', markersize=4)
        if self.performance_comparison['GA2']:
            ax2.plot(range(len(self.performance_comparison['GA2'])), self.performance_comparison['GA2'], 
                    'r-', label='GA2 Fitness', linewidth=3, marker='s', markersize=4)
        ax2.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Fitness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graphs 3-5: Trait Evolution (Speed, Sense, Size)
        trait_axes = [axes[1, 0], axes[1, 1], axes[2, 0]]
        trait_names = ['speed', 'sense', 'size']
        trait_titles = ['Speed Trait Evolution', 'Sense Trait Evolution', 'Size Trait Evolution']
        
        for i, (ax, trait, title) in enumerate(zip(trait_axes, trait_names, trait_titles)):
            if trait in self.genome_analyzer.trait_history['GA1']:
                ga1_trait = self.genome_analyzer.trait_history['GA1'][trait]
                ax.plot(range(len(ga1_trait)), ga1_trait, 'b-', label=f'GA1 {trait.title()}', 
                       linewidth=3, marker='o', markersize=4)
            if trait in self.genome_analyzer.trait_history['GA2']:
                ga2_trait = self.genome_analyzer.trait_history['GA2'][trait]
                ax.plot(range(len(ga2_trait)), ga2_trait, 'r-', label=f'GA2 {trait.title()}', 
                       linewidth=3, marker='s', markersize=4)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Generation')
            ax.set_ylabel(f'{trait.title()} (0.0 - 1.0)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # Graph 6: Predator-Prey Dynamics
        ax6 = axes[2, 1]
        if self.population_dynamics['predator_prey_ratio']:
            ax6.plot(range(len(self.population_dynamics['predator_prey_ratio'])), 
                    self.population_dynamics['predator_prey_ratio'], 
                    'g-', label='Predator:Prey Ratio', linewidth=3, marker='^', markersize=4)
            ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Ratio')
        ax6.set_title('Predator-Prey Dynamics', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Generation')
        ax6.set_ylabel('GA2:GA1 Ratio')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution graphs saved to: {save_path}")
        # plt.show()
    
    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary of the current simulation state.
        
        Returns:
            Formatted string containing the analysis summary
        """
        if not self.generation_data:
            return "No data collected yet. Run simulation to gather data."
        
        # Get latest generation data
        latest = self.generation_data[-1]
        
        # Calculate trend analysis if we have multiple generations
        trend_analysis = ""
        if len(self.generation_data) > 1:
            # Compare current generation to previous
            previous = self.generation_data[-2]
            
            ga1_trend = latest['ga1']['avg_fitness'] - previous['ga1']['avg_fitness'] 
            ga2_trend = latest['ga2']['avg_fitness'] - previous['ga2']['avg_fitness']
            
            trend_analysis = f"""
EVOLUTION TRENDS:
GA1 Fitness Change: {ga1_trend:+.2f}
GA2 Fitness Change: {ga2_trend:+.2f}
"""
        
        # Generate comprehensive report
        report = f"""
=== ARTIFICIAL WORLD SIMULATION REPORT ===
Generation: {latest['generation']}
Total Surviving Agents: {latest['total_agents']}

GA1 (COOPERATIVE STRATEGY):
  Population: {latest['ga1']['count']}/{config.POPULATION_SIZE_GA1} ({latest['ga1']['survival_rate']:.1%} survival)
  Avg Fitness: {latest['ga1']['avg_fitness']:.2f}
  Max Fitness: {latest['ga1']['max_fitness']:.2f}
  Avg Energy: {latest['ga1']['avg_energy']:.1f}
  Avg Food Collected: {latest['ga1']['avg_food_collected']:.2f}
  Avg Lifespan: {latest['ga1']['avg_age']:.1f} frames

GA2 (AGGRESSIVE STRATEGY):
  Population: {latest['ga2']['count']}/{config.POPULATION_SIZE_GA2} ({latest['ga2']['survival_rate']:.1%} survival)
  Avg Fitness: {latest['ga2']['avg_fitness']:.2f}
  Max Fitness: {latest['ga2']['max_fitness']:.2f}
  Avg Energy: {latest['ga2']['avg_energy']:.1f}
  Avg Food Collected: {latest['ga2']['avg_food_collected']:.2f}
  Avg Lifespan: {latest['ga2']['avg_age']:.1f} frames
  Total Attacks: {latest['ga2']['total_attacks']:.0f}
  Attack Success: {latest['ga2']['attack_success_rate']:.1%}

STRATEGY COMPARISON:
Best Strategy (Fitness): {"GA1" if latest['ga1']['avg_fitness'] > latest['ga2']['avg_fitness'] else "GA2"}
Best Strategy (Survival): {"GA1" if latest['ga1']['survival_rate'] > latest['ga2']['survival_rate'] else "GA2"}
{trend_analysis}
=== END REPORT ===
        """
        return report
    
    def _get_timestamp(self) -> int:
        """Get current timestamp for data recording."""
        return int(time.time())
    
    def save_data(self, filename: str = "simulation_data.json"):
        """
        Save all collected data to a JSON file for external analysis.
        
        Args:
            filename: Name of file to save data to
        """
        # Compile all data for export
        export_data = {
            'simulation_config': {
                'population_size_ga1': config.POPULATION_SIZE_GA1,
                'population_size_ga2': config.POPULATION_SIZE_GA2,
                'mutation_rate': config.MUTATION_RATE,
                'crossover_rate': config.CROSSOVER_RATE,
                'world_size': (config.WORLD_WIDTH, config.WORLD_HEIGHT),
                'food_count': config.FOOD_COUNT,
                'hazard_count': config.HAZARD_COUNT
            },
            'generation_data': self.generation_data,
            'performance_comparison': self.performance_comparison,
            'survival_history': self.survival_history,
            'extinction_events': self.extinction_events,
            'trait_evolution': dict(self.genome_analyzer.trait_history),
            'trait_fitness_correlations': dict(self.genome_analyzer.trait_fitness_correlations),
            'behavior_patterns': self.behavior_patterns,
            'population_dynamics': self.population_dynamics,
            'energy_flow': self.energy_flow,
            'summary_statistics': self._calculate_overall_summary()
        }
        
        # Write to file with proper formatting
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def _calculate_overall_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics across all generations."""
        if not self.generation_data:
            return {}
        
        return {
            'total_generations': len(self.generation_data),
            'ga1_avg_performance': np.mean(self.performance_comparison['GA1']) if self.performance_comparison['GA1'] else 0,
            'ga2_avg_performance': np.mean(self.performance_comparison['GA2']) if self.performance_comparison['GA2'] else 0,
            'ga1_best_performance': max(self.performance_comparison['GA1']) if self.performance_comparison['GA1'] else 0,
            'ga2_best_performance': max(self.performance_comparison['GA2']) if self.performance_comparison['GA2'] else 0,
            'ga1_extinction_count': len(self.extinction_events['GA1']),
            'ga2_extinction_count': len(self.extinction_events['GA2']),
            'ga1_avg_survival_rate': np.mean(self.survival_history['GA1']) / config.POPULATION_SIZE_GA1 if self.survival_history['GA1'] else 0,
            'ga2_avg_survival_rate': np.mean(self.survival_history['GA2']) / config.POPULATION_SIZE_GA2 if self.survival_history['GA2'] else 0
        }
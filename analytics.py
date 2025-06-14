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
                'genetic_diversity': 0.0
            }
        
        # Extract all genomes
        genomes = [agent.genome for agent in agents]
        
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
            'genetic_diversity': genetic_diversity
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
    
    The analytics system is essential for understanding which evolutionary
    strategies are most effective and how behaviors emerge over time.
    """
    
    def __init__(self):
        """Initialize analytics tracking systems."""
        # Raw data storage
        self.generation_data = []           # Detailed stats for each generation
        self.agent_data = []               # Individual agent performance records
        
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
        
        # Emergent behavior tracking
        self.behavior_patterns = {
            'GA1': {'exploration_rate': [], 'food_efficiency': [], 'energy_management': []},
            'GA2': {'exploration_rate': [], 'food_efficiency': [], 'energy_management': []}
        }
        
    def record_generation(self, generation: int, ga1, ga2):
        """
        Record comprehensive statistics for the current generation.
        
        This function captures a snapshot of both populations' performance,
        allowing for detailed analysis of evolutionary trends and strategy
        effectiveness over time.
        
        Args:
            generation: Current generation number
            ga1: Cooperative genetic algorithm population
            ga2: Aggressive genetic algorithm population
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
        
        # Create comprehensive generation record
        generation_record = {
            'generation': generation,
            'ga1': {**ga1_stats, 'genetics': ga1_genetics, 'behaviors': ga1_behaviors},
            'ga2': {**ga2_stats, 'genetics': ga2_genetics, 'behaviors': ga2_behaviors},
            'total_agents': len(ga1_agents) + len(ga2_agents),
            'timestamp': self._get_timestamp()
        }
        
        # Store data for analysis
        self.generation_data.append(generation_record)
        self.performance_comparison['GA1'].append(ga1_stats['avg_fitness'])
        self.performance_comparison['GA2'].append(ga2_stats['avg_fitness'])
    
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
                'food_efficiency': 0.0,
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
        
        # Calculate food efficiency (food per distance traveled)
        food_efficiency = []
        for agent in agents:
            if agent.distance_traveled > 0:
                food_efficiency.append(agent.food_collected / agent.distance_traveled)
            else:
                food_efficiency.append(0.0)
        
        # Calculate energy management (energy retention rate)
        energy_management = []
        for agent in agents:
            initial_energy = config.AGENT_ENERGY
            retention_rate = agent.energy / initial_energy
            energy_management.append(retention_rate)
        
        behaviors = {
            'exploration_rate': np.mean(exploration_rates) if exploration_rates else 0.0,
            'food_efficiency': np.mean(food_efficiency) if food_efficiency else 0.0,
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
                'total_attacks': 0
            }
        
        # Extract performance metrics
        fitnesses = [a.fitness for a in agents]
        energies = [a.energy for a in agents]
        food_counts = [a.food_collected for a in agents]
        ages = [a.age for a in agents]
        attacks = [a.attacks_made for a in agents]
        
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
            
            # Behavioral metrics
            'total_attacks': sum(attacks)
        }
    
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
        
        # Fitness evolution
        if self.performance_comparison['GA1']:
            ga1_fitness_trend = np.mean(self.performance_comparison['GA1'][-5:]) - np.mean(self.performance_comparison['GA1'][:5])
        else:
            ga1_fitness_trend = 0
            
        if self.performance_comparison['GA2']:
            ga2_fitness_trend = np.mean(self.performance_comparison['GA2'][-5:]) - np.mean(self.performance_comparison['GA2'][:5])
        else:
            ga2_fitness_trend = 0
        
        report.append(f"\nFITNESS EVOLUTION:")
        report.append(f"GA1 Fitness Trend: {ga1_fitness_trend:+.2f} (recent vs early generations)")
        report.append(f"GA2 Fitness Trend: {ga2_fitness_trend:+.2f} (recent vs early generations)")
        
        # Genetic diversity analysis
        if self.genome_analyzer.diversity_history['GA1']:
            ga1_diversity = np.mean(self.genome_analyzer.diversity_history['GA1'])
        else:
            ga1_diversity = 0
            
        if self.genome_analyzer.diversity_history['GA2']:
            ga2_diversity = np.mean(self.genome_analyzer.diversity_history['GA2'])
        else:
            ga2_diversity = 0
        
        report.append(f"\nGENETIC DIVERSITY:")
        report.append(f"GA1 Average Diversity: {ga1_diversity:.3f}")
        report.append(f"GA2 Average Diversity: {ga2_diversity:.3f}")
        
        # EMERGENT TRAIT ANALYSIS - New enhanced section
        report.append(f"\nEMERGENT TRAIT EVOLUTION:")
        for pop_name in ['GA1', 'GA2']:
            if pop_name in self.genome_analyzer.trait_history:
                report.append(f"\n{pop_name} TRAIT EVOLUTION:")
                traits = self.genome_analyzer.trait_history[pop_name]
                
                for trait_name, trait_values in traits.items():
                    if len(trait_values) >= 3:  # Need at least 3 generations to analyze
                        # Calculate trend and variance
                        start_avg = np.mean(trait_values[:2])
                        end_avg = np.mean(trait_values[-2:])
                        trend = end_avg - start_avg
                        variance = np.var(trait_values)
                        
                        # Determine if trait is evolving (significant change + variance)
                        is_evolving = abs(trend) > 0.1 and variance > 0.01
                        
                        status = "🔄 EVOLVING" if is_evolving else "📊 STABLE"
                        direction = "↗️ INCREASING" if trend > 0.05 else "↘️ DECREASING" if trend < -0.05 else "➡️ STEADY"
                        
                        report.append(f"  {trait_name:20} | {status:12} | {direction:12} | Change: {trend:+.3f}")
        
        # Latest generation detailed analysis
        if self.generation_data:
            latest = self.generation_data[-1]
            report.append(f"\nLATEST GENERATION ANALYSIS:")
            report.append(f"Generation {latest['generation']}:")
            
            # GA1 genetics
            if latest['ga1']['genetics']['avg_traits']:
                report.append(f"\nGA1 Current Genome Profile:")
                traits = latest['ga1']['genetics']['avg_traits']
                for trait, value in traits.items():
                    report.append(f"  {trait:20}: {value:.3f}")
            
            # GA2 genetics
            if latest['ga2']['genetics']['avg_traits']:
                report.append(f"\nGA2 Current Genome Profile:")
                traits = latest['ga2']['genetics']['avg_traits']
                for trait, value in traits.items():
                    report.append(f"  {trait:20}: {value:.3f}")
        
        # Combat analysis
        if self.generation_data:
            total_attacks_ga1 = sum(gen['ga1'].get('total_attacks', 0) for gen in self.generation_data)
            total_attacks_ga2 = sum(gen['ga2'].get('total_attacks', 0) for gen in self.generation_data)
            report.append(f"\nCOMBAT ANALYSIS:")
            report.append(f"GA1 Total Attacks: {total_attacks_ga1}")
            report.append(f"GA2 Total Attacks: {total_attacks_ga2}")
            if total_attacks_ga2 > 0:
                report.append("✅ Aggressive behavior is active!")
            else:
                report.append("⚠️  No aggressive behavior detected")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def create_evolution_graphs(self, save_path: str = "evolution_analysis.png"):
        """
        Create focused visualization graphs showing population and trait evolution.
        
        Args:
            save_path: Path to save the graph image
        """
        if not self.generation_data:
            print("No data to visualize. Run simulation first.")
            return
        
        # Create figure with 2x2 subplots - focused on key metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Population and Trait Evolution Analysis', fontsize=16, fontweight='bold')
        
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
        ax1.set_ylim(0, max(max(ga1_populations) if ga1_populations else [0], 
                            max(ga2_populations) if ga2_populations else [0]) + 5)
        
        # Graph 2: Speed Trait Evolution
        ax2 = axes[0, 1]
        if 'speed' in self.genome_analyzer.trait_history['GA1']:
            ga1_speed = self.genome_analyzer.trait_history['GA1']['speed']
            ax2.plot(range(len(ga1_speed)), ga1_speed, 'b-', label='GA1 Speed', linewidth=3, marker='o', markersize=4)
        if 'speed' in self.genome_analyzer.trait_history['GA2']:
            ga2_speed = self.genome_analyzer.trait_history['GA2']['speed']
            ax2.plot(range(len(ga2_speed)), ga2_speed, 'r-', label='GA2 Speed', linewidth=3, marker='s', markersize=4)
        ax2.set_title('Speed Trait Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Speed (0.0 - 1.0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Graph 3: Sense Trait Evolution
        ax3 = axes[1, 0]
        if 'sense' in self.genome_analyzer.trait_history['GA1']:
            ga1_sense = self.genome_analyzer.trait_history['GA1']['sense']
            ax3.plot(range(len(ga1_sense)), ga1_sense, 'b-', label='GA1 Sense', linewidth=3, marker='o', markersize=4)
        if 'sense' in self.genome_analyzer.trait_history['GA2']:
            ga2_sense = self.genome_analyzer.trait_history['GA2']['sense']
            ax3.plot(range(len(ga2_sense)), ga2_sense, 'r-', label='GA2 Sense', linewidth=3, marker='s', markersize=4)
        ax3.set_title('Sense Trait Evolution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Sense (0.0 - 1.0)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Graph 4: Size Trait Evolution
        ax4 = axes[1, 1]
        if 'size' in self.genome_analyzer.trait_history['GA1']:
            ga1_size = self.genome_analyzer.trait_history['GA1']['size']
            ax4.plot(range(len(ga1_size)), ga1_size, 'b-', label='GA1 Size', linewidth=3, marker='o', markersize=4)
        if 'size' in self.genome_analyzer.trait_history['GA2']:
            ga2_size = self.genome_analyzer.trait_history['GA2']['size']
            ax4.plot(range(len(ga2_size)), ga2_size, 'r-', label='GA2 Size', linewidth=3, marker='s', markersize=4)
        ax4.set_title('Size Trait Evolution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Size (0.0 - 1.0)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution graphs saved to: {save_path}")
        plt.show()
    
    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary of the current simulation state.
        
        This report provides key insights for researchers analyzing the
        simulation results, highlighting important trends and comparisons
        between the evolutionary strategies.
        
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
        
        This allows researchers to:
        - Analyze data in external tools (Python, R, Excel)
        - Create custom visualizations
        - Perform statistical tests
        - Compare different simulation runs
        
        Args:
            filename: Name of file to save data to
        """
        # Compile all data for export
        export_data = {
            'simulation_config': {
                'population_size': config.POPULATION_SIZE_GA1,
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
            'behavior_patterns': self.behavior_patterns,
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
            'ga1_avg_performance': np.mean(self.performance_comparison['GA1']),
            'ga2_avg_performance': np.mean(self.performance_comparison['GA2']),
            'ga1_best_performance': max(self.performance_comparison['GA1']) if self.performance_comparison['GA1'] else 0,
            'ga2_best_performance': max(self.performance_comparison['GA2']) if self.performance_comparison['GA2'] else 0,
            'ga1_extinction_count': len(self.extinction_events['GA1']),
            'ga2_extinction_count': len(self.extinction_events['GA2']),
            'ga1_avg_survival_rate': np.mean(self.survival_history['GA1']) / config.POPULATION_SIZE_GA1 if self.survival_history['GA1'] else 0,
            'ga2_avg_survival_rate': np.mean(self.survival_history['GA2']) / config.POPULATION_SIZE_GA2 if self.survival_history['GA2'] else 0
        } 
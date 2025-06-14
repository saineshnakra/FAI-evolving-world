"""
Analytics module for the Artificial World Simulation.

Contains comprehensive data collection and analysis systems for tracking
evolutionary progress and comparing genetic algorithm strategies.
"""

import numpy as np
import json
import time
from typing import List, Dict, Any

from config import config


class Analytics:
    """
    Comprehensive data collection and analysis system for the simulation.
    
    This class tracks:
    - Generation-by-generation performance statistics  
    - Population dynamics and survival rates
    - Behavioral evolution trends
    - Comparative performance between GA strategies
    
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
        
        # Calculate detailed statistics for each population
        ga1_stats = self._calculate_population_stats(ga1_agents, "GA1")
        ga2_stats = self._calculate_population_stats(ga2_agents, "GA2")
        
        # Create comprehensive generation record
        generation_record = {
            'generation': generation,
            'ga1': ga1_stats,
            'ga2': ga2_stats,
            'total_agents': len(ga1_agents) + len(ga2_agents),
            'timestamp': self._get_timestamp()  # For temporal analysis
        }
        
        # Store data for analysis
        self.generation_data.append(generation_record)
        self.performance_comparison['GA1'].append(ga1_stats['avg_fitness'])
        self.performance_comparison['GA2'].append(ga2_stats['avg_fitness'])
    
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
            'survival_rate': len(agents) / config.POPULATION_SIZE,
            
            # Behavioral metrics
            'total_attacks': sum(attacks)
        }
    
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
  Population: {latest['ga1']['count']}/{config.POPULATION_SIZE} ({latest['ga1']['survival_rate']:.1%} survival)
  Avg Fitness: {latest['ga1']['avg_fitness']:.2f}
  Max Fitness: {latest['ga1']['max_fitness']:.2f}
  Avg Energy: {latest['ga1']['avg_energy']:.1f}
  Avg Food Collected: {latest['ga1']['avg_food_collected']:.2f}
  Avg Lifespan: {latest['ga1']['avg_age']:.1f} frames

GA2 (AGGRESSIVE STRATEGY):
  Population: {latest['ga2']['count']}/{config.POPULATION_SIZE} ({latest['ga2']['survival_rate']:.1%} survival)
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
                'population_size': config.POPULATION_SIZE,
                'mutation_rate': config.MUTATION_RATE,
                'crossover_rate': config.CROSSOVER_RATE,
                'world_size': (config.WORLD_WIDTH, config.WORLD_HEIGHT),
                'food_count': config.FOOD_COUNT,
                'hazard_count': config.HAZARD_COUNT
            },
            'generation_data': self.generation_data,
            'performance_comparison': self.performance_comparison,
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
            'ga2_best_performance': max(self.performance_comparison['GA2']) if self.performance_comparison['GA2'] else 0
        } 
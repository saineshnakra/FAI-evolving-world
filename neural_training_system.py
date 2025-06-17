"""
Neural Training System for Automated Hyperparameter Optimization
================================================================

This system integrates neural networks with the genetic algorithm simulation
to automatically discover optimal hyperparameters for three objectives:
1. GA1 Dominance - Make cooperative agents dominate
2. GA2 Dominance - Make aggressive agents dominate  
3. Coexistence - Achieve stable coexistence between populations
"""

import time
import numpy as np
from typing import Dict
from dataclasses import asdict
import json
from datetime import datetime

from neural_hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSet
from config import config


class AutomatedTrainingSystem:
    """Automated system for training neural networks to optimize simulation hyperparameters."""
    
    def __init__(self, training_episodes_per_objective: int = 30):
        """Initialize the automated training system."""
        print("Initializing Automated Neural Training System...")
        
        self.optimizer = HyperparameterOptimizer()
        self.training_episodes_per_objective = training_episodes_per_objective
        self.max_simulation_time = 60  # Maximum seconds per simulation
        
        print(f"✓ Training episodes per objective: {training_episodes_per_objective}")
    
    def run_full_training_cycle(self):
        """Run a complete training cycle for all three objectives."""
        print("\nSTARTING AUTOMATED NEURAL TRAINING CYCLE")
        print("=" * 60)
        
        objectives = ['ga1_dominance', 'ga2_dominance', 'coexistence']
        
        for objective in objectives:
            print(f"\n🎯 TRAINING OBJECTIVE: {objective.upper()}")
            self._train_objective(objective)
        
        # Save final results
        self.optimizer.save_models("final_trained_models")
        self._save_training_summary()
        
        print("\n✓ All models trained and saved successfully!")
    
    def _train_objective(self, objective: str):
        """Train neural network for a specific objective."""
        for episode in range(self.training_episodes_per_objective):
            print(f"Episode {episode + 1}/{self.training_episodes_per_objective}")
            
            # Generate hyperparameters using neural network
            hyperparams = self.optimizer.generate_hyperparameters(objective)
            
            # Run simulation with these hyperparameters
            results = self._run_simulation_with_hyperparameters(hyperparams)
            
            # Train neural network on results
            self.optimizer.train_on_simulation_result(objective, hyperparams, results)
            
            print(f"Results - GA1: {results['ga1_survival_rate']:.3f}, "
                  f"GA2: {results['ga2_survival_rate']:.3f}")
    
    def _run_simulation_with_hyperparameters(self, hyperparams: HyperparameterSet) -> Dict[str, float]:
        """Run a simulation with specific hyperparameters."""
        # Apply hyperparameters
        original_config = self._backup_config()
        self._apply_hyperparameters(hyperparams)
        
        try:
            # Run simplified simulation
            results = self._run_headless_simulation()
        finally:
            # Restore original configuration
            self._restore_config(original_config)
        
        return results
    
    def _run_headless_simulation(self) -> Dict[str, float]:
        """Run a simplified simulation without visualization."""
        from world import World
        from genetics import GeneticAlgorithm
        from config import AgentType
        
        # Create simulation components
        world = World()
        ga1 = GeneticAlgorithm(AgentType.COOPERATIVE)
        ga2 = GeneticAlgorithm(AgentType.AGGRESSIVE)
        
        # Initialize populations
        spawn_positions = world.get_spawn_positions(
            config.POPULATION_SIZE_GA1 + config.POPULATION_SIZE_GA2
        )
        ga1.initialize_population(spawn_positions)
        ga2.initialize_population(spawn_positions)
        
        # Run simulation loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Check termination conditions
            if time.time() - start_time > self.max_simulation_time:
                break
            
            max_gen = max(ga1.generation, ga2.generation)
            if max_gen >= 50:  # Cap at 50 generations for training
                break
            
            # Update simulation
            all_agents = ga1.population + ga2.population
            world.update(all_agents)
            
            # Update agents
            for agent in all_agents:
                if agent.alive:
                    world_state = world.get_world_state_for_agent(agent, all_agents)
                    dx, dy = agent.update(world_state)
                    world.apply_agent_movement(agent, dx, dy)
            
            # Check for generation evolution
            if world.is_generation_complete(all_agents):
                spawn_positions = world.get_spawn_positions(
                    config.POPULATION_SIZE_GA1 + config.POPULATION_SIZE_GA2
                )
                ga1.evolve_generation(spawn_positions)
                ga2.evolve_generation(spawn_positions)
                world.reset_for_new_generation()
            
            frame_count += 1
            
            # Safety check
            if frame_count > 30000:
                break
        
        # Calculate results
        return self._calculate_simulation_metrics(ga1, ga2, frame_count)
    
    def _calculate_simulation_metrics(self, ga1, ga2, frame_count) -> Dict[str, float]:
        """Calculate key metrics from simulation."""
        ga1_alive = len([a for a in ga1.population if a.alive])
        ga2_alive = len([a for a in ga2.population if a.alive])
        
        ga1_survival_rate = ga1_alive / max(len(ga1.population), 1)
        ga2_survival_rate = ga2_alive / max(len(ga2.population), 1)
        
        # Calculate diversity
        all_alive = [a for a in ga1.population + ga2.population if a.alive]
        diversity = 0.0
        if len(all_alive) > 1:
            traits = [(a.genome.speed, a.genome.sense, a.genome.size) for a in all_alive]
            diversity = np.std(traits) if len(traits) > 1 else 0.0
        
        return {
            'ga1_survival_rate': ga1_survival_rate,
            'ga2_survival_rate': ga2_survival_rate,
            'generations_completed': max(ga1.generation, ga2.generation),
            'genetic_diversity': min(diversity, 1.0),
            'total_frames': frame_count
        }
    
    def _backup_config(self) -> Dict:
        """Backup current configuration."""
        return {
            'POPULATION_SIZE_GA1': config.POPULATION_SIZE_GA1,
            'POPULATION_SIZE_GA2': config.POPULATION_SIZE_GA2,
            'FOOD_COUNT': config.FOOD_COUNT,
            'HAZARD_COUNT': config.HAZARD_COUNT,
            'FOOD_SPAWN_RATE': config.FOOD_SPAWN_RATE,
            'AGENT_ENERGY': config.AGENT_ENERGY,
            'MOVEMENT_COST': config.MOVEMENT_COST,
            'EATING_REWARD': config.EATING_REWARD,
            'MUTATION_RATE': config.MUTATION_RATE,
            'CROSSOVER_RATE': config.CROSSOVER_RATE,
            'TOURNAMENT_SIZE': config.TOURNAMENT_SIZE
        }
    
    def _apply_hyperparameters(self, hyperparams: HyperparameterSet):
        """Apply hyperparameters to global config."""
        config.POPULATION_SIZE_GA1 = hyperparams.population_size_ga1
        config.POPULATION_SIZE_GA2 = hyperparams.population_size_ga2
        config.FOOD_COUNT = hyperparams.food_count
        config.HAZARD_COUNT = hyperparams.hazard_count
        config.FOOD_SPAWN_RATE = hyperparams.food_spawn_rate
        config.AGENT_ENERGY = hyperparams.agent_energy
        config.MOVEMENT_COST = hyperparams.movement_cost
        config.EATING_REWARD = hyperparams.eating_reward
        config.MUTATION_RATE = hyperparams.mutation_rate
        config.CROSSOVER_RATE = hyperparams.crossover_rate
        config.TOURNAMENT_SIZE = hyperparams.tournament_size
    
    def _restore_config(self, backup: Dict):
        """Restore configuration from backup."""
        for key, value in backup.items():
            setattr(config, key, value)
    
    def _save_training_summary(self):
        """Save comprehensive training summary."""
        summary = {
            'training_config': {
                'episodes_per_objective': self.training_episodes_per_objective,
                'max_simulation_time': self.max_simulation_time
            },
            'neural_network_summary': self.optimizer.get_training_summary(),
            'best_parameters': {
                objective: asdict(params) if params else None 
                for objective, params in self.optimizer.best_params.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('neural_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✓ Training summary saved")


def main():
    """Main function to run the automated neural training system."""
    print("Neural Hyperparameter Training System")
    print("=====================================")
    
    # Create training system
    training_system = AutomatedTrainingSystem(training_episodes_per_objective=20)
    
    # Run full training cycle
    training_system.run_full_training_cycle()
    
    print("\n🎉 Neural training completed successfully!")


if __name__ == "__main__":
    main() 
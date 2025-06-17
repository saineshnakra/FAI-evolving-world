#!/usr/bin/env python3
"""
Neural Hyperparameter Optimization Demo
=======================================

This script demonstrates how to use the neural network system to optimize
hyperparameters for three different objectives in the genetic algorithm simulation.

Usage:
    python demo_neural_optimization.py

Features:
- Trains three neural networks for different objectives
- Demonstrates the optimized hyperparameters
- Shows performance comparisons
- Saves visualizations and results
"""

import os
import sys
import time
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_training_system import AutomatedTrainingSystem
from neural_hyperparameter_optimizer import create_training_visualization
from config import config


def print_banner():
    """Print an attractive banner for the demo."""
    print("\n" + "=" * 70)
    print("🧠 NEURAL HYPERPARAMETER OPTIMIZATION DEMO")
    print("=" * 70)
    print("This demo shows how neural networks can optimize genetic algorithm")
    print("simulation parameters for three different objectives:")
    print()
    print("1. 🔵 GA1 DOMINANCE - Make cooperative agents dominate")
    print("2. 🟠 GA2 DOMINANCE - Make aggressive agents dominate")
    print("3. 🟢 COEXISTENCE   - Achieve balanced survival")
    print()
    print("Board size increased to 1200x900 pixels")
    print("Population scaled: GA1=150, GA2=30 agents")
    print("Environment enhanced: 180 food, 12 hazards")
    print("=" * 70)


def print_current_config():
    """Print the current simulation configuration."""
    print("\n📋 CURRENT SIMULATION CONFIGURATION:")
    print("-" * 40)
    print(f"Board Size:      {config.WORLD_WIDTH}x{config.WORLD_HEIGHT} pixels")
    print(f"GA1 Population:  {config.POPULATION_SIZE_GA1} agents")
    print(f"GA2 Population:  {config.POPULATION_SIZE_GA2} agents")
    print(f"Food Items:      {config.FOOD_COUNT}")
    print(f"Hazard Areas:    {config.HAZARD_COUNT}")
    print(f"Food Spawn Rate: {config.FOOD_SPAWN_RATE}")
    print(f"Agent Energy:    {config.AGENT_ENERGY}")
    print(f"Mutation Rate:   {config.MUTATION_RATE}")
    print(f"Crossover Rate:  {config.CROSSOVER_RATE}")


def demonstrate_quick_training():
    """Run a quick demonstration of the neural training system."""
    print("\n🚀 STARTING NEURAL TRAINING DEMONSTRATION")
    print("-" * 50)
    print("Training 3 neural networks with 10 episodes each...")
    print("(This is a quick demo - full training uses 30+ episodes)")
    
    # Create training system with fewer episodes for demo
    training_system = AutomatedTrainingSystem(training_episodes_per_objective=10)
    
    # Run the training
    start_time = time.time()
    training_system.run_full_training_cycle()
    end_time = time.time()
    
    training_duration = end_time - start_time
    print(f"\n✅ Training completed in {training_duration:.1f} seconds")
    
    return training_system


def demonstrate_best_parameters(training_system):
    """Show the best parameters discovered for each objective."""
    print("\n🏆 BEST HYPERPARAMETERS DISCOVERED:")
    print("=" * 50)
    
    objectives = [
        ('ga1_dominance', '🔵 GA1 DOMINANCE'),
        ('ga2_dominance', '🟠 GA2 DOMINANCE'),
        ('coexistence', '🟢 COEXISTENCE')
    ]
    
    for obj_key, obj_name in objectives:
        print(f"\n{obj_name}:")
        print("-" * 30)
        
        best_params = training_system.optimizer.best_params.get(obj_key)
        if best_params:
            print(f"  GA1 Population:  {best_params.population_size_ga1}")
            print(f"  GA2 Population:  {best_params.population_size_ga2}")
            print(f"  Food Count:      {best_params.food_count}")
            print(f"  Hazard Count:    {best_params.hazard_count}")
            print(f"  Food Spawn Rate: {best_params.food_spawn_rate:.3f}")
            print(f"  Agent Energy:    {best_params.agent_energy}")
            print(f"  Mutation Rate:   {best_params.mutation_rate:.3f}")
            print(f"  Crossover Rate:  {best_params.crossover_rate:.3f}")
            
            # Show training performance
            history = training_system.optimizer.training_history[obj_key]
            if history:
                best_reward = max(h['reward'] for h in history)
                print(f"  Best Reward:     {best_reward:.4f}")
        else:
            print("  No optimized parameters found")


def create_summary_report(training_system):
    """Create a comprehensive summary report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"neural_optimization_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("Neural Hyperparameter Optimization Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SYSTEM OVERVIEW:\n")
        f.write("- Enhanced board size: 1200x900 pixels (50% larger)\n")
        f.write("- Scaled populations: GA1=150, GA2=30 agents\n")
        f.write("- Three neural networks trained for different objectives\n")
        f.write("- Automated hyperparameter discovery\n\n")
        
        f.write("TRAINING SUMMARY:\n")
        f.write(training_system.optimizer.get_training_summary())
        
        f.write("\nFILES GENERATED:\n")
        f.write("- final_trained_models_*.pth (Neural network weights)\n")
        f.write("- neural_training_summary.json (Detailed results)\n")
        f.write("- training_progress_*.png (Visualization graphs)\n")
        f.write(f"- {report_filename} (This report)\n")
    
    print(f"\n📄 Comprehensive report saved: {report_filename}")


def main():
    """Main demonstration function."""
    print_banner()
    print_current_config()
    
    # Ask user if they want to proceed
    print("\n❓ This demo will train 3 neural networks to optimize simulation parameters.")
    print("   Training will take approximately 2-5 minutes.")
    
    proceed = input("\nProceed with neural training demo? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Demo cancelled.")
        return
    
    try:
        # Run the training demonstration
        training_system = demonstrate_quick_training()
        
        # Show results
        demonstrate_best_parameters(training_system)
        
        # Create visualization
        print("\n📊 Creating training progress visualization...")
        create_training_visualization(training_system.optimizer, "demo_training_progress.png")
        
        # Generate comprehensive report
        create_summary_report(training_system)
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 70)
        print("✅ Three neural networks successfully trained")
        print("✅ Optimal hyperparameters discovered for each objective")
        print("✅ Visualizations and reports generated")
        print("✅ Models saved for future use")
        print()
        print("🔍 Check the generated files:")
        print("   - demo_training_progress.png (Training visualization)")
        print("   - neural_training_summary.json (Detailed results)")
        print("   - neural_optimization_report_*.txt (Summary report)")
        print("   - final_trained_models_*.pth (Trained neural networks)")
        print()
        print("🚀 You can now use these optimized parameters in your simulations!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies.")
        print("Please install: pip install torch matplotlib scikit-learn")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
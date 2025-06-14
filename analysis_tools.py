"""
Analysis Tools for Evolutionary Simulation
==========================================

This module provides comprehensive analysis tools for examining
simulation results, generating reports, and creating visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import pandas as pd

def load_simulation_data(filename: str = "simulation_data.json") -> Dict[str, Any]:
    """
    Load simulation data from JSON file.
    
    Args:
        filename: Path to the simulation data file
        
    Returns:
        Dictionary containing all simulation data
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run simulation first.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}")
        return {}

def create_comprehensive_analysis(data: Dict[str, Any], save_prefix: str = "analysis"):
    """
    Create comprehensive analysis visualizations from simulation data.
    
    Args:
        data: Simulation data dictionary
        save_prefix: Prefix for saved image files
    """
    if not data:
        print("No data to analyze")
        return
    
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create the main dashboard
    create_evolution_dashboard(data, f"{save_prefix}_dashboard.png")
    
    # Create survival analysis
    create_survival_analysis(data, f"{save_prefix}_survival.png")
    
    # Create genome evolution analysis
    if 'trait_evolution' in data:
        create_genome_evolution_plots(data, f"{save_prefix}_genetics.png")
    
    # Create behavior analysis
    if 'behavior_patterns' in data:
        create_behavior_analysis(data, f"{save_prefix}_behaviors.png")
    
    print(f"Analysis complete! Generated visualizations with prefix '{save_prefix}'")

def create_evolution_dashboard(data: Dict[str, Any], save_path: str):
    """Create main evolution dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Evolutionary Simulation Dashboard', fontsize=20, fontweight='bold')
    
    # Extract data
    generations = range(len(data['performance_comparison']['GA1']))
    ga1_fitness = data['performance_comparison']['GA1']
    ga2_fitness = data['performance_comparison']['GA2']
    
    # 1. Fitness Evolution
    ax1 = axes[0, 0]
    ax1.plot(generations, ga1_fitness, 'b-', label='GA1 (Cooperative)', linewidth=3, marker='o', markersize=4)
    ax1.plot(generations, ga2_fitness, 'r-', label='GA2 (Aggressive)', linewidth=3, marker='s', markersize=4)
    ax1.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Average Fitness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Survival Rates
    if 'survival_history' in data:
        ax2 = axes[0, 1]
        population_size = data['simulation_config']['population_size']
        ga1_survival = [s/population_size*100 for s in data['survival_history']['GA1']]
        ga2_survival = [s/population_size*100 for s in data['survival_history']['GA2']]
        
        ax2.plot(range(len(ga1_survival)), ga1_survival, 'b-', label='GA1', linewidth=3, marker='o', markersize=4)
        ax2.plot(range(len(ga2_survival)), ga2_survival, 'r-', label='GA2', linewidth=3, marker='s', markersize=4)
        ax2.set_title('Survival Rates Over Generations', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Survival Rate (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Extinction Events
    if 'extinction_events' in data:
        ax3 = axes[0, 2]
        ga1_extinctions = data['extinction_events']['GA1']
        ga2_extinctions = data['extinction_events']['GA2']
        
        # Create histogram of extinction events
        all_gens = list(range(len(generations)))
        ga1_extinct = [1 if g in ga1_extinctions else 0 for g in all_gens]
        ga2_extinct = [1 if g in ga2_extinctions else 0 for g in all_gens]
        
        ax3.bar(all_gens, ga1_extinct, alpha=0.7, label='GA1 Extinctions', color='blue')
        ax3.bar(all_gens, ga2_extinct, alpha=0.7, label='GA2 Extinctions', color='red', bottom=ga1_extinct)
        ax3.set_title('Extinction Events by Generation', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Extinction (1=Yes, 0=No)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Performance Distribution
    ax4 = axes[1, 0]
    valid_ga1 = [f for f in ga1_fitness if f > 0]
    valid_ga2 = [f for f in ga2_fitness if f > 0]
    
    if valid_ga1:
        ax4.hist(valid_ga1, alpha=0.7, label='GA1', bins=15, color='blue')
    if valid_ga2:
        ax4.hist(valid_ga2, alpha=0.7, label='GA2', bins=15, color='red')
    ax4.set_title('Fitness Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Fitness Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Generation Summary
    ax5 = axes[1, 1]
    generation_data = data['generation_data']
    if generation_data:
        total_agents_per_gen = [gen['total_agents'] for gen in generation_data]
        ax5.plot(range(len(total_agents_per_gen)), total_agents_per_gen, 'g-', linewidth=3, marker='o', markersize=4)
        ax5.set_title('Total Surviving Agents per Generation', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Total Survivors')
        ax5.grid(True, alpha=0.3)
    
    # 6. Strategy Comparison
    ax6 = axes[1, 2]
    if valid_ga1 and valid_ga2:
        strategies = ['GA1\n(Cooperative)', 'GA2\n(Aggressive)']
        avg_fitness = [np.mean(valid_ga1), np.mean(valid_ga2)]
        colors = ['blue', 'red']
        
        bars = ax6.bar(strategies, avg_fitness, color=colors, alpha=0.7)
        ax6.set_title('Average Strategy Performance', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Average Fitness')
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_fitness):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to: {save_path}")
    plt.show()

def create_survival_analysis(data: Dict[str, Any], save_path: str):
    """Create detailed survival analysis."""
    if 'survival_history' not in data:
        print("No survival data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Survival Analysis Deep Dive', fontsize=16, fontweight='bold')
    
    population_size = data['simulation_config']['population_size']
    ga1_survival = data['survival_history']['GA1']
    ga2_survival = data['survival_history']['GA2']
    
    # 1. Survival trends with moving average
    ax1 = axes[0, 0]
    window = min(5, len(ga1_survival) // 3) if len(ga1_survival) > 0 else 1
    
    if len(ga1_survival) >= window:
        ga1_ma = pd.Series(ga1_survival).rolling(window=window).mean()
        ax1.plot(range(len(ga1_survival)), ga1_survival, 'b-', alpha=0.5, label='GA1 Raw')
        ax1.plot(range(len(ga1_ma)), ga1_ma, 'b-', linewidth=3, label=f'GA1 MA({window})')
    
    if len(ga2_survival) >= window:
        ga2_ma = pd.Series(ga2_survival).rolling(window=window).mean()
        ax1.plot(range(len(ga2_survival)), ga2_survival, 'r-', alpha=0.5, label='GA2 Raw')
        ax1.plot(range(len(ga2_ma)), ga2_ma, 'r-', linewidth=3, label=f'GA2 MA({window})')
    
    ax1.set_title('Survival Trends with Moving Average')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Survivors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Survival rate statistics
    ax2 = axes[0, 1]
    if ga1_survival and ga2_survival:
        stats_data = [
            [np.mean(ga1_survival), np.mean(ga2_survival)],
            [np.max(ga1_survival), np.max(ga2_survival)],
            [np.min(ga1_survival), np.min(ga2_survival)],
            [np.std(ga1_survival), np.std(ga2_survival)]
        ]
        
        x = np.arange(len(['Mean', 'Max', 'Min', 'Std Dev']))
        width = 0.35
        
        ax2.bar(x - width/2, [row[0] for row in stats_data], width, label='GA1', color='blue', alpha=0.7)
        ax2.bar(x + width/2, [row[1] for row in stats_data], width, label='GA2', color='red', alpha=0.7)
        
        ax2.set_title('Survival Statistics Comparison')
        ax2.set_ylabel('Number of Survivors')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Mean', 'Max', 'Min', 'Std Dev'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Extinction probability over time
    ax3 = axes[1, 0]
    if 'extinction_events' in data:
        ga1_extinctions = data['extinction_events']['GA1']
        ga2_extinctions = data['extinction_events']['GA2']
        
        total_gens = len(ga1_survival) if ga1_survival else len(ga2_survival)
        gen_ranges = range(0, total_gens, max(1, total_gens // 10))
        
        ga1_extinct_prob = []
        ga2_extinct_prob = []
        
        for i, start in enumerate(gen_ranges[:-1]):
            end = gen_ranges[i + 1]
            ga1_extinct_in_range = len([g for g in ga1_extinctions if start <= g < end])
            ga2_extinct_in_range = len([g for g in ga2_extinctions if start <= g < end])
            
            ga1_extinct_prob.append(ga1_extinct_in_range / (end - start) * 100)
            ga2_extinct_prob.append(ga2_extinct_in_range / (end - start) * 100)
        
        if ga1_extinct_prob:
            ax3.bar(range(len(ga1_extinct_prob)), ga1_extinct_prob, alpha=0.7, label='GA1', color='blue')
        if ga2_extinct_prob:
            ax3.bar(range(len(ga2_extinct_prob)), ga2_extinct_prob, alpha=0.7, label='GA2', color='red')
        
        ax3.set_title('Extinction Probability by Generation Range')
        ax3.set_xlabel('Generation Range')
        ax3.set_ylabel('Extinction Probability (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Population stability
    ax4 = axes[1, 1]
    if ga1_survival and ga2_survival:
        # Calculate coefficient of variation (stability measure)
        ga1_cv = np.std(ga1_survival) / np.mean(ga1_survival) if np.mean(ga1_survival) > 0 else 0
        ga2_cv = np.std(ga2_survival) / np.mean(ga2_survival) if np.mean(ga2_survival) > 0 else 0
        
        stability_scores = [1 / (1 + ga1_cv), 1 / (1 + ga2_cv)]  # Higher = more stable
        strategies = ['GA1\n(Cooperative)', 'GA2\n(Aggressive)']
        colors = ['blue', 'red']
        
        bars = ax4.bar(strategies, stability_scores, color=colors, alpha=0.7)
        ax4.set_title('Population Stability Score')
        ax4.set_ylabel('Stability (Higher = More Stable)')
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, stability_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Survival analysis saved to: {save_path}")
    plt.show()

def create_genome_evolution_plots(data: Dict[str, Any], save_path: str):
    """Create genome evolution analysis plots."""
    trait_data = data['trait_evolution']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Genome Evolution Analysis', fontsize=16, fontweight='bold')
    
    traits_to_plot = ['food_priority', 'aggression', 'energy_conservation', 
                      'risk_tolerance', 'attack_threshold', 'flee_threshold']
    
    for i, trait in enumerate(traits_to_plot):
        if i >= 6:  # Only plot first 6 traits
            break
            
        ax = axes[i // 3, i % 3]
        
        if trait in trait_data.get('GA1', {}):
            ga1_values = trait_data['GA1'][trait]
            ax.plot(range(len(ga1_values)), ga1_values, 'b-', label='GA1', linewidth=2, marker='o', markersize=3)
        
        if trait in trait_data.get('GA2', {}):
            ga2_values = trait_data['GA2'][trait]
            ax.plot(range(len(ga2_values)), ga2_values, 'r-', label='GA2', linewidth=2, marker='s', markersize=3)
        
        ax.set_title(f'{trait.replace("_", " ").title()} Evolution')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Trait Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Genome evolution plots saved to: {save_path}")
    plt.show()

def create_behavior_analysis(data: Dict[str, Any], save_path: str):
    """Create emergent behavior analysis."""
    behavior_data = data['behavior_patterns']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Emergent Behavior Analysis', fontsize=16, fontweight='bold')
    
    behaviors = ['exploration_rate', 'food_efficiency', 'energy_management']
    
    for i, behavior in enumerate(behaviors):
        if i >= 3:
            break
            
        ax = axes[i // 2, i % 2]
        
        if behavior in behavior_data.get('GA1', {}):
            ga1_values = behavior_data['GA1'][behavior]
            ax.plot(range(len(ga1_values)), ga1_values, 'b-', label='GA1', linewidth=2, marker='o', markersize=4)
        
        if behavior in behavior_data.get('GA2', {}):
            ga2_values = behavior_data['GA2'][behavior]
            ax.plot(range(len(ga2_values)), ga2_values, 'r-', label='GA2', linewidth=2, marker='s', markersize=4)
        
        ax.set_title(f'{behavior.replace("_", " ").title()} Evolution')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Behavior Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Summary comparison in the 4th subplot
    ax4 = axes[1, 1]
    if all(behavior in behavior_data.get('GA1', {}) and behavior in behavior_data.get('GA2', {}) 
           for behavior in behaviors):
        
        ga1_final = [behavior_data['GA1'][b][-1] if behavior_data['GA1'][b] else 0 for b in behaviors]
        ga2_final = [behavior_data['GA2'][b][-1] if behavior_data['GA2'][b] else 0 for b in behaviors]
        
        x = np.arange(len(behaviors))
        width = 0.35
        
        ax4.bar(x - width/2, ga1_final, width, label='GA1', color='blue', alpha=0.7)
        ax4.bar(x + width/2, ga2_final, width, label='GA2', color='red', alpha=0.7)
        
        ax4.set_title('Final Behavior Comparison')
        ax4.set_ylabel('Behavior Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels([b.replace('_', '\n') for b in behaviors])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Behavior analysis saved to: {save_path}")
    plt.show()

def generate_text_report(data: Dict[str, Any]) -> str:
    """Generate comprehensive text report."""
    if not data:
        return "No data available for analysis."
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE EVOLUTIONARY SIMULATION ANALYSIS")
    report.append("="*80)
    
    # Basic simulation info
    config = data.get('simulation_config', {})
    report.append(f"\nSIMULATION CONFIGURATION:")
    report.append(f"Population Size: {config.get('population_size', 'Unknown')}")
    report.append(f"World Size: {config.get('world_size', 'Unknown')}")
    report.append(f"Food Count: {config.get('food_count', 'Unknown')}")
    report.append(f"Hazard Count: {config.get('hazard_count', 'Unknown')}")
    
    # Performance summary
    summary = data.get('summary_statistics', {})
    report.append(f"\nPERFORMANCE SUMMARY:")
    report.append(f"Total Generations: {summary.get('total_generations', 0)}")
    report.append(f"GA1 Average Performance: {summary.get('ga1_avg_performance', 0):.2f}")
    report.append(f"GA2 Average Performance: {summary.get('ga2_avg_performance', 0):.2f}")
    report.append(f"GA1 Best Performance: {summary.get('ga1_best_performance', 0):.2f}")
    report.append(f"GA2 Best Performance: {summary.get('ga2_best_performance', 0):.2f}")
    
    # Survival analysis
    if 'survival_history' in data:
        ga1_survival = data['survival_history']['GA1']
        ga2_survival = data['survival_history']['GA2']
        
        report.append(f"\nSURVIVAL ANALYSIS:")
        if ga1_survival:
            report.append(f"GA1 Average Survivors: {np.mean(ga1_survival):.1f}")
            report.append(f"GA1 Best Generation: {max(ga1_survival)} survivors")
            report.append(f"GA1 Survival Rate: {np.mean(ga1_survival)/config.get('population_size', 50)*100:.1f}%")
        
        if ga2_survival:
            report.append(f"GA2 Average Survivors: {np.mean(ga2_survival):.1f}")
            report.append(f"GA2 Best Generation: {max(ga2_survival)} survivors")
            report.append(f"GA2 Survival Rate: {np.mean(ga2_survival)/config.get('population_size', 50)*100:.1f}%")
    
    # Extinction analysis
    if 'extinction_events' in data:
        report.append(f"\nEXTINCTION ANALYSIS:")
        report.append(f"GA1 Extinction Events: {len(data['extinction_events']['GA1'])}")
        report.append(f"GA2 Extinction Events: {len(data['extinction_events']['GA2'])}")
    
    report.append("="*80)
    return "\n".join(report)

if __name__ == "__main__":
    # Load and analyze data
    data = load_simulation_data()
    if data:
        # Generate comprehensive analysis
        create_comprehensive_analysis(data)
        
        # Print text report
        print("\n" + generate_text_report(data))
        
        print("\nAnalysis complete! Check the generated image files for visualizations.") 
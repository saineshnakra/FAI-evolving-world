"""
Neural Network Hyperparameter Optimizer for Genetic Algorithm Simulation
=========================================================================

This module implements neural networks to optimize hyperparameters for achieving
three different evolutionary objectives:
1. GA1 Dominance: Train hyperparameters to make cooperative agents dominate
2. GA2 Dominance: Train hyperparameters to make aggressive agents dominate  
3. Coexistence: Train hyperparameters to achieve stable coexistence

The system uses deep reinforcement learning to discover optimal simulation parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from datetime import datetime

from config import config


@dataclass
class HyperparameterSet:
    """
    Container for all hyperparameters that can be optimized by neural networks.
    """
    # Population parameters
    population_size_ga1: int
    population_size_ga2: int
    
    # Environment parameters
    food_count: int
    hazard_count: int
    food_spawn_rate: float
    
    # Energy parameters
    agent_energy: int
    movement_cost: float
    eating_reward: int
    
    # Genetic algorithm parameters
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    
    def to_tensor(self) -> torch.Tensor:
        """Convert hyperparameters to normalized tensor for neural network input."""
        params = [
            self.population_size_ga1 / 200.0,  # Normalize to 0-1 range
            self.population_size_ga2 / 50.0,
            self.food_count / 300.0,
            self.hazard_count / 20.0,
            self.food_spawn_rate,  # Already 0-1
            self.agent_energy / 300.0,
            self.movement_cost / 2.0,
            self.eating_reward / 50.0,
            self.mutation_rate,  # Already 0-1
            self.crossover_rate,  # Already 0-1
            self.tournament_size / 5.0
        ]
        return torch.tensor(params, dtype=torch.float32)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'HyperparameterSet':
        """Create hyperparameter set from normalized tensor."""
        return cls(
            population_size_ga1=int(tensor[0] * 200.0),
            population_size_ga2=int(tensor[1] * 50.0),
            food_count=int(tensor[2] * 300.0),
            hazard_count=int(tensor[3] * 20.0),
            food_spawn_rate=float(tensor[4]),
            agent_energy=int(tensor[5] * 300.0),
            movement_cost=float(tensor[6] * 2.0),
            eating_reward=int(tensor[7] * 50.0),
            mutation_rate=float(tensor[8]),
            crossover_rate=float(tensor[9]),
            tournament_size=int(tensor[10] * 5.0)
        )


class ObjectiveNetwork(nn.Module):
    """
    Neural network that learns to predict optimal hyperparameters for a specific objective.
    """
    
    def __init__(self, input_size: int = 11, hidden_size: int = 64, output_size: int = 11):
        super(ObjectiveNetwork, self).__init__()
        
        # Multi-layer neural network with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()  # Output between 0 and 1 for normalized parameters
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class HyperparameterOptimizer:
    """
    Main optimizer that manages three neural networks for different objectives.
    """
    
    def __init__(self, learning_rate: float = 0.001):
        """Initialize the three objective networks."""
        print("Initializing Neural Hyperparameter Optimizer...")
        
        # Three networks for three objectives
        self.ga1_dominance_net = ObjectiveNetwork()
        self.ga2_dominance_net = ObjectiveNetwork()
        self.coexistence_net = ObjectiveNetwork()
        
        # Optimizers for each network
        self.ga1_optimizer = optim.Adam(self.ga1_dominance_net.parameters(), lr=learning_rate)
        self.ga2_optimizer = optim.Adam(self.ga2_dominance_net.parameters(), lr=learning_rate)
        self.coexist_optimizer = optim.Adam(self.coexistence_net.parameters(), lr=learning_rate)
        
        # Training history
        self.training_history = {
            'ga1_dominance': [],
            'ga2_dominance': [],
            'coexistence': []
        }
        
        # Current best parameters for each objective
        self.best_params = {
            'ga1_dominance': None,
            'ga2_dominance': None,
            'coexistence': None
        }
        
        print("✓ GA1 Dominance Network initialized")
        print("✓ GA2 Dominance Network initialized") 
        print("✓ Coexistence Network initialized")
    
    def generate_hyperparameters(self, objective: str, context: Optional[torch.Tensor] = None) -> HyperparameterSet:
        """
        Generate hyperparameters for a specific objective.
        
        Args:
            objective: 'ga1_dominance', 'ga2_dominance', or 'coexistence'
            context: Optional context tensor (current simulation state)
            
        Returns:
            HyperparameterSet optimized for the objective
        """
        if context is None:
            # Use random context if none provided
            context = torch.randn(11)
        
        # Select appropriate network
        if objective == 'ga1_dominance':
            network = self.ga1_dominance_net
        elif objective == 'ga2_dominance':
            network = self.ga2_dominance_net
        elif objective == 'coexistence':
            network = self.coexistence_net
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Generate parameters
        with torch.no_grad():
            network.eval()
            normalized_params = network(context)
            params = HyperparameterSet.from_tensor(normalized_params)
        
        return params
    
    def train_on_simulation_result(self, objective: str, hyperparams: HyperparameterSet, 
                                 simulation_outcome: Dict[str, float]):
        """
        Train the network based on simulation results.
        
        Args:
            objective: Which objective was being optimized
            hyperparams: The hyperparameters used
            simulation_outcome: Results from the simulation
        """
        # Calculate reward based on objective
        reward = self._calculate_reward(objective, simulation_outcome)
        
        # Convert to tensors
        input_tensor = hyperparams.to_tensor()
        target_reward = torch.tensor([reward], dtype=torch.float32)
        
        # Select network and optimizer
        if objective == 'ga1_dominance':
            network = self.ga1_dominance_net
            optimizer = self.ga1_optimizer
        elif objective == 'ga2_dominance':
            network = self.ga2_dominance_net
            optimizer = self.ga2_optimizer
        elif objective == 'coexistence':
            network = self.coexistence_net
            optimizer = self.coexist_optimizer
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Training step
        network.train()
        optimizer.zero_grad()
        
        # Forward pass
        predicted_params = network(input_tensor)
        
        # Calculate loss (difference between predicted and actual parameters weighted by reward)
        loss = F.mse_loss(predicted_params, input_tensor) * (1.0 - reward)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record training history
        self.training_history[objective].append({
            'loss': loss.item(),
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update best parameters if this is the best result so far
        if (self.best_params[objective] is None or 
            reward > max([h['reward'] for h in self.training_history[objective][:-1]], default=0)):
            self.best_params[objective] = hyperparams
    
    def _calculate_reward(self, objective: str, outcome: Dict[str, float]) -> float:
        """
        Calculate reward based on simulation outcome and objective.
        
        Args:
            objective: The objective being optimized
            outcome: Dictionary with simulation results
            
        Returns:
            Reward value between 0 and 1
        """
        ga1_survival = outcome.get('ga1_survival_rate', 0.0)
        ga2_survival = outcome.get('ga2_survival_rate', 0.0)
        total_generations = outcome.get('generations_completed', 1)
        diversity = outcome.get('genetic_diversity', 0.0)
        
        if objective == 'ga1_dominance':
            # Reward GA1 having high survival and GA2 having low survival
            dominance_ratio = ga1_survival / max(ga2_survival, 0.1)  # Avoid division by zero
            return min(dominance_ratio / 5.0, 1.0)  # Normalize to 0-1
            
        elif objective == 'ga2_dominance':
            # Reward GA2 having high survival and GA1 having low survival
            dominance_ratio = ga2_survival / max(ga1_survival, 0.1)
            return min(dominance_ratio / 5.0, 1.0)
            
        elif objective == 'coexistence':
            # Reward both populations surviving with good diversity
            balance = 1.0 - abs(ga1_survival - ga2_survival)  # Penalize imbalance
            survival_bonus = min(ga1_survival + ga2_survival, 1.0)
            diversity_bonus = diversity
            longevity_bonus = min(total_generations / 100.0, 1.0)
            
            return (balance * 0.4 + survival_bonus * 0.3 + diversity_bonus * 0.2 + longevity_bonus * 0.1)
        
        return 0.0
    
    def save_models(self, filepath_prefix: str):
        """Save all trained models."""
        torch.save(self.ga1_dominance_net.state_dict(), f"{filepath_prefix}_ga1_dominance.pth")
        torch.save(self.ga2_dominance_net.state_dict(), f"{filepath_prefix}_ga2_dominance.pth")
        torch.save(self.coexistence_net.state_dict(), f"{filepath_prefix}_coexistence.pth")
        
        # Save training history
        with open(f"{filepath_prefix}_training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"✓ Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load pre-trained models."""
        try:
            self.ga1_dominance_net.load_state_dict(torch.load(f"{filepath_prefix}_ga1_dominance.pth"))
            self.ga2_dominance_net.load_state_dict(torch.load(f"{filepath_prefix}_ga2_dominance.pth"))
            self.coexistence_net.load_state_dict(torch.load(f"{filepath_prefix}_coexistence.pth"))
            
            # Load training history
            with open(f"{filepath_prefix}_training_history.json", 'r') as f:
                self.training_history = json.load(f)
            
            print(f"✓ Models loaded from prefix: {filepath_prefix}")
            return True
        except FileNotFoundError:
            print(f"⚠ No saved models found with prefix: {filepath_prefix}")
            return False
    
    def get_training_summary(self) -> str:
        """Generate a summary of training progress."""
        summary = "Neural Hyperparameter Optimizer Training Summary\n"
        summary += "=" * 50 + "\n\n"
        
        for objective in ['ga1_dominance', 'ga2_dominance', 'coexistence']:
            history = self.training_history[objective]
            if history:
                best_reward = max(h['reward'] for h in history)
                avg_reward = np.mean([h['reward'] for h in history])
                latest_reward = history[-1]['reward']
                
                summary += f"{objective.upper()}:\n"
                summary += f"  Training Episodes: {len(history)}\n"
                summary += f"  Best Reward: {best_reward:.4f}\n"
                summary += f"  Average Reward: {avg_reward:.4f}\n"
                summary += f"  Latest Reward: {latest_reward:.4f}\n\n"
            else:
                summary += f"{objective.upper()}: No training data\n\n"
        
        return summary


def create_training_visualization(optimizer: HyperparameterOptimizer, save_path: str = "neural_training_progress.png"):
    """
    Create a visualization of the neural network training progress.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Neural Hyperparameter Optimizer Training Progress', fontsize=16)
    
    objectives = ['ga1_dominance', 'ga2_dominance', 'coexistence']
    colors = ['blue', 'orange', 'green']
    
    # Plot 1: Reward progression for each objective
    ax1 = axes[0, 0]
    for obj, color in zip(objectives, colors):
        history = optimizer.training_history[obj]
        if history:
            rewards = [h['reward'] for h in history]
            ax1.plot(rewards, label=obj.replace('_', ' ').title(), color=color, alpha=0.7)
    
    ax1.set_title('Reward Progression')
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss progression
    ax2 = axes[0, 1]
    for obj, color in zip(objectives, colors):
        history = optimizer.training_history[obj]
        if history:
            losses = [h['loss'] for h in history]
            ax2.plot(losses, label=obj.replace('_', ' ').title(), color=color, alpha=0.7)
    
    ax2.set_title('Loss Progression')
    ax2.set_xlabel('Training Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward distribution
    ax3 = axes[1, 0]
    all_rewards = []
    labels = []
    for obj in objectives:
        history = optimizer.training_history[obj]
        if history:
            rewards = [h['reward'] for h in history]
            all_rewards.append(rewards)
            labels.append(obj.replace('_', ' ').title())
    
    if all_rewards:
        ax3.boxplot(all_rewards, labels=labels)
    ax3.set_title('Reward Distribution by Objective')
    ax3.set_ylabel('Reward')
    
    # Plot 4: Training timeline
    ax4 = axes[1, 1]
    total_episodes = sum(len(optimizer.training_history[obj]) for obj in objectives)
    if total_episodes > 0:
        episode_counts = [len(optimizer.training_history[obj]) for obj in objectives]
        ax4.pie(episode_counts, labels=[obj.replace('_', ' ').title() for obj in objectives], 
                colors=colors, autopct='%1.1f%%')
    ax4.set_title('Training Episodes by Objective')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training visualization saved to: {save_path}") 